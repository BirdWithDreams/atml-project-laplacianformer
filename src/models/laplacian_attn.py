import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_2d_rope


class NewtonSchulzInverse(nn.Module):
    """
    Computes the pseudo-inverse of a positive semi-definite matrix
    using the Newton-Schulz iteration[cite: 240].
    """

    def __init__(self, num_iters: int = 5, eps: float = 1e-4):
        super().__init__()
        self.num_iters = num_iters
        self.eps = eps

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        # W shape: (B, H, m, m)
        orig_dtype = W.dtype
        device_type = W.device.type

        # Keep the inverse solve in fp32 even under mixed precision autocast.
        with torch.amp.autocast(device_type=device_type, enabled=False):
            W = W.float()
            B, H, m, _ = W.shape
            I = torch.eye(m, device=W.device, dtype=W.dtype).view(1, 1, m, m)

            # Add small diagonal perturbation for strictly positive definite constraint [cite: 246]
            W_eps = W + self.eps * I

            # Initialize scaling factor alpha = 2 / ||W||_2 [cite: 247]
            # Using Frobenius norm as a differentiable proxy for spectral norm
            alpha = 2.0 / (torch.linalg.norm(W_eps, dim=(-2, -1), keepdim=True) + 1e-8)

            # Initialize X_0 = alpha * W^T [cite: 248]
            X = alpha * W_eps.transpose(-2, -1)

            # Iterative update: X_{k+1} = X_k (2I - W X_k) [cite: 252]
            for _ in range(self.num_iters):
                X = X @ (2 * I - W_eps @ X)

            residual = torch.linalg.norm(I - W_eps @ X, dim=(-2, -1))
            invalid = (
                ~torch.isfinite(X).all(dim=(-2, -1))
                | ~torch.isfinite(residual)
                | (residual > 1.0)
            )

            if invalid.any():
                X = X.clone()
                invalid_w = W_eps[invalid]
                X[invalid] = torch.linalg.pinv(invalid_w).to(dtype=X.dtype)

        return X.to(orig_dtype)


class LaplacianLinearAttention(nn.Module):
    """
    The core Laplacian attention mechanism from the LaplacianFormer paper.
    """

    def __init__(
            self, dim: int, num_heads: int = 8, lambda_scale: float = 4.0,
            pool_ratio: int = 2, ns_iters: int = 5, use_rope: bool = False,
            rope_base: float = 10000.0
            ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Paper explicitly sets lambda to 4 as the optimal scale
        self.lambda_scale = lambda_scale
        self.pool_ratio = pool_ratio
        self.use_rope = use_rope
        self.rope_base = rope_base

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # Newton-Schulz solver for the landmark kernel inverse [cite: 240]
        self.inverse_solver = NewtonSchulzInverse(num_iters=ns_iters)

        # Depth-wise convolution for local context V modeling
        self.dwc = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

        # Small epsilon for whitening numerical stability [cite: 212]
        self.white_eps = 1e-5

    def laplacian_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes exp(-||x - y||_1 / lambda)[cite: 60].
        Shapes: x -> (B, H, N, d), y -> (B, H, M, d)
        Returns: (B, H, N, M)
        """
        # Efficient L1 distance via broadcasting
        diff = x.unsqueeze(-2) - y.unsqueeze(-3)  # (B, H, N, M, d)
        l1_dist = torch.norm(diff, p=1, dim=-1)  # (B, H, N, M)
        return torch.exp(-l1_dist / self.lambda_scale)

    def forward(self, x: torch.Tensor, H_sp: int, W_sp: int) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, N, C) where N = H_sp * W_sp
            H_sp: Spatial height of the feature map
            W_sp: Spatial width of the feature map
        """
        B, N, C = x.shape
        if N != H_sp * W_sp:
            raise ValueError(f"Expected N == H_sp * W_sp, got {N} vs {H_sp} * {W_sp}")

        # 1. Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Shapes: (B, H, N, d)
        if self.use_rope:
            q = apply_2d_rope(q, H_sp, W_sp, self.rope_base)
            k = apply_2d_rope(k, H_sp, W_sp, self.rope_base)

        effective_pool = min(self.pool_ratio, H_sp, W_sp)

        # 2. Local Context via DWC on V [cite: 226, 228]
        # Reshape V to spatial dimensions, apply DWC, and flatten back
        v_spatial = v.transpose(1, 2).reshape(B, C, H_sp, W_sp)
        v_local = self.dwc(v_spatial).flatten(2).transpose(1, 2)  # (B, N, C)

        # 3. Landmark Selection via Spatial Pooling [cite: 295, 296]
        # We pool Q and K to get the landmark set of size m
        q_spatial = q.transpose(1, 2).reshape(B, C, H_sp, W_sp)
        k_spatial = k.transpose(1, 2).reshape(B, C, H_sp, W_sp)

        q_pool = F.avg_pool2d(q_spatial, effective_pool, effective_pool)
        k_pool = F.avg_pool2d(k_spatial, effective_pool, effective_pool)

        q_landmark = q_pool.flatten(2).transpose(1, 2).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k_landmark = k_pool.flatten(2).transpose(1, 2).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # q_landmark, k_landmark shape: (B, H, m, d)

        # 4. Nyström Kernel Computations [cite: 231, 238]
        # W = k(Q_landmark, K_landmark)
        W = self.laplacian_kernel(q_landmark, k_landmark)  # (B, H, m, m)
        W_inv = self.inverse_solver(W)  # (B, H, m, m)

        # C_q = k(Q, K_landmark) -> Cross-kernel between all queries and landmarks [cite: 239]
        C_q = self.laplacian_kernel(q, k_landmark)  # (B, H, N, m)

        # C_k = k(K, Q_landmark)
        C_k = self.laplacian_kernel(k, q_landmark)  # (B, H, N, m)

        # 5. Diagonal Whitening Normalization on Queries (Eq 4, 5, 6)
        # The paper normalizes the feature similarity maps to ensure injectivity.
        # We apply this to C_q across the batch/sequence to mimic the diagonal estimator.
        mu = C_q.mean(dim=(0, 2), keepdim=True)  # Mean across Batch and Seq
        var = C_q.var(dim=(0, 2), keepdim=True)  # Var across Batch and Seq
        C_q_norm = (C_q - mu) / torch.sqrt(var + self.white_eps)

        # 6. Linear Attention Assembly
        # Attn = C_q_norm @ W_inv @ C_k^T @ V
        # Computed right-to-left for O(N) complexity
        context = torch.matmul(C_k.transpose(-2, -1), v)  # (B, H, m, d)
        context = torch.matmul(W_inv, context)  # (B, H, m, d)
        global_attn = torch.matmul(C_q_norm, context)  # (B, H, N, d)

        # Reshape and project
        global_attn = global_attn.transpose(1, 2).reshape(B, N, C)

        # Combine Global Attention (Z * V) with Local DWC(V)
        output = self.proj(global_attn + v_local)

        return output


# --- Example Usage ---
if __name__ == "__main__":
    batch_size = 2
    dim = 64
    H_sp, W_sp = 14, 14
    seq_len = H_sp * W_sp

    x = torch.randn(batch_size, seq_len, dim)

    laplacian_attn = LaplacianLinearAttention(
        dim=dim, num_heads=4, lambda_scale=4.0, pool_ratio=2, ns_iters=5
    )

    out = laplacian_attn(x, H_sp, W_sp)
    print(f"2D Input shape:  {x.shape}")
    print(f"2D Output shape: {out.shape}")
