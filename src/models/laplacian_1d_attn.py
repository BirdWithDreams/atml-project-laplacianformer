import torch
import torch.nn as nn
import torch.nn.functional as F

from .laplacian_1d_cuda_ops import (
    can_use_laplacian_1d_cuda,
    extension_diagnostics,
    laplacian_kernel_1d_cuda,
)
from .laplacian_cuda_ops import (
    can_use_laplacian_cuda as can_use_newton_cuda,
    extension_diagnostics as newton_extension_diagnostics,
    newton_inverse_cuda,
)


class CudaNewtonSchulzInverse(nn.Module):
    """Newton inverse backed by the original LaplacianFormer CUDA extension."""

    def __init__(self, num_iters: int = 5):
        super().__init__()
        self.num_iters = num_iters

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        matrix_size = W.shape[-1]
        if can_use_newton_cuda(W, matrix_size=matrix_size):
            return newton_inverse_cuda(W.contiguous(), self.num_iters)

        raise RuntimeError(
            "1D Laplacian attention requires the CUDA Newton inverse. "
            f"shape={tuple(W.shape)}, dtype={W.dtype}, device={W.device}. "
            f"{newton_extension_diagnostics()}"
        )


class CudaLaplacian1DLinearAttention(nn.Module):
    """1D Laplacian attention for text using the custom CUDA distance kernel."""

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            lambda_scale: float = 4.0,
            pool_ratio: int = 2,
            ns_iters: int = 5,
            ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.lambda_scale = lambda_scale
        self.pool_ratio = pool_ratio

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.inverse_solver = CudaNewtonSchulzInverse(num_iters=ns_iters)
        self.dwc = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.white_eps = 1e-5

    def laplacian_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if can_use_laplacian_1d_cuda(x):
            return laplacian_kernel_1d_cuda(x, y, self.lambda_scale)

        raise RuntimeError(
            "1D Laplacian attention requires the custom CUDA kernel. "
            f"x_shape={tuple(x.shape)}, y_shape={tuple(y.shape)}, "
            f"dtype={x.dtype}, device={x.device}. {extension_diagnostics()}"
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape
        if attention_mask is None:
            token_mask = torch.ones(batch_size, seq_len, device=x.device, dtype=x.dtype)
        else:
            if attention_mask.shape != (batch_size, seq_len):
                raise ValueError(
                    f"Expected attention_mask shape {(batch_size, seq_len)}, "
                    f"got {tuple(attention_mask.shape)}"
                )
            token_mask = attention_mask.to(dtype=x.dtype)

        effective_pool = min(self.pool_ratio, seq_len)
        token_mask_seq = token_mask.unsqueeze(1)
        token_mask_heads = token_mask.unsqueeze(1).unsqueeze(-1)

        qkv = self.qkv(x).reshape(
            batch_size,
            seq_len,
            3,
            self.num_heads,
            self.head_dim,
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * token_mask_heads
        k = k * token_mask_heads
        v = v * token_mask_heads

        v_spatial = v.transpose(1, 2).reshape(batch_size, channels, seq_len)
        v_local = self.dwc(v_spatial).transpose(1, 2)
        v_local = v_local * token_mask.unsqueeze(-1)

        q_spatial = q.transpose(1, 2).reshape(batch_size, channels, seq_len)
        k_spatial = k.transpose(1, 2).reshape(batch_size, channels, seq_len)
        pooled_mask = F.avg_pool1d(
            token_mask_seq,
            effective_pool,
            effective_pool,
            ceil_mode=True,
        )
        pooled_mask_safe = pooled_mask.clamp_min(1e-6)

        q_pool = F.avg_pool1d(
            q_spatial * token_mask_seq,
            effective_pool,
            effective_pool,
            ceil_mode=True,
        ) / pooled_mask_safe
        k_pool = F.avg_pool1d(
            k_spatial * token_mask_seq,
            effective_pool,
            effective_pool,
            ceil_mode=True,
        ) / pooled_mask_safe
        landmark_mask = (pooled_mask > 0).to(dtype=x.dtype).squeeze(1)

        q_landmark = q_pool.transpose(1, 2).reshape(
            batch_size,
            -1,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        k_landmark = k_pool.transpose(1, 2).reshape(
            batch_size,
            -1,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        W = self.laplacian_kernel(q_landmark, k_landmark)
        num_landmarks = W.shape[-1]
        landmark_pair_mask = landmark_mask[:, None, :, None] * landmark_mask[:, None, None, :]
        identity = torch.eye(num_landmarks, device=x.device, dtype=x.dtype).view(
            1,
            1,
            num_landmarks,
            num_landmarks,
        )
        W = W * landmark_pair_mask + identity * (1 - landmark_pair_mask)
        W_inv = self.inverse_solver(W)

        landmark_attn_mask = landmark_mask[:, None, None, :]
        C_q = self.laplacian_kernel(q, k_landmark) * landmark_attn_mask
        C_k = self.laplacian_kernel(k, q_landmark) * landmark_attn_mask

        query_mask = token_mask[:, None, :, None]
        valid_queries = query_mask.sum(dim=(0, 2), keepdim=True).clamp_min(1.0)
        mu = (C_q * query_mask).sum(dim=(0, 2), keepdim=True) / valid_queries
        centered = (C_q - mu) * query_mask
        var = centered.pow(2).sum(dim=(0, 2), keepdim=True) / valid_queries
        C_q_norm = (C_q - mu) / torch.sqrt(var + self.white_eps)
        C_q_norm = C_q_norm * query_mask

        context = torch.matmul(C_k.transpose(-2, -1), v)
        context = torch.matmul(W_inv, context)
        global_attn = torch.matmul(C_q_norm, context)
        global_attn = global_attn * token_mask_heads

        global_attn = global_attn.transpose(1, 2).reshape(batch_size, seq_len, channels)
        output = self.proj(global_attn + v_local)
        return output * token_mask.unsqueeze(-1)
