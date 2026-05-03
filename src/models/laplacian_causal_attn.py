import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.laplacian_attn import NewtonSchulzInverse

class CausalLaplacianLinearAttention(nn.Module):
    """
    Pure PyTorch implementation of 1D Laplacian Attention WITH Causal Masking.
    Required for Autoregressive text generation.
    """
    def __init__(
            self, 
            dim: int, 
            num_heads: int = 8, 
            lambda_scale: float = 4.0,
            ns_iters: int = 5,
        ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.lambda_scale = lambda_scale
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.inverse_solver = NewtonSchulzInverse(num_iters=ns_iters)
        
        # We disable 1D pooling (pool_ratio=1) for causal generation because 
        # pooling ahead leaks future information into current landmarks.
        self.white_eps = 1e-5

    def laplacian_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: (B, H, N, d), y: (B, H, M, d) -> Returns: (B, H, N, M)
        diff = x.unsqueeze(-2) - y.unsqueeze(-3) 
        l1_dist = torch.norm(diff, p=1, dim=-1)  
        return torch.exp(-l1_dist / self.lambda_scale)

    def forward(self, x: torch.Tensor, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape
        
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 1. Compute full kernel
        W = self.laplacian_kernel(q, k) # (B, H, N, N)
        
        # 2. Apply Causal Masking (Lower Triangular)
        if tgt_mask is None:
            tgt_mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).view(1, 1, seq_len, seq_len)
        else:
            # Ensure it broadcasts to (B, H, N, N)
            if tgt_mask.dim() == 3:
                tgt_mask = tgt_mask.unsqueeze(1)
        
        # Mask out future tokens by setting their distance/similarity to 0
        W = W * tgt_mask

        # 3. Inverse and Attention Assembly
        W_inv = self.inverse_solver(W) 
        
        # Because we aren't pooling (landmarks = all tokens), C_q and C_k are just W
        C_q = W
        C_k = W
        
        mu = C_q.mean(dim=(0, 2), keepdim=True)  
        var = C_q.var(dim=(0, 2), keepdim=True)  
        C_q_norm = (C_q - mu) / torch.sqrt(var + self.white_eps)

        context = torch.matmul(C_k.transpose(-2, -1), v) 
        context = torch.matmul(W_inv, context)  
        global_attn = torch.matmul(C_q_norm, context)  

        global_attn = global_attn.transpose(1, 2).reshape(batch_size, seq_len, channels)
        output = self.proj(global_attn)
        return output