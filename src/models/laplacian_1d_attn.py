import math

import torch
import torch.nn as nn

from .laplacian_1d_cuda_ops import (
    can_use_laplacian_1d_cuda,
    exact_laplacian_attention_1d,
    extension_diagnostics,
    nystrom_laplacian_attention_1d,
)


class CudaLaplacian1DLinearAttention(nn.Module):
    """1D Laplacian attention for text using the custom CUDA L1 distance kernel."""

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            lambda_scale: float = 4.0,
            pool_ratio: int = 2,
            ns_iters: int = 5,
            normalization: str = "paper",
            attention_mode: str = "exact",
            use_dwconv: bool = True,
            use_fused_newton: bool = True,
            ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        if normalization not in {"paper", "row"}:
            raise ValueError("normalization must be 'paper' or 'row'")
        if attention_mode not in {"exact", "nystrom"}:
            raise ValueError("attention_mode must be 'exact' or 'nystrom'")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.lambda_scale = lambda_scale
        self.pool_ratio = pool_ratio
        self.ns_iters = ns_iters
        self.normalization = normalization
        self.attention_mode = attention_mode
        self.use_fused_newton = use_fused_newton

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dwc = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim) if use_dwconv else None
        self.eps = 1e-6

    def _dwconv_value(self, value: torch.Tensor, token_mask: torch.Tensor | None) -> torch.Tensor:
        if self.dwc is None:
            return torch.zeros_like(value)

        batch_size, num_heads, seq_len, head_dim = value.shape
        v_1d = value.permute(0, 1, 3, 2).reshape(batch_size, num_heads * head_dim, seq_len)
        local = self.dwc(v_1d)
        local = local.reshape(batch_size, num_heads, head_dim, seq_len).permute(0, 1, 3, 2)
        if token_mask is not None:
            local = local * token_mask[:, None, :, None]
        return local

    def _global_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if not can_use_laplacian_1d_cuda(q):
            raise RuntimeError(
                "1D Laplacian attention requires the custom CUDA distance kernel. "
                f"shape={tuple(q.shape)}, dtype={q.dtype}, device={q.device}. "
                f"{extension_diagnostics()}"
            )

        if self.attention_mode == "exact":
            return exact_laplacian_attention_1d(
                query=q,
                key=k,
                value=v,
                lambda_=self.lambda_scale,
                eps=self.eps,
                normalization=self.normalization,
            )

        if self.normalization != "row":
            raise RuntimeError(
                "Nyström 1D attention currently implements the row-normalized "
                "kernel baseline. Use attention_mode='exact' for paper normalization."
            )

        seq_len = q.shape[-2]
        pool_ratio = max(1, min(self.pool_ratio, seq_len))
        num_landmarks = max(1, math.ceil(seq_len / pool_ratio))
        return nystrom_laplacian_attention_1d(
            query=q,
            key=k,
            value=v,
            num_landmarks=num_landmarks,
            lambda_=self.lambda_scale,
            eps=1e-4,
            ns_iters=self.ns_iters,
            use_fused_newton=self.use_fused_newton,
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape
        if channels != self.dim:
            raise ValueError(f"Expected input dim {self.dim}, got {channels}")

        if attention_mask is None:
            token_mask = None
        else:
            if attention_mask.shape != (batch_size, seq_len):
                raise ValueError(
                    f"Expected attention_mask shape {(batch_size, seq_len)}, "
                    f"got {tuple(attention_mask.shape)}"
                )
            token_mask = attention_mask.to(dtype=x.dtype)

        qkv = self.qkv(x).reshape(
            batch_size,
            seq_len,
            3,
            self.num_heads,
            self.head_dim,
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if token_mask is not None:
            token_mask_heads = token_mask[:, None, :, None]
            q = q * token_mask_heads
            k = k * token_mask_heads
            v = v * token_mask_heads

        global_attn = self._global_attention(q, k, v)
        local_attn = self._dwconv_value(v, token_mask)

        y = global_attn + local_attn
        if token_mask is not None:
            y = y * token_mask[:, None, :, None]

        y = y.transpose(1, 2).reshape(batch_size, seq_len, channels)
        output = self.proj(y)
        if token_mask is not None:
            output = output * token_mask.unsqueeze(-1)
        return output
