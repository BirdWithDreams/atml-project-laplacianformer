import torch

from .laplacian_attn import Laplacian1DLinearAttention
from .laplacian_1d_cuda_ops import (
    can_use_laplacian_1d_cuda,
    extension_diagnostics,
    laplacian_kernel_1d_cuda,
)


class FastCudaLaplacian1DLinearAttention(Laplacian1DLinearAttention):
    """1D Laplacian attention using the sequence CUDA distance kernel."""

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            lambda_scale: float = 4.0,
            pool_ratio: int = 2,
            ns_iters: int = 5,
            fallback_to_torch: bool = True,
            ):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            lambda_scale=lambda_scale,
            pool_ratio=pool_ratio,
            ns_iters=ns_iters,
        )
        self.fallback_to_torch = fallback_to_torch

    def laplacian_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if can_use_laplacian_1d_cuda(x):
            return laplacian_kernel_1d_cuda(x, y, self.lambda_scale)

        if self.fallback_to_torch:
            return super().laplacian_kernel(x, y)

        raise RuntimeError(
            "Fast CUDA 1D Laplacian kernel is unavailable for this input. "
            f"x_shape={tuple(x.shape)}, y_shape={tuple(y.shape)}, "
            f"dtype={x.dtype}, device={x.device}. {extension_diagnostics()}"
        )
