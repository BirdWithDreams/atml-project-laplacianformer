import torch
import torch.nn as nn

from .laplacian_attn import LaplacianLinearAttention, NewtonSchulzInverse
from .laplacian_cuda_ops import (
    can_use_laplacian_cuda,
    extension_diagnostics,
    laplacian_kernel_cuda,
    newton_inverse_cuda,
)


class FastCudaNewtonSchulzInverse(nn.Module):
    """Newton inverse layer backed by the authors' CUDA kernel when possible."""

    def __init__(
            self,
            num_iters: int = 5,
            eps: float = 1e-4,
            fallback_to_torch: bool = True,
            ):
        super().__init__()
        self.num_iters = num_iters
        self.fallback_to_torch = fallback_to_torch
        self.torch_solver = NewtonSchulzInverse(num_iters=num_iters, eps=eps)

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        matrix_size = W.shape[-1]
        if can_use_laplacian_cuda(W, matrix_size=matrix_size):
            return newton_inverse_cuda(W.contiguous(), self.num_iters)

        if self.fallback_to_torch:
            return self.torch_solver(W)

        raise RuntimeError(
            "Fast CUDA Newton inverse is unavailable for this input. "
            f"shape={tuple(W.shape)}, dtype={W.dtype}, device={W.device}. "
            f"{extension_diagnostics()}"
        )


class FastCudaLaplacianLinearAttention(LaplacianLinearAttention):
    """2D Laplacian attention that uses the authors' CUDA kernels as a fast path.

    This class intentionally lives beside the pure PyTorch implementation. It
    keeps the same public forward interface as `LaplacianLinearAttention`, so
    vision backbones can switch with a config flag.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            lambda_scale: float = 4.0,
            pool_ratio: int = 2,
            ns_iters: int = 5,
            use_rope: bool = False,
            rope_base: float = 10000.0,
            fallback_to_torch: bool = True,
            ):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            lambda_scale=lambda_scale,
            pool_ratio=pool_ratio,
            ns_iters=ns_iters,
            use_rope=use_rope,
            rope_base=rope_base,
        )
        self.fallback_to_torch = fallback_to_torch
        self.inverse_solver = FastCudaNewtonSchulzInverse(
            num_iters=ns_iters,
            fallback_to_torch=fallback_to_torch,
        )

    def laplacian_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if can_use_laplacian_cuda(x):
            return laplacian_kernel_cuda(x, y, self.lambda_scale)

        if self.fallback_to_torch:
            return super().laplacian_kernel(x, y)

        raise RuntimeError(
            "Fast CUDA Laplacian kernel is unavailable for this input. "
            f"x_shape={tuple(x.shape)}, y_shape={tuple(y.shape)}, "
            f"dtype={x.dtype}, device={x.device}. {extension_diagnostics()}"
        )
