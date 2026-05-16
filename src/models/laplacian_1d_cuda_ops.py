import sys
from pathlib import Path

import torch


_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXTENSION_DIR = _REPO_ROOT / "libs" / "laplacianformer_1d"
if _EXTENSION_DIR.exists():
    extension_dir = str(_EXTENSION_DIR)
    if extension_dir not in sys.path:
        sys.path.insert(0, extension_dir)

from laplacian_1d import (  # noqa: E402
    L1PairwiseDistance,
    LaplacianAttention1D,
    avg_pool_landmarks_1d,
    can_use_fused_newton_cuda,
    can_use_laplacian_1d_cuda,
    exact_laplacian_attention_1d,
    extension_diagnostics,
    fused_newton_diagnostics,
    fused_newton_schulz_inverse,
    is_fused_newton_cuda_available,
    is_laplacian_1d_cuda_available,
    l1_pairwise_distance,
    laplacian_attention_1d,
    laplacian_kernel,
    newton_schulz_inverse,
    nystrom_laplacian_attention_1d,
)


def laplace_l1_distance_1d_cuda(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Backward-compatible alias for the CUDA pairwise L1 distance primitive."""
    return l1_pairwise_distance(query, key)


def laplacian_kernel_1d_cuda(
    query: torch.Tensor,
    key: torch.Tensor,
    lambda_scale: float,
) -> torch.Tensor:
    """Backward-compatible alias for exp(-L1(query, key) / lambda_scale)."""
    return laplacian_kernel(query, key, lambda_=lambda_scale)


__all__ = [
    "L1PairwiseDistance",
    "LaplacianAttention1D",
    "avg_pool_landmarks_1d",
    "can_use_fused_newton_cuda",
    "can_use_laplacian_1d_cuda",
    "exact_laplacian_attention_1d",
    "extension_diagnostics",
    "fused_newton_diagnostics",
    "fused_newton_schulz_inverse",
    "is_fused_newton_cuda_available",
    "is_laplacian_1d_cuda_available",
    "l1_pairwise_distance",
    "laplace_l1_distance_1d_cuda",
    "laplacian_attention_1d",
    "laplacian_kernel",
    "laplacian_kernel_1d_cuda",
    "newton_schulz_inverse",
    "nystrom_laplacian_attention_1d",
]
