"""Python wrappers for the 1D Laplacian attention CUDA extension."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch


_EXTENSION_NAME = "laplacian_1d_cuda"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXTENSION_DIR = _REPO_ROOT / "libs" / "laplacianformer_1d"
_SUPPORTED_DTYPES = {torch.float32, torch.float64}

_EXTENSION = None
_IMPORT_ERROR: Exception | None = None


def _load_extension():
    global _EXTENSION, _IMPORT_ERROR
    if _EXTENSION is not None:
        return _EXTENSION
    if _IMPORT_ERROR is not None:
        return None

    if _EXTENSION_DIR.exists():
        extension_dir = str(_EXTENSION_DIR)
        if extension_dir not in sys.path:
            sys.path.insert(0, extension_dir)

    try:
        _EXTENSION = importlib.import_module(_EXTENSION_NAME)
    except Exception as exc:  # pragma: no cover - depends on local CUDA build
        _IMPORT_ERROR = exc
        return None
    return _EXTENSION


def is_laplacian_1d_cuda_available() -> bool:
    return _load_extension() is not None


def extension_diagnostics() -> str:
    if _load_extension() is not None:
        return f"{_EXTENSION_NAME} loaded"

    detail = f"{type(_IMPORT_ERROR).__name__}: {_IMPORT_ERROR}" if _IMPORT_ERROR else "not imported"
    return (
        f"{_EXTENSION_NAME} is unavailable ({detail}). Build it with "
        "`cd libs/laplacianformer_1d && uv run python setup.py build_ext --inplace`."
    )


def can_use_laplacian_1d_cuda(tensor: torch.Tensor) -> bool:
    return (
        _load_extension() is not None
        and tensor.is_cuda
        and tensor.dtype in _SUPPORTED_DTYPES
    )


class _Laplacian1DDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        ext = _load_extension()
        if ext is None:
            raise RuntimeError(extension_diagnostics())
        if not query.is_cuda or not key.is_cuda:
            raise RuntimeError("1D Laplacian CUDA path requires CUDA tensors.")
        if query.dtype not in _SUPPORTED_DTYPES:
            raise RuntimeError(
                "1D Laplacian CUDA path supports float32 and float64. "
                f"Got {query.dtype}; use trainer.precision=32."
            )
        if query.shape[:2] != key.shape[:2] or query.shape[-1] != key.shape[-1]:
            raise ValueError(
                "Expected query/key shapes (B,H,N,D) and (B,H,M,D); "
                f"got {tuple(query.shape)} and {tuple(key.shape)}."
            )

        query_c = query.contiguous()
        key_c = key.contiguous()
        output = torch.empty(
            (*query.shape[:2], query.shape[-2], key.shape[-2]),
            device=query.device,
            dtype=query.dtype,
        )
        ext.laplacian_1d_forward(query_c, key_c, output)
        ctx.save_for_backward(query_c, key_c)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        ext = _load_extension()
        if ext is None:
            raise RuntimeError(extension_diagnostics())

        query, key = ctx.saved_tensors
        grad_query = torch.empty_like(query)
        grad_key = torch.empty_like(key)
        ext.laplacian_1d_backward(
            grad_output.contiguous(),
            query,
            key,
            grad_query,
            grad_key,
        )
        return grad_query, grad_key


def laplace_l1_distance_1d_cuda(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    return _Laplacian1DDistanceFunction.apply(query, key)


def laplacian_kernel_1d_cuda(
        query: torch.Tensor,
        key: torch.Tensor,
        lambda_scale: float,
        ) -> torch.Tensor:
    distances = laplace_l1_distance_1d_cuda(query, key)
    return torch.exp(-distances / lambda_scale)
