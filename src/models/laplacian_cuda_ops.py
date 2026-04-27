"""Python wrappers for the authors' LaplacianFormer CUDA extension.

The extension is intentionally optional. Importing this module must be safe on
machines where the custom CUDA operators have not been built yet.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch


_EXTENSION_NAME = "Laplace_subtraction_cuda"
_EXTENSION_DIR = Path(__file__).resolve().parents[1] / "LaplacianFormer"
_SUPPORTED_LAPLACE_DTYPES = {torch.float16, torch.float32, torch.float64}

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


def is_laplacian_cuda_available() -> bool:
    return _load_extension() is not None


def extension_diagnostics() -> str:
    ext = _load_extension()
    if ext is None:
        detail = f"{type(_IMPORT_ERROR).__name__}: {_IMPORT_ERROR}" if _IMPORT_ERROR else "not imported"
        return (
            f"{_EXTENSION_NAME} is unavailable ({detail}). Build it with "
            "`cd src/LaplacianFormer && python setup.py build_ext --inplace`."
        )

    try:
        compiler = ext.get_compiler_version()
        cuda = ext.get_cuda_version()
        has_cuda = ext.has_cuda()
    except Exception as exc:  # pragma: no cover - defensive diagnostics only
        return f"{_EXTENSION_NAME} imported, but diagnostics failed: {exc}"

    return f"{_EXTENSION_NAME} loaded: compiler={compiler}, cuda={cuda}, has_cuda={has_cuda}"


def _require_extension():
    ext = _load_extension()
    if ext is None:
        raise RuntimeError(extension_diagnostics())
    return ext


def can_use_laplacian_cuda(tensor: torch.Tensor, matrix_size: int | None = None) -> bool:
    ext = _load_extension()
    if ext is None or not tensor.is_cuda or tensor.dtype not in _SUPPORTED_LAPLACE_DTYPES:
        return False
    if matrix_size is not None:
        try:
            return bool(ext.NewtonInverse_is_supported(int(matrix_size)))
        except Exception:
            return False
    return True


class _LaplaceSubtractionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        ext = _require_extension()
        if not query.is_cuda or not key.is_cuda:
            raise RuntimeError("LaplaceSubtraction CUDA path requires CUDA tensors.")
        if query.dtype not in _SUPPORTED_LAPLACE_DTYPES:
            raise RuntimeError(
                "LaplaceSubtraction CUDA path supports float16, float32, and float64. "
                f"Got {query.dtype}; use trainer.precision=16-mixed or 32-true, "
                "or allow the PyTorch fallback."
            )
        if query.shape[:2] != key.shape[:2] or query.shape[-1] != key.shape[-1]:
            raise ValueError(
                "Expected query/key shapes (B, H, N, D) and (B, H, M, D); "
                f"got {tuple(query.shape)} and {tuple(key.shape)}."
            )

        query_c = query.contiguous()
        key_t = key.transpose(-2, -1).contiguous()
        output = torch.empty(
            (*query.shape[:2], query.shape[-2], key.shape[-2]),
            device=query.device,
            dtype=query.dtype,
        )
        ext.LaplaceSubtraction_forward(query_c, key_t, output)
        ctx.save_for_backward(query_c, key_t)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        ext = _require_extension()
        query, key_t = ctx.saved_tensors
        grad_output = grad_output.contiguous()

        grad_query = torch.empty_like(query)
        grad_key_t = torch.empty_like(key_t)
        ext.LaplaceSubtraction_backward_query(grad_output, query, key_t, grad_query)
        ext.LaplaceSubtraction_backward_key(grad_output, query, key_t, grad_key_t)
        return grad_query, grad_key_t.transpose(-2, -1).contiguous()


class _NewtonInverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrix: torch.Tensor, num_iters: int) -> torch.Tensor:
        ext = _require_extension()
        if not matrix.is_cuda:
            raise RuntimeError("NewtonInverse CUDA path requires a CUDA tensor.")
        if matrix.dim() != 4 or matrix.shape[-1] != matrix.shape[-2]:
            raise ValueError(f"Expected matrix shape (B, H, N, N), got {tuple(matrix.shape)}.")
        matrix_size = matrix.shape[-1]
        if not bool(ext.NewtonInverse_is_supported(int(matrix_size))):
            raise RuntimeError(f"NewtonInverse CUDA path does not support N={matrix_size}.")

        matrix_c = matrix.contiguous()
        output = torch.empty_like(matrix_c)
        ext.NewtonInverse_forward(matrix_c, output, int(num_iters))
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        ext = _require_extension()
        (inverse,) = ctx.saved_tensors
        grad_matrix = torch.empty_like(inverse)
        ext.NewtonInverse_backward(grad_output.contiguous(), inverse.contiguous(), grad_matrix)
        return grad_matrix, None


def laplace_l1_distance_cuda(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Return ||query - key||_1 for all query/key pairs.

    Args:
        query: Tensor with shape (B, H, N, D).
        key: Tensor with shape (B, H, M, D).
    """

    return _LaplaceSubtractionFunction.apply(query, key)


def laplacian_kernel_cuda(
        query: torch.Tensor,
        key: torch.Tensor,
        lambda_scale: float,
        ) -> torch.Tensor:
    distances = laplace_l1_distance_cuda(query, key)
    return torch.exp(-distances / lambda_scale)


def newton_inverse_cuda(matrix: torch.Tensor, num_iters: int) -> torch.Tensor:
    return _NewtonInverseFunction.apply(matrix, int(num_iters))
