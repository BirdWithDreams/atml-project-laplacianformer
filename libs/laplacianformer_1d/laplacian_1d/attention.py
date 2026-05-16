from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


_EXTENSION_NAME = "laplacian_1d_cuda"
_NEWTON_EXTENSION_NAME = "Laplace_subtraction_cuda"
_SUPPORTED_DTYPES = {torch.float32, torch.float64}
_SUPPORTED_NEWTON_DTYPES = {torch.float16, torch.float32, torch.float64}

_EXTENSION = None
_IMPORT_ERROR: Exception | None = None
_NEWTON_EXTENSION = None
_NEWTON_IMPORT_ERROR: Exception | None = None

_LIBS_DIR = Path(__file__).resolve().parents[2]
_NEWTON_EXTENSION_DIR = _LIBS_DIR / "laplacianformer"
if _NEWTON_EXTENSION_DIR.exists():
    newton_extension_dir = str(_NEWTON_EXTENSION_DIR)
    if newton_extension_dir not in sys.path:
        sys.path.insert(0, newton_extension_dir)


def _load_extension():
    global _EXTENSION, _IMPORT_ERROR
    if _EXTENSION is not None:
        return _EXTENSION
    if _IMPORT_ERROR is not None:
        return None

    try:
        _EXTENSION = importlib.import_module(_EXTENSION_NAME)
    except Exception as exc:  # pragma: no cover - depends on local CUDA build
        _IMPORT_ERROR = exc
        return None
    return _EXTENSION


def _load_newton_extension():
    global _NEWTON_EXTENSION, _NEWTON_IMPORT_ERROR
    if _NEWTON_EXTENSION is not None:
        return _NEWTON_EXTENSION
    if _NEWTON_IMPORT_ERROR is not None:
        return None

    try:
        _NEWTON_EXTENSION = importlib.import_module(_NEWTON_EXTENSION_NAME)
    except Exception as exc:  # pragma: no cover - depends on local CUDA build
        _NEWTON_IMPORT_ERROR = exc
        return None
    return _NEWTON_EXTENSION


def is_laplacian_1d_cuda_available() -> bool:
    return _load_extension() is not None


def is_fused_newton_cuda_available() -> bool:
    return _load_newton_extension() is not None


def extension_diagnostics() -> str:
    if _load_extension() is not None:
        return f"{_EXTENSION_NAME} loaded"

    detail = f"{type(_IMPORT_ERROR).__name__}: {_IMPORT_ERROR}" if _IMPORT_ERROR else "not imported"
    return (
        f"{_EXTENSION_NAME} is unavailable ({detail}). Build it with "
        "`cd libs/laplacianformer_1d && uv run python setup.py build_ext --inplace`."
    )


def fused_newton_diagnostics() -> str:
    ext = _load_newton_extension()
    if ext is None:
        detail = (
            f"{type(_NEWTON_IMPORT_ERROR).__name__}: {_NEWTON_IMPORT_ERROR}"
            if _NEWTON_IMPORT_ERROR
            else "not imported"
        )
        return (
            f"{_NEWTON_EXTENSION_NAME} is unavailable ({detail}). Build it with "
            "`cd libs/laplacianformer && python setup.py build_ext --inplace`."
        )

    try:
        return (
            f"{_NEWTON_EXTENSION_NAME} loaded: compiler={ext.get_compiler_version()}, "
            f"cuda={ext.get_cuda_version()}, has_cuda={ext.has_cuda()}"
        )
    except Exception as exc:  # pragma: no cover - defensive diagnostics only
        return f"{_NEWTON_EXTENSION_NAME} imported, but diagnostics failed: {exc}"


def can_use_laplacian_1d_cuda(tensor: torch.Tensor) -> bool:
    return (
        _load_extension() is not None
        and tensor.is_cuda
        and tensor.dtype in _SUPPORTED_DTYPES
    )


def can_use_fused_newton_cuda(matrix: torch.Tensor) -> bool:
    ext = _load_newton_extension()
    if (
        ext is None
        or not matrix.is_cuda
        or matrix.dtype not in _SUPPORTED_NEWTON_DTYPES
        or matrix.dim() != 4
        or matrix.shape[-1] != matrix.shape[-2]
    ):
        return False

    try:
        return bool(ext.NewtonInverse_is_supported(int(matrix.shape[-1])))
    except Exception:
        return False


def _check_query_key(query: torch.Tensor, key: torch.Tensor) -> None:
    if query.dim() != 4 or key.dim() != 4:
        raise ValueError(
            "Expected query/key shapes (B,H,N,D) and (B,H,M,D); "
            f"got {tuple(query.shape)} and {tuple(key.shape)}."
        )
    if query.shape[:2] != key.shape[:2] or query.shape[-1] != key.shape[-1]:
        raise ValueError(
            "Expected query/key shapes (B,H,N,D) and (B,H,M,D); "
            f"got {tuple(query.shape)} and {tuple(key.shape)}."
        )
    if query.dtype != key.dtype:
        raise RuntimeError(f"query/key dtype mismatch: {query.dtype} vs {key.dtype}")
    if query.dtype not in _SUPPORTED_DTYPES:
        raise RuntimeError(
            "laplacian_1d_cuda supports float32 and float64 tensors. "
            f"Got {query.dtype}; use trainer.precision=32 for CUDA-backed training."
        )
    if not query.is_cuda or not key.is_cuda:
        raise RuntimeError("laplacian_1d_cuda requires CUDA tensors.")


class L1PairwiseDistance(Function):
    @staticmethod
    def forward(ctx, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """
        query: (B, H, N, D)
        key:   (B, H, M, D)
        return: (B, H, N, M)
        """
        ext = _load_extension()
        if ext is None:
            raise RuntimeError(extension_diagnostics())

        _check_query_key(query, key)

        query = query.contiguous()
        key = key.contiguous()

        B, H, N, _ = query.shape
        M = key.shape[-2]
        output = torch.empty(B, H, N, M, device=query.device, dtype=query.dtype)

        ext.laplacian_1d_forward(query, key, output)

        ctx.save_for_backward(query, key)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        ext = _load_extension()
        if ext is None:
            raise RuntimeError(extension_diagnostics())

        query, key = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_query = torch.empty_like(query)
        grad_key = torch.empty_like(key)

        ext.laplacian_1d_backward(
            grad_output,
            query,
            key,
            grad_query,
            grad_key,
        )

        return grad_query, grad_key


class FusedNewtonSchulzInverse(Function):
    @staticmethod
    def forward(ctx, matrix: torch.Tensor, num_iters: int) -> torch.Tensor:
        ext = _load_newton_extension()
        if ext is None:
            raise RuntimeError(fused_newton_diagnostics())
        if not matrix.is_cuda:
            raise RuntimeError("Fused Newton-Schulz inverse requires a CUDA tensor.")
        if matrix.dim() != 4 or matrix.shape[-1] != matrix.shape[-2]:
            raise ValueError(f"Expected matrix shape (B,H,N,N), got {tuple(matrix.shape)}.")
        if matrix.dtype not in _SUPPORTED_NEWTON_DTYPES:
            raise RuntimeError(
                "Fused Newton-Schulz inverse supports float16, float32, and float64. "
                f"Got {matrix.dtype}."
            )
        if not bool(ext.NewtonInverse_is_supported(int(matrix.shape[-1]))):
            raise RuntimeError(
                f"Fused Newton-Schulz inverse does not support N={matrix.shape[-1]}."
            )

        matrix = matrix.contiguous()
        output = torch.empty_like(matrix)
        ext.NewtonInverse_forward(matrix, output, int(num_iters))
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        ext = _load_newton_extension()
        if ext is None:
            raise RuntimeError(fused_newton_diagnostics())

        (inverse,) = ctx.saved_tensors
        grad_matrix = torch.empty_like(inverse)
        ext.NewtonInverse_backward(grad_output.contiguous(), inverse.contiguous(), grad_matrix)
        return grad_matrix, None


def l1_pairwise_distance(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """
    Compute D[b,h,i,j] = ||query[b,h,i] - key[b,h,j]||_1.
    """
    return L1PairwiseDistance.apply(query, key)


def fused_newton_schulz_inverse(matrix: torch.Tensor, num_iters: int = 6) -> torch.Tensor:
    """
    Use the original LaplacianFormer fused CUDA Newton-Schulz subroutine.

    The native kernel applies its own diagonal regularization internally,
    matching the behavior of `libs/laplacianformer`.
    """
    return FusedNewtonSchulzInverse.apply(matrix, int(num_iters))


def laplacian_kernel(
    query: torch.Tensor,
    key: torch.Tensor,
    lambda_: float = 4.0,
) -> torch.Tensor:
    """
    query: (B, H, N, D)
    key:   (B, H, M, D)
    return: G[b,h,i,j] = exp(-||q_i - k_j||_1 / lambda_)
    """
    if lambda_ <= 0:
        raise ValueError(f"lambda_ must be positive, got {lambda_}")

    dist = l1_pairwise_distance(query, key)
    return torch.exp(-dist / lambda_)


def _normalize_kernel(G: torch.Tensor, eps: float, normalization: str) -> torch.Tensor:
    if normalization == "row":
        return G / (G.sum(dim=-1, keepdim=True) + eps)

    if normalization == "paper":
        row_mean = G.mean(dim=-1, keepdim=True)
        G_centered = G - row_mean
        var = G_centered.var(dim=-2, keepdim=True, unbiased=False)
        return G_centered / torch.sqrt(var + eps) + (1.0 / G.shape[-1])

    raise ValueError(f"Unknown normalization mode: {normalization}")


def exact_laplacian_attention_1d(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    lambda_: float = 4.0,
    eps: float = 1e-6,
    normalization: str = "paper",
) -> torch.Tensor:
    """
    query: (B, H, N, Dq)
    key:   (B, H, M, Dq)
    value: (B, H, M, Dv)
    return: (B, H, N, Dv)
    """
    if key.shape[:-1] != value.shape[:-1]:
        raise ValueError(
            "Expected key/value shapes (B,H,M,Dq) and (B,H,M,Dv); "
            f"got {tuple(key.shape)} and {tuple(value.shape)}."
        )

    G = laplacian_kernel(query, key, lambda_=lambda_)
    Z = _normalize_kernel(G, eps=eps, normalization=normalization)
    return torch.matmul(Z, value)


def _dwconv1d_value(value: torch.Tensor, dwconv: nn.Conv1d) -> torch.Tensor:
    B, H, N, D = value.shape
    channels = H * D
    if dwconv.in_channels != channels or dwconv.out_channels != channels:
        raise ValueError(
            "DWConv1D channel count must match num_heads * head_dim; "
            f"got conv ({dwconv.in_channels}, {dwconv.out_channels}) and value channels {channels}."
        )

    v_1d = value.permute(0, 1, 3, 2).reshape(B, channels, N)
    local = dwconv(v_1d)
    return local.reshape(B, H, D, N).permute(0, 1, 3, 2)


def laplacian_attention_1d(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    lambda_: float = 4.0,
    eps: float = 1e-6,
    normalization: str = "paper",
    dwconv: nn.Conv1d | None = None,
) -> torch.Tensor:
    """
    Exact Laplacian attention. When dwconv is provided, returns
    Normalize(exp(-||Q-K||_1/lambda)) @ V + DWConv1D(V).
    """
    y = exact_laplacian_attention_1d(
        query=query,
        key=key,
        value=value,
        lambda_=lambda_,
        eps=eps,
        normalization=normalization,
    )

    if dwconv is not None:
        if query.shape[-2] != value.shape[-2]:
            raise ValueError("DWConv1D residual requires query and value to share sequence length.")
        y = y + _dwconv1d_value(value, dwconv)

    return y


class LaplacianAttention1D(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        lambda_: float = 4.0,
        eps: float = 1e-6,
        normalization: str = "paper",
        use_dwconv: bool = True,
        dwconv_kernel_size: int = 3,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.lambda_ = lambda_
        self.eps = eps
        self.normalization = normalization
        self.use_dwconv = use_dwconv

        channels = num_heads * head_dim
        if use_dwconv:
            self.dwconv = nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=dwconv_kernel_size,
                padding=dwconv_kernel_size // 2,
                groups=channels,
                bias=True,
            )
        else:
            self.dwconv = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        query, key, value: (B, H, N, D)
        return: (B, H, N, D)
        """
        return laplacian_attention_1d(
            query=query,
            key=key,
            value=value,
            lambda_=self.lambda_,
            eps=self.eps,
            normalization=self.normalization,
            dwconv=self.dwconv,
        )


def avg_pool_landmarks_1d(x: torch.Tensor, num_landmarks: int) -> torch.Tensor:
    """
    x: (B, H, N, D)
    return: (B, H, num_landmarks, D)
    """
    if x.dim() != 4:
        raise ValueError(f"Expected x shape (B,H,N,D), got {tuple(x.shape)}")
    if num_landmarks <= 0:
        raise ValueError(f"num_landmarks must be positive, got {num_landmarks}")

    B, H, N, D = x.shape
    m = num_landmarks
    if m >= N:
        return x

    r = math.ceil(N / m)
    target_len = r * m
    pad_len = target_len - N

    if pad_len > 0:
        x = F.pad(x, pad=(0, 0, 0, pad_len), mode="replicate")

    x = x.reshape(B * H, target_len, D).transpose(1, 2)
    pooled = F.avg_pool1d(x, kernel_size=r, stride=r)
    return pooled.transpose(1, 2).reshape(B, H, m, D)


def newton_schulz_inverse(
    W: torch.Tensor,
    eps: float = 1e-4,
    num_iters: int = 6,
    use_fused_cuda: bool = True,
) -> torch.Tensor:
    """
    Approximate W^{-1} using Newton-Schulz.

    W: (B, H, m, m)
    """
    if W.shape[-1] != W.shape[-2]:
        raise ValueError(f"Expected square matrices, got {tuple(W.shape)}")

    if use_fused_cuda and can_use_fused_newton_cuda(W):
        return fused_newton_schulz_inverse(W, num_iters=num_iters)

    orig_dtype = W.dtype
    device_type = W.device.type
    with torch.amp.autocast(device_type=device_type, enabled=False):
        W_work = W.float() if W.dtype in {torch.float16, torch.bfloat16} else W
        m = W_work.shape[-1]
        I = torch.eye(m, device=W_work.device, dtype=W_work.dtype)
        I = I.view(*([1] * (W_work.ndim - 2)), m, m)

        W_eps = W_work + eps * I

        norm_1 = W_eps.abs().sum(dim=-2).amax(dim=-1)
        norm_inf = W_eps.abs().sum(dim=-1).amax(dim=-1)
        alpha = 1.0 / (norm_1 * norm_inf + eps)

        X = alpha[..., None, None] * W_eps.transpose(-1, -2)
        for _ in range(num_iters):
            X = X @ (2 * I - W_eps @ X)

        residual = torch.linalg.norm(I - W_eps @ X, dim=(-2, -1))
        invalid = ~torch.isfinite(X).all(dim=(-2, -1)) | ~torch.isfinite(residual) | (residual > 1.0)
        if invalid.any():
            X = X.clone()
            X[invalid] = torch.linalg.pinv(W_eps[invalid]).to(dtype=X.dtype)

    return X.to(orig_dtype)


def nystrom_laplacian_attention_1d(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_landmarks: int,
    lambda_: float = 4.0,
    eps: float = 1e-4,
    ns_iters: int = 6,
    use_fused_newton: bool = True,
) -> torch.Tensor:
    """
    Efficient row-normalized Nyström approximation to Laplacian attention.

    query: (B, H, N, D)
    key:   (B, H, N, D)
    value: (B, H, N, Dv)
    """
    if query.shape[-2] != key.shape[-2] or key.shape[:-1] != value.shape[:-1]:
        raise ValueError(
            "Nyström 1D attention expects self-attention shapes "
            "(B,H,N,D), (B,H,N,D), and (B,H,N,Dv)."
        )

    q_landmarks = avg_pool_landmarks_1d(query, num_landmarks)
    k_landmarks = avg_pool_landmarks_1d(key, num_landmarks)

    C_q = laplacian_kernel(query, k_landmarks, lambda_=lambda_)
    C_k = laplacian_kernel(q_landmarks, key, lambda_=lambda_)
    W = laplacian_kernel(q_landmarks, k_landmarks, lambda_=lambda_)
    W_inv = newton_schulz_inverse(
        W,
        eps=eps,
        num_iters=ns_iters,
        use_fused_cuda=use_fused_newton,
    )

    kv = torch.matmul(C_k, value)
    middle = torch.matmul(W_inv, kv)
    numerator = torch.matmul(C_q, middle)

    ones = torch.ones(
        value.shape[:-1] + (1,),
        device=value.device,
        dtype=value.dtype,
    )
    k_ones = torch.matmul(C_k, ones)
    middle_den = torch.matmul(W_inv, k_ones)
    denominator = torch.matmul(C_q, middle_den)

    return numerator / (denominator + eps)
