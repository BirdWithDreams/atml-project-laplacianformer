from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
EXTENSION_DIR = REPO_ROOT / "libs" / "laplacianformer_1d"
sys.path.insert(0, str(EXTENSION_DIR))

from laplacian_1d import (  # noqa: E402
    LaplacianAttention1D,
    can_use_fused_newton_cuda,
    exact_laplacian_attention_1d,
    fused_newton_schulz_inverse,
    is_laplacian_1d_cuda_available,
    l1_pairwise_distance,
    laplacian_kernel,
    newton_schulz_inverse,
    nystrom_laplacian_attention_1d,
)


def torch_l1_distance(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    return (q.unsqueeze(-2) - k.unsqueeze(-3)).abs().sum(dim=-1)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_laplacian_1d_cuda_available(),
    reason="1D Laplacian CUDA extension is not available",
)


def test_l1_forward_matches_torch():
    q = torch.randn(2, 3, 5, 7, device="cuda", dtype=torch.float32)
    k = torch.randn(2, 3, 6, 7, device="cuda", dtype=torch.float32)

    out_cuda = l1_pairwise_distance(q, k)
    out_ref = torch_l1_distance(q, k)

    torch.testing.assert_close(out_cuda, out_ref, rtol=1e-5, atol=1e-5)


def test_l1_gradcheck():
    q = torch.randn(1, 1, 4, 3, device="cuda", dtype=torch.float64, requires_grad=True)
    k = torch.randn(1, 1, 5, 3, device="cuda", dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(
        lambda q_, k_: l1_pairwise_distance(q_, k_),
        (q, k),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-4,
    )


def test_laplacian_kernel_matches_reference():
    q = torch.randn(2, 3, 5, 7, device="cuda")
    k = torch.randn(2, 3, 6, 7, device="cuda")
    lambda_ = 4.0

    dist_ref = torch_l1_distance(q, k)
    G_ref = torch.exp(-dist_ref / lambda_)
    G = laplacian_kernel(q, k, lambda_=lambda_)

    torch.testing.assert_close(G, G_ref, rtol=1e-5, atol=1e-5)


def test_exact_row_attention_matches_reference():
    q = torch.randn(2, 3, 5, 7, device="cuda")
    k = torch.randn(2, 3, 6, 7, device="cuda")
    v = torch.randn(2, 3, 6, 11, device="cuda")
    lambda_ = 4.0

    dist_ref = torch_l1_distance(q, k)
    G_ref = torch.exp(-dist_ref / lambda_)
    A_ref = G_ref / (G_ref.sum(dim=-1, keepdim=True) + 1e-6)
    y_ref = torch.matmul(A_ref, v)

    y = exact_laplacian_attention_1d(q, k, v, lambda_=lambda_, normalization="row")

    torch.testing.assert_close(y, y_ref, rtol=1e-5, atol=1e-5)


def test_exact_attention_shape_and_grad():
    q = torch.randn(2, 4, 8, 16, device="cuda", requires_grad=True)
    k = torch.randn(2, 4, 8, 16, device="cuda", requires_grad=True)
    v = torch.randn(2, 4, 8, 16, device="cuda", requires_grad=True)

    y = exact_laplacian_attention_1d(q, k, v, normalization="paper")

    assert y.shape == v.shape
    assert torch.isfinite(y).all()

    loss = y.square().mean()
    loss.backward()

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None


def test_laplacian_attention_module_shape_and_grad():
    q = torch.randn(2, 4, 8, 16, device="cuda", requires_grad=True)
    k = torch.randn(2, 4, 8, 16, device="cuda", requires_grad=True)
    v = torch.randn(2, 4, 8, 16, device="cuda", requires_grad=True)
    attn = LaplacianAttention1D(num_heads=4, head_dim=16).cuda()

    y = attn(q, k, v)

    assert y.shape == v.shape
    assert torch.isfinite(y).all()

    y.mean().backward()

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None


def test_newton_schulz_python_fallback_shape_and_grad():
    W = torch.eye(16, device="cuda").reshape(1, 1, 16, 16).clone().requires_grad_()

    W_inv = newton_schulz_inverse(W, num_iters=6, use_fused_cuda=False)

    assert W_inv.shape == W.shape
    assert torch.isfinite(W_inv).all()

    W_inv.mean().backward()

    assert W.grad is not None


def test_fused_newton_schulz_shape_and_grad_when_available():
    W = torch.eye(16, device="cuda").reshape(1, 1, 16, 16).clone().requires_grad_()
    if not can_use_fused_newton_cuda(W):
        pytest.skip("fused Newton-Schulz CUDA extension is not available for N=16")

    W_inv = fused_newton_schulz_inverse(W, num_iters=6)

    assert W_inv.shape == W.shape
    assert torch.isfinite(W_inv).all()

    W_inv.mean().backward()

    assert W.grad is not None


def test_nystrom_attention_shape():
    q = torch.randn(2, 4, 64, 16, device="cuda", requires_grad=True)
    k = torch.randn(2, 4, 64, 16, device="cuda", requires_grad=True)
    v = torch.randn(2, 4, 64, 16, device="cuda", requires_grad=True)

    y = nystrom_laplacian_attention_1d(
        q,
        k,
        v,
        num_landmarks=16,
        lambda_=4.0,
    )

    assert y.shape == v.shape
    assert torch.isfinite(y).all()

    y.mean().backward()

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
