# LaplacianFormer 1D CUDA

This extension provides a CUDA primitive for pairwise L1 distances:

```text
D[b,h,i,j] = ||q[b,h,i] - k[b,h,j]||_1
```

It does not directly compute full Laplacian attention. The Python module builds
Laplacian attention as:

```text
G = exp(-D / lambda)
Y = Normalize(G) @ V + DWConv1D(V)
```

This mirrors `libs/laplacianformer` at the subroutine level: CUDA handles the
pairwise L1 distance primitive, PyTorch applies `exp(-D / lambda)`, the optional
Nyström path uses the original fused CUDA Newton-Schulz inverse when
`Laplace_subtraction_cuda` is available, and PyTorch performs the remaining
matmul assembly. There is no end-to-end fused attention CUDA kernel here.

Build in place:

```bash
uv run python setup.py build_ext --inplace
```

The Python API exposes:

- `l1_pairwise_distance(q, k)`
- `laplacian_kernel(q, k, lambda_=4.0)`
- `exact_laplacian_attention_1d(q, k, v, normalization="paper")`
- `LaplacianAttention1D`
- `nystrom_laplacian_attention_1d(..., use_fused_newton=True)`
- `fused_newton_schulz_inverse(matrix, num_iters=6)`

The initial CUDA primitive supports `float32` and `float64`. Use
`trainer.precision=32` for this path unless half/bfloat16 support is added.
