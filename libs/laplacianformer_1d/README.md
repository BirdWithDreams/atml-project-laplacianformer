# LaplacianFormer 1D CUDA

CUDA extension for sequence-shaped Laplacian attention kernels used by NLP
models. It computes pairwise L1 distances for tensors shaped `(B, H, N, D)` and
`(B, H, M, D)`, returning `(B, H, N, M)`.

Build in place:

```bash
uv run python setup.py build_ext --inplace
```

The initial implementation supports `float32` and `float64`. Use
`trainer.precision=32` for the CUDA-backed NLP attention path.
