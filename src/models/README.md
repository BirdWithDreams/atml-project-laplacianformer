# Model Architecture

This directory intentionally keeps only the model code used by the task configs
in `configs/task`.

## Vision

Image tasks use one backbone family:

- `pvt.py`: PVT-style hierarchical vision backbone.
- `rope.py`: 2D RoPE used inside PVT attention blocks.
- `segmentation.py`: segmentation decoder on top of PVT feature maps.
- `laplacian_attn.py`: 2D Laplacian attention and the shared 1D base.
- `laplacian_fast_attn.py`: CUDA-backed 2D Laplacian attention wrapper.
- `laplacian_cuda_ops.py`: autograd wrapper for `libs/laplacianformer`.
- `vanilla_attn.py`: vanilla softmax attention baseline used inside PVT.

Supported image configs:

- `vanilla_pvt_small`
- `vanilla_pvt_medium`
- `laplacian_pvt_small_cuda`
- `laplacian_pvt_medium_cuda`

All image configs use `backbone_type: pvt` and `use_rope: true`. The Laplacian
PVT configs use the CUDA backend and do not fall back to torch.

## NLP

Text tasks use 1D sequence backbones:

- `text.py`: sequence classification backbone.
- `text_ner.py`: token classification backbone for NER.
- `laplacian_fast_1d_attn.py`: CUDA-backed 1D Laplacian attention wrapper.
- `laplacian_1d_cuda_ops.py`: autograd wrapper for `libs/laplacianformer_1d`.

Supported text configs:

- `vanilla_1d_small`
- `vanilla_1d_medium`
- `laplacian_1d_cuda_small`
- `laplacian_1d_cuda_medium`

The CUDA Laplacian text configs require the 1D extension from
`libs/laplacianformer_1d` and should be run with `trainer.precision=32`.
