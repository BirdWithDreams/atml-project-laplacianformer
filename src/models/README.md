# Model Architecture

This directory contains the reusable model building blocks for the project. The codebase now has two different roles:

1. A paper-oriented computer vision path centered on Laplacian attention.
2. Experimental extensions of the same attention idea to NLP and NER.

The important distinction is that not every model in this directory is meant to reproduce the paper. Some files are baselines, some are extensions, and one path is the best paper match we currently have.

## Directory Map

- `laplacian_attn.py`: Laplacian attention implementations for vision (2D) and text (1D).
- `vanilla_attn.py`: standard softmax Transformer components used as a reference baseline.
- `rope.py`: 2D rotary positional embedding utilities for vision attention.
- `pvt.py`: hierarchical PVT-style vision backbone with optional RoPE and either Laplacian or vanilla attention.
- `vision.py`: legacy single-stage ViT-like vision backbone.
- `vit_wrapper.py`: alternative ViT-style wrapper that mirrors the same block logic.
- `text.py`: text classification backbone.
- `text_ner.py`: token-level NER backbone.

## Core Attention Modules

### `laplacian_attn.py`

This is the main paper-inspired attention implementation.

For vision, `LaplacianLinearAttention` works on spatial tokens and uses:

- learned `Q`, `K`, `V` projections
- a Laplacian kernel of the form `exp(-||q-k||_1 / lambda)`
- average-pooled landmark tokens
- a Newton-Schulz inverse solver for the landmark kernel
- a depthwise convolution residual path on `V` for local context
- optional 2D RoPE on `Q` and `K`

For NLP, `LaplacianLinearAttention1D` reuses the same idea in sequence form and now supports padding-aware masking.

This file is the closest implementation of the paper's main methodological contribution.

### `vanilla_attn.py`

This is the standard softmax-attention reference implementation. We keep it for:

- baseline comparisons
- ablations against Laplacian attention
- fallback attention inside non-paper paths

It should be treated as supporting infrastructure, not as the paper model itself.

### `rope.py`

This file provides 2D rotary positional encoding for vision attention. In the current repo, RoPE is used by the PVT-style backbone and can also be applied inside the 2D Laplacian attention module.

## Vision Backbones

### `pvt.py`: PVT-Style Hierarchical Backbone

`PyramidVisionBackbone` is the recommended vision architecture when we want to match the paper as closely as possible.

Its structure is:

1. Overlapping patch embedding for stage 1.
2. A stack of Transformer-style blocks at that stage.
3. Spatial downsampling into the next stage.
4. Repeat for 4 pyramid stages.
5. Mean pool the final-stage features for classification.

Each `PyramidBlock` can use either:

- Laplacian attention, or
- vanilla softmax attention

When configured with Laplacian attention and RoPE, this is the strongest architecture-level match to the paper currently available in the repository.

Why this is the best match:

- it is hierarchical rather than flat
- it follows a PVT-style staged pyramid instead of a single-resolution ViT encoder
- it supports RoPE in the vision attention path
- it uses the Laplacian attention implementation in the actual spatial blocks

### `vision.py` and `vit_wrapper.py`: ViT-Like Backbones

These files implement older, flat, single-stage ViT-style encoders.

They are still useful for:

- debugging
- controlled comparisons
- ablation studies against the newer PVT path

But they are not the best paper match because they differ from the paper's hierarchical PVT-style design. They should be described as paper-inspired baselines rather than reproductions.

## NLP Backbones

### `text.py`

`TextBackbone` adapts the attention modules to sequence classification. It supports both vanilla and Laplacian attention and now propagates padding masks through the model.

### `text_ner.py`

`TextBackboneNER` is the token-level variant for NER. It shares the same attention block structure as `text.py`.

These NLP models are extensions of the repository's ideas. They are useful for additional experiments, but they are not part of the original vision-paper reproduction claim.

## What We Should Treat As The Best Match For The Paper

Within this repository, the best runnable paper match is:

- backbone: `PyramidVisionBackbone` in `pvt.py`
- attention: `LaplacianLinearAttention` in `laplacian_attn.py`
- positional encoding: 2D RoPE from `rope.py`
- task path: `CVClassificationTask` with `backbone_type: "pvt"`

In practice, that means:

- `configs/model/laplacian_pvt_tiny.yaml` is the default best-match reference config
- `configs/model/laplacian_pvt_small.yaml` is the larger paper-aligned variant

These should be preferred when we describe results as the closest reproduction of the paper available in this codebase.

## What Should Not Be Presented As The Paper Match

- `configs/model/laplacian.yaml`: useful Laplacian baseline, but still built on the flat ViT-like backbone
- `configs/model/laplacian_paper.yaml`: paper-reference metadata for discussion, not the best runnable architecture anymore
- `vision.py` or `vit_wrapper.py` Laplacian runs: valid baselines, but not the closest architecture match
- `text.py` and `text_ner.py`: project extensions beyond the original paper scope

## Remaining Gaps Versus An Exact Reproduction

Even the PVT+RoPE path should be described carefully.

What it matches well:

- hierarchical PVT-style vision backbone
- Laplacian attention in the spatial blocks
- RoPE in the vision path
- landmark pooling, Newton-Schulz inverse, and depthwise local mixing

What still differs from a perfect reproduction:

- the implementation is plain PyTorch, not custom CUDA kernels
- we do not have the authors' official code
- the exact training recipe and optimization details may still differ from the paper

So the correct wording is:

- architecture-faithful and method-faithful approximation: yes
- exact paper reproduction: not guaranteed

## Recommended Usage

For the closest paper-oriented CV experiments, start from:

```bash
uv run train.py task=cv_classification model=laplacian_pvt_tiny datamodule=cifar100
```

Use the ViT-like models only when you explicitly want a baseline or ablation against the paper-oriented backbone.
