# Laplacianformer Experiments

This repository supports four Hydra task configs:

- `cv_classification`
- `semantic_segmentation`
- `nlp_classification`
- `ner_task`

## Model Matrix

Image tasks use only PVT-style backbones with 2D RoPE:

- `vanilla_pvt_small`
- `vanilla_pvt_medium`
- `laplacian_pvt_small_cuda`
- `laplacian_pvt_medium_cuda`

Text tasks use only 1D sequence backbones:

- `vanilla_1d_small`
- `vanilla_1d_medium`
- `laplacian_1d_cuda_small`
- `laplacian_1d_cuda_medium`

The CUDA Laplacian configs do not fall back to torch. Run them with
`trainer.precision=32`.

## Examples

CV classification:

```bash
uv run train.py task=cv_classification model=vanilla_pvt_small datamodule=cifar100
uv run train.py task=cv_classification model=laplacian_pvt_small_cuda datamodule=cifar100 trainer.precision=32
```

Semantic segmentation:

```bash
uv run train.py task=semantic_segmentation model=vanilla_pvt_small datamodule=voc2012_segmentation
uv run train.py task=semantic_segmentation model=laplacian_pvt_small_cuda datamodule=voc2012_segmentation trainer.precision=32
```

Text classification:

```bash
uv run train.py task=nlp_classification model=vanilla_1d_small datamodule=sst2
uv run train.py task=nlp_classification model=laplacian_1d_cuda_small datamodule=sst2 trainer.precision=32
```

NER:

```bash
uv run train.py task=ner_task model=vanilla_1d_small datamodule=conll2003
uv run train.py task=ner_task model=laplacian_1d_cuda_small datamodule=ontonotes5 trainer.precision=32
```

## Matrix Scripts

NER matrix:

```bash
scripts/run_ner_model_matrix.sh
```

Segmentation matrix:

```bash
scripts/run_segmentation_model_matrix.sh
```

Both scripts accept space-separated environment overrides, for example:

```bash
MODELS="vanilla_pvt_small laplacian_pvt_small_cuda" \
DATASETS="voc2012_segmentation coco_segmentation" \
MAX_EPOCHS=1 \
scripts/run_segmentation_model_matrix.sh
```

## Model Code

`src/models` is intentionally narrow:

- image: `pvt.py`, `rope.py`, `segmentation.py`, 2D Laplacian CUDA wrappers
- text: `text.py`, `text_ner.py`, 1D Laplacian CUDA wrappers
- shared attention: `vanilla_attn.py`, `laplacian_attn.py`

The old flat ViT-style vision path and non-task model configs were removed.
