# Laplacianformer Experiments

This repository supports four Hydra task configs:

- `cv_classification`
- `semantic_segmentation`
- `nlp_classification`
- `ner_task`
- `generation_nlp`

## Environment Setup

Create and activate the project virtual environment, then install the Python
dependencies and local CUDA extension:

```bash
uv venv .venv --python 3.10
source ./.venv/bin/activate
uv pip install -r requirements.txt
cd ./libs/laplacianformer
uv pip install -e . --no-build-isolation
cd ../..
uv pip install seqeval
```

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

Text Generation (Summarization/Translation):
```bash
uv run train.py task=generation_nlp model=vanilla_seq2seq_base datamodule=seq2seq

uv run train.py task=generation_nlp model=vanilla_seq2seq_base model.attn_type=laplacian datamodule=seq2seq trainer.precision=32
```

ImageNet Sample (Imagenette) Classification:

To test ImageNet classification without downloading the full 150GB dataset, you can fetch a 10-class sample (Imagenette).

1. Download and extract the sample:
```bash
uv run python download_sample.py
```

2. Run the training task:
```bash
uv run train.py task=cv_classification model=vanilla_pvt_small datamodule=imagenet
```

Visualize Attention Maps (Vanilla vs. Laplacian):
You can generate side-by-side heatmaps to compare how the different kernels distribute attention weights. This script supports both images and text and outputs a `.png` file.

1. For Computer Vision (Provide a path to an image):
```bash
uv run python scripts/visualize_attention.py --mode vision --data_path "./data/imagenette2-320/train/n01440764/ILSVRC2012_val_00000293.JPEG"
```

2. For NLP (Provide a text prompt wrapped in quotes):

```bash
uv run python scripts/visualize_attention.py --mode text --data_path "The Laplacian kernel decays slower than Gaussian, allowing the network to retain distant context."
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

Or with `screen` (preffered):
```bash
LOG="./logs/ner_log_$(date +%Y%m%d_%H%M%S).log" && screen -S ner_run -L -Logfile "$LOG" -dm bash -lc 'cd /workspace/atml-project-laplacianformer && source .venv/bin/activate && bash scripts/run_ner_model_matrix.sh'
LOG="./logs/ner_log_$(date +%Y%m%d_%H%M%S).log" && screen -S ner_run -L -Logfile "$LOG" -dm bash -lc 'cd /workspace/atml-project-laplacianformer && source .venv/bin/activate && bash scripts/run_ner_gen2_model_matrix.sh --skip 3'

LOG="./logs/seg_log_$(date +%Y%m%d_%H%M%S).log" && screen -S seg_run -L -Logfile "$LOG" -dm bash -lc 'cd /workspace/atml-project-laplacianformer && source .venv/bin/activate && bash scripts/run_segmentation_model_matrix.sh --skip 4'
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
- text: `text.py`, `text_ner.py`, `laplacian_1d_attn.py`, 1D Laplacian CUDA wrapper
- shared attention: `vanilla_attn.py`, `laplacian_attn.py`

The old flat ViT-style vision path and non-task model configs were removed.
