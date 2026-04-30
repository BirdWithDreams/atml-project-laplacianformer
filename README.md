# Laplacian vs. Vanilla Transformer

This repository is a research codebase for comparing standard softmax attention against Laplacian linear attention from *LaplacianFormer: Rethinking Linear Attention with Laplacian Kernel*.

The project now serves two purposes:

- paper-oriented computer vision experiments
- extension experiments in NLP, NER, and semantic segmentation

The current best paper match is the PVT-style vision backbone with RoPE and Laplacian attention. The older ViT-like vision models and the NLP/NER models are still useful, but they should be treated as baselines or extensions rather than as the closest reproduction of the paper.

For model-specific architecture notes, see [src/models/README.md](</D:/Studying/USI MAI/atml-project/src/models/README.md:1>).

## Project Layout

- `src/models/`: reusable backbones and attention modules
- `src/tasks/`: Lightning tasks for CV classification, NLP classification, NER, and semantic segmentation
- `src/datamodules/`: dataset wrappers for CIFAR-100, SST-2, CoNLL-2003, OntoNotes 5, VOC 2012, and COCO
- `configs/`: Hydra configuration groups
- `scripts/`: convenience scripts for experiment sweeps
- `train.py`: Hydra entrypoint that composes the config and launches training

## Model Families

### Vision

- `vision.py` and `vit_wrapper.py`: flat ViT-like baselines
- `pvt.py`: hierarchical PVT-style backbone with optional RoPE
- `segmentation.py`: lightweight PVT segmentation decoder
- `laplacian_attn.py`: 2D Laplacian attention for vision
- `laplacian_fast_attn.py`: optional CUDA-backed Laplacian attention fast path
- `rope.py`: 2D rotary positional encoding utilities

Best paper match:

- `model=laplacian_pvt_tiny`
- `model=laplacian_pvt_small`

### Text and NER

- `text.py`: sequence classification backbone
- `text_ner.py`: token classification backbone for NER
- `laplacian_attn.py`: 1D Laplacian attention with padding-mask support
- `vanilla_attn.py`: standard softmax attention baseline

These are project extensions, not part of the original paper reproduction claim.

## Hydra Config Structure

The config tree is now organized by concern:

- `configs/config.yaml`: root composition entrypoint
- `configs/task/`: task identity and task-level defaults
- `configs/model/`: architecture presets
- `configs/datamodule/`: dataset and tokenizer settings
- `configs/optimizer/`: optimizer presets
- `configs/trainer/`: Lightning trainer settings
- `configs/logger/`: logging backend settings

At runtime, the composed config has these top-level groups:

- `task`
- `model`
- `datamodule`
- `optimizer`
- `trainer`
- `logger`

Task configs override the default optimizer group through Hydra. For example:

- `task=cv_classification` defaults to `optimizer=adamw_cv_default`
- `task=ner_task` defaults to `optimizer=adamw_text_default`

You can still override the optimizer explicitly from the CLI.

## Model Presets

### ViT-like Text and NER Presets

Vanilla presets:

- `vanilla_tiny`
- `vanilla_small`
- `vanilla_medium`

Laplacian presets:

- `laplacian_tiny`
- `laplacian_small`
- `laplacian_medium`

These are the main size-controlled presets for NLP and NER experiments.

### Vision-Oriented Presets

- `vanilla`: legacy ViT-like vanilla baseline
- `laplacian`: legacy ViT-like Laplacian baseline
- `laplacian_pvt_tiny`: recommended paper-oriented vision config
- `laplacian_pvt_small`: larger paper-oriented vision config
- `laplacian_pvt_tiny_cuda`: same tiny PVT config with the authors' CUDA kernels enabled
- `laplacian_pvt_small_cuda`: same small PVT config with the authors' CUDA kernels enabled
- `laplacian_paper`: reference metadata config, not the best runnable architecture

## Optimizer Presets

The optimizer group currently includes:

- `adamw_cv_default`
- `adamw_text_default`
- `adamw_text_high_lr`
- `adam_text_baseline`

These configs control:

- optimizer type
- learning rate
- weight decay

Example:

```bash
python train.py task=ner_task model=laplacian_small datamodule=conll2003 optimizer=adam_text_baseline
```

## Environment Setup

This project uses a standard Python virtual environment.

Create and activate the environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies with:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running Experiments

### Computer Vision

Legacy ViT-like vanilla baseline:

```bash
python train.py task=cv_classification model=vanilla datamodule=cifar100
```

Legacy ViT-like Laplacian baseline:

```bash
python train.py task=cv_classification model=laplacian datamodule=cifar100
```

Best paper-oriented vision run:

```bash
python train.py task=cv_classification model=laplacian_pvt_tiny datamodule=cifar100
```

CUDA-backed PVT run after building `libs/laplacianformer`:

```bash
python train.py task=cv_classification model=laplacian_pvt_tiny_cuda datamodule=cifar100 trainer.precision=16-mixed trainer.compile=false
```

See [docs/laplacian_cuda_server.md](</D:/Studying/USI MAI/atml-project/docs/laplacian_cuda_server.md:1>) for GB10 server setup and extension build instructions.

### NLP Classification

```bash
python train.py task=nlp_classification model=laplacian_small datamodule=sst2
```

### NER

CoNLL-2003 with a vanilla tiny model:

```bash
python train.py task=ner_task model=vanilla_tiny datamodule=conll2003
```

OntoNotes 5 with a Laplacian medium model:

```bash
python train.py task=ner_task model=laplacian_medium datamodule=ontonotes5
```

### Semantic Segmentation

PASCAL VOC 2012 subset with a Laplacian PVT decoder:

```bash
python train.py task=semantic_segmentation model=laplacian_pvt_segmentation_tiny datamodule=voc2012_segmentation trainer.compile=false
```

MS COCO 2017 subset with the same task path:

```bash
python train.py task=semantic_segmentation model=laplacian_pvt_segmentation_tiny datamodule=coco_segmentation trainer.compile=false
```

COCO expects the standard local layout under `./data/coco`: `train2017`, `val2017`, and `annotations/instances_{train,val}2017.json`. Both segmentation datamodule configs default to small subsets; set `datamodule.max_train_samples=null datamodule.max_val_samples=null` for full splits.

### Useful Overrides

Change epochs, precision, or batch size directly from the CLI:

```bash
python train.py task=ner_task model=laplacian_small datamodule=conll2003 trainer.max_epochs=5 trainer.precision=32-true datamodule.batch_size=16
```

Override the default optimizer selected by the task:

```bash
python train.py task=ner_task model=vanilla_small datamodule=ontonotes5 optimizer=adamw_text_high_lr
```

Add custom W&B tags while keeping the automatic config-name tags:

```bash
python train.py task=cv_classification model=laplacian_pvt_tiny datamodule=cifar100 logger.extra_tags='[ablation, smoke]'
```

Every run is tagged automatically with the selected config groups, for example `task:cv_classification`, `model:laplacian_pvt_tiny`, `datamodule:cifar100`, and `optimizer:adamw_cv_default`. Use `logger.extra_tags` for free-form labels.

## Sequential NER Sweep

To run the current NER matrix sequentially across both NER datasets, use:

```bash
bash scripts/run_ner_model_matrix.sh
```

The script runs:

- 6 model presets
- 3 optimizer presets
- 2 datasets

Current datasets:

- `conll2003`
- `ontonotes5`

Current model list:

- `vanilla_tiny`
- `vanilla_small`
- `vanilla_medium`
- `laplacian_tiny`
- `laplacian_small`
- `laplacian_medium`

Current optimizer list:

- `adamw_text_default`
- `adamw_text_high_lr`
- `adam_text_baseline`

The script uses `python train.py` from the active virtual environment, continues after failed runs, and reports any failures at the end.

You can adjust the sweep with environment variables:

```bash
MAX_EPOCHS=5 ACCELERATOR=gpu DEVICES=1 PRECISION=bf16-mixed WANDB_PROJECT=ner-model-matrix bash scripts/run_ner_model_matrix.sh
```

Additional Hydra overrides passed to the script are forwarded to every run.

## Notes On Paper Matching

What should be treated as the closest paper match in this repository:

- `PyramidVisionBackbone` from `src/models/pvt.py`
- `LaplacianLinearAttention` from `src/models/laplacian_attn.py`
- 2D RoPE from `src/models/rope.py`
- `model=laplacian_pvt_tiny` or `model=laplacian_pvt_small`
- `model=laplacian_pvt_tiny_cuda` or `model=laplacian_pvt_small_cuda` when the authors' CUDA extension is built and benchmarked

What should not be described as the closest paper reproduction:

- the flat ViT-like `vanilla` and `laplacian` vision configs
- the NLP and NER extensions
- `laplacian_paper`, which is now only a reference metadata config

The default implementation is architecture-faithful and method-faithful in plain PyTorch. The optional CUDA backend now uses the authors' custom kernels for the heavy Laplacian pieces, but it still needs server-side build and benchmark validation and is not a guarantee that the full official training recipe is reproduced.

## Extending The Repo

If you add a new experiment family, keep the same separation of concerns:

- add a new `src/datamodules/*.py` file for data handling
- add or extend a model in `src/models/`
- add a task in `src/tasks/`
- add Hydra presets under the appropriate group in `configs/`
- wire the new task/datamodule path in `train.py`

That keeps the CLI stable and makes sweep scripts easy to compose.
