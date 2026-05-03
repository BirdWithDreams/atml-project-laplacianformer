# Laplacian vs. Vanilla Transformer

This repository is a research codebase for comparing standard softmax attention against Laplacian linear attention from *LaplacianFormer: Rethinking Linear Attention with Laplacian Kernel*.

- `cv_classification`
- `semantic_segmentation`
- `nlp_classification`
- `ner_task`
- `generation_nlp`

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

Text Generation (Summarization/Translation):
```bash
uv run train.py task=generation_nlp model=vanilla_seq2seq_base datamodule=seq2seq

uv run train.py task=generation_nlp model=vanilla_seq2seq_base model.attn_type=laplacian datamodule=seq2seq trainer.precision=32
```


## Matrix Scripts

NER matrix:

```bash
python train.py task=cv_classification model=laplacian datamodule=cifar100
```

Best paper-oriented vision run:

```bash
python train.py task=cv_classification model=laplacian_pvt_tiny datamodule=cifar100
```

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

### Useful Overrides

Change epochs, precision, or batch size directly from the CLI:

```bash
python train.py task=ner_task model=laplacian_small datamodule=conll2003 trainer.max_epochs=5 trainer.precision=32-true datamodule.batch_size=16
```

Override the default optimizer selected by the task:

```bash
python train.py task=ner_task model=vanilla_small datamodule=ontonotes5 optimizer=adamw_text_high_lr
```

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

What should not be described as the closest paper reproduction:

- the flat ViT-like `vanilla` and `laplacian` vision configs
- the NLP and NER extensions
- `laplacian_paper`, which is now only a reference metadata config

The current implementation is architecture-faithful and method-faithful in plain PyTorch, but it is not a guaranteed exact reproduction of the paper because the original custom CUDA kernels and official training code are not available here.

## Extending The Repo

If you add a new experiment family, keep the same separation of concerns:

- add a new `src/datamodules/*.py` file for data handling
- add or extend a model in `src/models/`
- add a task in `src/tasks/`
- add Hydra presets under the appropriate group in `configs/`
- wire the new task/datamodule path in `train.py`

That keeps the CLI stable and makes sweep scripts easy to compose.
