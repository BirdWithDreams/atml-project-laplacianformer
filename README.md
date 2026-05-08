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
uv run train.py task=nlp_classification model=vanilla_1d_small datamodule=sst2
uv run train.py task=nlp_classification model=laplacian_1d_cuda_small datamodule=sst2 trainer.precision=32
uv run train.py task=nlp_classification_ag_news
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
python train.py task=cv_classification model=laplacian datamodule=cifar100
```

Best paper-oriented vision run:

```bash
python train.py task=cv_classification model=laplacian_pvt_tiny datamodule=cifar100
```


### NLP Classification
=======
=======
NLP classification matrix:

```bash
scripts/run_nlp_classification_model_matrix.sh
```

=======
## Benchmarking Checkpoints

`benchmark.py` evaluates Lightning checkpoints on a deterministic test subset and
writes JSONL/CSV summaries with metrics, forward-only inference time, and memory:

```bash
uv run python benchmark.py \
  'runs=[{name:ag_news_vanilla,task:nlp_classification,datamodule:ag_news,checkpoint_path:results/path/to/checkpoint.ckpt}]'
```

For multiple checkpoints, use a glob:

```bash
uv run python benchmark.py \
  'runs=[{name:ag_news_matrix,task:nlp_classification,datamodule:ag_news,checkpoint_glob:results/**/last.ckpt}]'
```

Defaults live in `configs/benchmark/default.yaml`. Common overrides:
`max_samples=2048`, `warmup_batches=10`, `device=cuda`, and per-run
`datamodule_overrides={batch_size:32,num_workers:0}`.

A single benchmark config can mix tasks:

```yaml
runs:
  - name: cifar100_vanilla
    task: cv_classification
    datamodule: cifar100
    checkpoint_path: results/cifar100.ckpt
  - name: ag_news_vanilla
    task: nlp_classification
    datamodule: ag_news
    checkpoint_path: results/ag_news.ckpt
  - name: conll2003_vanilla
    task: ner_task
    datamodule: conll2003
    checkpoint_path: results/conll2003.ckpt
  - name: voc2012_vanilla
    task: semantic_segmentation
    datamodule: voc2012_segmentation
    checkpoint_path: results/voc2012.ckpt
```

>>>>>>> 82c88df (Add checkpoint benchmark runner)
Or with `screen` (preffered):
```bash
LOG="./logs/ner_log_$(date +%Y%m%d_%H%M%S).log" && screen -S ner_run -L -Logfile "$LOG" -dm bash -lc 'cd /workspace/atml-project-laplacianformer && source .venv/bin/activate && bash scripts/run_ner_model_matrix.sh'
LOG="./logs/ner_log_$(date +%Y%m%d_%H%M%S).log" && screen -S ner_run -L -Logfile "$LOG" -dm bash -lc 'cd /workspace/atml-project-laplacianformer && source .venv/bin/activate && bash scripts/run_ner_gen2_model_matrix.sh --skip 3'

LOG="./logs/seg_log_$(date +%Y%m%d_%H%M%S).log" && screen -S seg_run -L -Logfile "$LOG" -dm bash -lc 'cd /workspace/atml-project-laplacianformer && source .venv/bin/activate && bash scripts/run_segmentation_model_matrix.sh --skip 4'
LOG="./logs/nlp_log_$(date +%Y%m%d_%H%M%S).log" && screen -S nlp_run -L -Logfile "$LOG" -dm bash -lc 'cd /workspace/atml-project-laplacianformer && source .venv/bin/activate && bash scripts/run_nlp_classification_model_matrix.sh'
```

Both scripts accept space-separated environment overrides, for example:
>>>>>>> 17fd14d (Introduces  argument to segmentation batch script)

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
