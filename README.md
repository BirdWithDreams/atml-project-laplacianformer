# LaplacianFormer Experiments

Training, evaluation, and benchmarking code for comparing vanilla attention with
Laplacian attention across vision classification, semantic segmentation, NLP
classification, NER, and sequence-to-sequence tasks.

The main entrypoint is [train.py](/workspace/atml-project-laplacianformer/train.py),
configured through Hydra files under [configs](/workspace/atml-project-laplacianformer/configs).

## Setup

Create an environment and install the Python packages used by the training
stack. Use a PyTorch build that matches your CUDA toolkit and GPU.

```bash
uv venv .venv --python 3.10
source ./.venv/bin/activate
uv pip install torch torchvision lightning hydra-core omegaconf torchmetrics loguru wandb datasets transformers evaluate seqeval scikit-learn
```

Build the CUDA extensions used by Laplacian attention:

```bash
cd libs/laplacianformer
uv run python setup.py build_ext --inplace

cd ../laplacianformer_1d
uv run python setup.py build_ext --inplace

cd ../..
```

CUDA Laplacian configs should be run with `trainer.precision=32` unless the
specific CUDA path supports the lower precision you intend to use.

## Task Configs

Hydra task configs live in [configs/task](/workspace/atml-project-laplacianformer/configs/task):

- `cv_classification`
- `semantic_segmentation`
- `nlp_classification`
- `nlp_classification_ag_news`
- `ner_task`
- `generation_nlp`

Common datamodules live in [configs/datamodule](/workspace/atml-project-laplacianformer/configs/datamodule):

- vision: `cifar100`, `imagenet`
- segmentation: `voc2012_segmentation`, `cityscapes_segmentation`, `stanford_background_segmentation`
- text classification: `sst2`, `ag_news`
- NER: `conll2003`, `ontonotes5`
- seq2seq: `seq2seq`

## Model Configs

Vision models use PVT-style backbones with optional 2D Laplacian CUDA attention:

- `vanilla_pvt_tiny`, `vanilla_pvt_small`, `vanilla_pvt_medium`
- `laplacian_pvt_tiny_cuda`, `laplacian_pvt_small_cuda`, `laplacian_pvt_medium_cuda`
- segmentation baselines: `torchvision_fcn_resnet50`, `torchvision_deeplabv3_resnet50`

Text models use 1D sequence backbones:

- `vanilla_1d_tiny`, `vanilla_1d_small`, `vanilla_1d_medium`
- `vanilla_1d_large_1024`, `vanilla_1d_large_1024_d8`
- `laplacian_1d_cuda_tiny`, `laplacian_1d_cuda_small`, `laplacian_1d_cuda_medium`
- stability variants such as `laplacian_1d_cuda_medium_lambda8_pool2_ns8_dropout10`
- seq2seq baselines: `vanilla_seq2seq_small`, `vanilla_seq2seq_medium`, `vanilla_seq2seq_large`

The 1D CUDA extension is a pairwise L1 distance primitive. The Python package
builds Laplacian attention as `G = exp(-D / lambda)`, normalized attention, and
`Y = Normalize(G) @ V + DWConv1D(V)`. The optional Nyström path uses the
original fused CUDA Newton-Schulz inverse when it is available.

## Training Examples

CV classification:

```bash
uv run python train.py task=cv_classification model=vanilla_pvt_small datamodule=cifar100
uv run python train.py task=cv_classification model=laplacian_pvt_small_cuda datamodule=cifar100 trainer.precision=32
```

Semantic segmentation:

```bash
uv run python train.py task=semantic_segmentation model=vanilla_pvt_small datamodule=voc2012_segmentation
uv run python train.py task=semantic_segmentation model=laplacian_pvt_small_cuda datamodule=voc2012_segmentation trainer.precision=32
```

Text classification:

```bash
uv run python train.py task=nlp_classification model=vanilla_1d_small datamodule=sst2
uv run python train.py task=nlp_classification model=laplacian_1d_cuda_small datamodule=sst2 trainer.precision=32
uv run python train.py task=nlp_classification_ag_news model=laplacian_1d_cuda_small trainer.precision=32
```

NER:

```bash
uv run python train.py task=ner_task model=vanilla_1d_small datamodule=conll2003
uv run python train.py task=ner_task model=laplacian_1d_cuda_small datamodule=ontonotes5 trainer.precision=32
```

Seq2seq:

```bash
uv run python train.py task=generation_nlp model=vanilla_seq2seq_small datamodule=seq2seq
```

Curated experiment configs live in [configs/experiment](/workspace/atml-project-laplacianformer/configs/experiment):

```bash
uv run python train.py +experiment=nlp_sst2_laplacian_medium_stable
uv run python train.py +experiment=ner_conll2003_final model=laplacian_1d_cuda_small
```

## Data Notes

Cityscapes expects files under:

```text
./data/cityscapes/leftImg8bit/{train,val}
./data/cityscapes/gtFine/{train,val}
```

For ImageNet-style smoke testing without the full dataset, fetch the sample:

```bash
uv run python download_sample.py
uv run python train.py task=cv_classification model=vanilla_pvt_small datamodule=imagenet
```

## Runner Scripts

Matrix and final-run helpers live in [scripts](/workspace/atml-project-laplacianformer/scripts):

```bash
scripts/run_cv_classification_matrix.sh
scripts/run_segmentation_model_matrix.sh
scripts/run_nlp_classification_model_matrix.sh
scripts/run_ner_model_matrix.sh
scripts/run_ner_final_configs.sh
scripts/run_nlp_best_stability_configs.sh
scripts/rerun_fixed_laplacian_1d_experiments.sh
```

Most runner scripts accept space-separated environment overrides:

```bash
MODELS="vanilla_pvt_small laplacian_pvt_small_cuda" \
DATASETS="voc2012_segmentation cityscapes_segmentation" \
scripts/run_segmentation_model_matrix.sh
```

For long jobs, run scripts inside `screen`, `tmux`, or your cluster scheduler and
capture stdout/stderr to a log file. Use `DRY_RUN=true` with scripts that support
it to inspect commands before launch.

## Benchmarking

[benchmark.py](/workspace/atml-project-laplacianformer/benchmark.py) evaluates
Lightning checkpoints on a deterministic subset and writes:

- `benchmark_results.jsonl`
- `benchmark_results.csv`

Each expanded checkpoint is written as its own row. Numeric quality, timing,
memory, and efficiency fields keep their per-checkpoint values and receive
adjacent `_std` columns computed across checkpoints from the same benchmark run
name. Single-checkpoint runs therefore have `0.0` std values. Efficiency scores
use the selected quality metric divided by CUDA peak allocated GB or ms/batch.

Single checkpoint:

```bash
uv run python benchmark.py \
  'runs=[{name:ag_news_laplacian,task:nlp_classification,datamodule:ag_news,checkpoint_path:results/path/to/last.ckpt}]'
```

Checkpoint glob:

```bash
uv run python benchmark.py \
  'runs=[{name:ag_news_laplacian,task:nlp_classification,datamodule:ag_news,checkpoint_glob:results/**/last.ckpt}]'
```

Common overrides:

```bash
uv run python benchmark.py runs=final max_samples=2048 warmup_batches=10 device=cuda
```

Default quality metrics are:

- classification: `f1_macro`, then `accuracy`
- NER: `Entity_F1`, then `Token_F1`
- segmentation: `mIoU`, then `foreground_mIoU`

Override per benchmark run with `quality_metric`.

## Attention Visualization

Generate side-by-side attention heatmaps for vision or text:

```bash
uv run python scripts/visualize_attention.py \
  --mode vision \
  --data_path "./data/imagenette2-320/train/n01440764/ILSVRC2012_val_00000293.JPEG"

uv run python scripts/visualize_attention.py \
  --mode text \
  --data_path "The Laplacian kernel decays slower than Gaussian, allowing the network to retain distant context."
```

## Code Layout

- [src/models](/workspace/atml-project-laplacianformer/src/models): PVT, text backbones, vanilla attention, Laplacian attention wrappers
- [src/tasks](/workspace/atml-project-laplacianformer/src/tasks): Lightning tasks
- [src/datamodules](/workspace/atml-project-laplacianformer/src/datamodules): dataset loading
- [libs/laplacianformer](/workspace/atml-project-laplacianformer/libs/laplacianformer): original 2D Laplacian CUDA subroutines
- [libs/laplacianformer_1d](/workspace/atml-project-laplacianformer/libs/laplacianformer_1d): 1D pairwise L1 CUDA primitive and Python Laplacian attention package
- [configs](/workspace/atml-project-laplacianformer/configs): Hydra configs
- [scripts](/workspace/atml-project-laplacianformer/scripts): experiment runners
