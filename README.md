# Laplacian vs. Vanilla Transformer: An Attention Comparison
 
This project serves as a scalable, robust research repository to compare traditional $\mathcal{O}(N^2)$ **Vanilla Softmax Attention** against the $\mathcal{O}(N)$ **Laplacian Linear Attention** mechanism (introduced in the paper *"LaplacianFormer: Rethinking Linear Attention with Laplacian Kernel"*).

Standard linear attention models often rely on Gaussian kernels, which can over-suppress mid-range token interactions and lead to vanishing gradients. This repository tests the hypothesis that replacing the Gaussian kernel with a Laplacian kernel ($l_1$ distance) improves gradient flow and token representation while maintaining linear computational complexity.

The architecture is built on top of **PyTorch Lightning** and **Hydra** for maximum reproducibility, clean code separation, and effortless configuration.

---

## 🏗️ Architecture: What is implemented, where and why?

We follow a strict separation of concerns, decoupling the **Backbone** (how tokens attend to each other) from the **Task** (how loss is computed) and the **Data** (what feeds the model).

### 1. Backbones (`src/models/`)
The backbone's only job is to turn raw sequences or images into rich embeddings. 
- **`vision.py`**: Contains `VisionBackbone` (a ViT-like architecture). It splits images into patches and returns the `[CLS]` token embedding.
- **`text.py`**: Contains `TextBackbone`, which embeds standard 1D text token sequences (like BERT). It returns the average pooled sequence embedding.
- **`vanilla_attn.py` & `laplacian_attn.py`**: Contain the core attention logic. Notably, `laplacian_attn.py` has been expanded to include `Laplacian1DLinearAttention` specifically designed for 1D text sequences.

### 2. Lightning Modules / Tasks (`src/tasks/`)
Tasks are PyTorch Lightning modules (`LightningModule`). They contain the training loop, optimizer configuration, and metrics logic.
- **`classification_cv.py`**: Defines `CVClassificationTask`. It initializes a Vision Backbone, creates a Linear Head for classification, utilizes `CrossEntropyLoss`, tracking accuracy, precision, and recall via `torchmetrics`.
- **`classification_nlp.py`**: Defines `NLPClassificationTask`. Similar to CV, it initializes a Text Backbone and trains the model for text classification.

### 3. Data Modules (`src/datamodules/`)
These handle downloading, tokenizing, and wrapping datasets in PyTorch DataLoaders.
- **`cv_datamodule.py`**: Wraps the `torchvision` CIFAR-100 dataset.
- **`nlp_datamodule.py`**: Uses HuggingFace's `datasets` and `transformers` to automatically download and tokenize the GLUE SST-2 (Sentence Classification) dataset.

### 4. Configuration System (`configs/`)
Everything is assembled powerfully by **Hydra** using hierarchical YAML files. You do not need to modify Python code to change parameters.
- `configs/config.yaml`: The entry configuration defining defaults.
- `configs/model/`: Definitions for model dimensions, attention heads, and depth (`vanilla.yaml`, `laplacian.yaml`).
- `configs/task/`: Hyperparameters like Learning Rate and Optimizers (`cv_classification.yaml`).
- `configs/datamodule/`: Batch sizes and dataset configurations (`cifar100.yaml`, `sst2.yaml`).

---

## 🚀 Environment Setup 

We use `uv`, an extremely fast Python package resolver to manage dependencies.

**1. Install `uv`:**
- **Windows (PowerShell):** `irm https://astral.sh/uv/install.ps1 | iex`
- **macOS/Linux:** `curl -LsSf https://astral.sh/uv/install.sh | sh`

**2. Sync Environment & Install Dependencies:**
Navigate to the project root and run:
```bash
uv sync
```
*This will create the virtual environment and install `transformers`, `lightning`, `hydra-core`, `torchmetrics`, `loguru`, etc. based on `pyproject.toml`.*

---

## 🏃‍♂️ Running the Experiments

By leveraging Hydra, you can launch entirely different task domains purely via command-line arguments. Logs will seamlessly upload to Weights & Biases (W&B).

**Run Computer Vision (CIFAR100) with Vanilla Attention:**
```bash
uv run python train.py task=cv_classification model=vanilla datamodule=cifar100
```
**Run Computer Vision (CIFAR100) with Laplacian Attention:**
```bash
uv run python train.py task=cv_classification model=laplacian datamodule=cifar100
```

**Run NLP (SST-2 Sentence Classification) with Laplacian Attention:**
```bash
uv run python train.py task=nlp_classification model=laplacian datamodule=sst2
```

### Advanced Overrides
Use `+` or `=` to override config nodes. For example, to change batch size, learning rate, and run a fast 1-epoch debug test:
```bash
uv run python train.py task=cv_classification datamodule.batch_size=64 task.lr=1e-4 trainer.max_epochs=1 +trainer.fast_dev_run=true
```

---

## 🛠️ How to Extend for New Tasks

The repository is modular. If you want to train on a new Dataset or Task (e.g., Object Detection), follow these steps:

### 1. Add a New DataModule
Create a new file (e.g. `src/datamodules/my_new_data.py`).
1. Make your class inherit from `lightning.LightningDataModule`.
2. Implement `prepare_data()` (download logic) and `setup()` (splitting subsets).
3. Return PyTorch dataloaders in `train_dataloader()`, `val_dataloader()`, etc.
4. **Hydra Config:** Create `configs/datamodule/my_new_data.yaml` defining its parameters (batch size, etc).

### 2. Add a New Backbone (Optional)
If your task requires different feature representations (e.g. audio spectrograms or point clouds), create `src/models/audio.py` that implements `nn.Module`. Make sure it uses `self.attn = MultiHeadAttention()` vs `self.attn = LaplacianLinearAttention()` controlled via an `attn_type` flag.

### 3. Add a New Task (Lightning Module)
Create a new file (e.g. `src/tasks/detection.py`).
1. Make your class inherit from `lightning.LightningModule`.
2. In `__init__`, conditionally instantiate your backbone and define your task head (e.g., bounding box regressor).
3. Implement `training_step()` calculating loss and logging metrics. Configure your optimizer natively in `configure_optimizers()`.
4. **Hydra Config:** Create `configs/task/detection.yaml` with learning rates, optimizer variables, and task-specific logic.

### 4. Register in `train.py`
Open `train.py`. Under `# 1. Setup DataModule` and `# 2. Setup Task & Model`, add simple `if / elif` conditionals checking `cfg.task.name` to instantiate your newest classes!

```python
# snippet representing addition to train.py
elif cfg.task.name == "my_new_task":
    from src.tasks.detection import MyNewTask
    task = MyNewTask(...)
```