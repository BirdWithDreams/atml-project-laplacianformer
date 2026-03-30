# Laplacian vs. Vanilla Transformer: An Attention Comparison

This project compares the traditional $\mathcal{O}(N^2)$ **Vanilla Softmax Attention** against the $\mathcal{O}(N)$ **Laplacian Linear Attention** mechanism introduced in the paper *"LaplacianFormer: Rethinking Linear Attention with Laplacian Kernel"*. 

Standard linear attention models often rely on Gaussian kernels, which can over-suppress mid-range token interactions and lead to vanishing gradients. This project tests the hypothesis that replacing the Gaussian kernel with a Laplacian kernel ($l_1$ distance) improves gradient flow and token representation while maintaining linear computational complexity.

---

## 📁 Project Structure

```text
laplacian_vs_vanilla/
│
├── models/
│   ├── __init__.py
│   ├── vanilla_attn.py      # Implements standard Multi-Head Softmax Attention.
│   ├── laplacian_attn.py    # Implements Laplacian Linear Attention with Newton-Schulz inverse.
│   └── vit_wrapper.py       # A modular Vision Transformer backbone to easily swap attention types.
│
├── data/                    # Automatically generated folder for downloaded datasets (CIFAR-100).
├── train.py                 # Main training loop, data loading, and Weights & Biases logging.
├── pyproject.toml           # Project metadata and dependency declarations.
├── uv.lock                  # Locked dependency versions for reproducible builds.
└── README.md                # Project documentation.
```

---

## 🚀 Environment Setup (using `uv`)

We use `uv`, an extremely fast Python package installer and resolver, to manage the environment.

**1. Install `uv` (if you haven't already):**
If you don't have `uv` installed globally, you can install it via pip or terminal.
* **Windows (PowerShell):** `irm https://astral.sh/uv/install.ps1 | iex`
* **macOS/Linux:** `curl -LsSf https://astral.sh/uv/install.sh | sh`

**2. Create a Virtual Environment:**
Navigate to the project root directory and create a new virtual environment:
```bash
uv venv
```

**3. Activate the Environment:**
* **Windows:**
  ```cmd
  .venv\Scripts\activate
  ```
* **macOS/Linux:**
  ```bash
  source .venv/bin/activate
  ```

**4. Install Dependencies:**
This project uses `pyproject.toml` and `uv.lock` for reproducible builds. To install the exact dependencies specified in the lockfile into your virtual environment, simply run:
```bash
uv sync
```

---

## 📊 Weights & Biases (W&B) Setup

This project uses Weights & Biases to track training loss, validation accuracy, and step times.

**Log In:**
Before running the training script, log in to your W&B account from the terminal:
```bash
wandb login
```
*It will prompt you to paste your API key, which you can find at [https://wandb.ai/authorize](https://wandb.ai/authorize).*

---

## 🏃‍♂️ Running the Experiments

You can easily toggle between the attention mechanisms using the `--attn_type` flag. The default dataset is CIFAR-100 to allow for quicker iterations on a single GPU.

**Run the Vanilla Baseline:**
```bash
uv run python train.py --attn_type vanilla --dataset cifar100
```

**Run the Laplacian Challenger:**
```bash
uv run python train.py --attn_type laplacian --dataset cifar100
```

### Additional Command Line Arguments
You can customize the training run by passing the following flags:
* `--batch_size`: Default is 128.
* `--lr`: Learning rate (Default: 3e-4).
* `--epochs`: Number of training epochs (Default: 30).
* `--dataset`: Choose between `cifar100` and `imagenet`. *(Note: ImageNet requires manual downloading and placement in the `/data/imagenet` folder).*