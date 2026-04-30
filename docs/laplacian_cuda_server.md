# LaplacianFormer CUDA Backend On GB10

This project has two Laplacian attention backends:

- `laplacian_backend: "torch"`: the existing pure PyTorch implementation.
- `laplacian_backend: "cuda"`: a fast path that calls the authors' custom CUDA kernels from `libs/laplacianformer`.

The CUDA backend is currently wired for the paper-oriented vision path, especially the PVT configs. The ready-to-run configs are:

- `model=laplacian_pvt_tiny_cuda`
- `model=laplacian_pvt_small_cuda`

These configs set `laplacian_fallback_to_torch: false`, so a missing extension, unsupported dtype, or unsupported GPU target fails loudly instead of silently falling back.

## 1. Verify The Server

On the GB10 server:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
nvcc --version
python3 --version
```

Expected GPU capability for NVIDIA GB10 is `12.1`. The CUDA toolkit should be CUDA 13.x for the cleanest GB10 build.

If `nvcc --version` fails, the CUDA toolkit is not visible in your shell yet. On a module-based cluster, load the CUDA module first, for example:

```bash
module avail cuda
module load cuda/13.0
```

The exact module name is cluster-specific. A CUDA-enabled PyTorch wheel is not enough for `setup.py build_ext`; building the extension also needs the toolkit root, headers, libraries, and `nvcc`.

## 2. Create An Environment

From the repository root:

```bash
git submodule update --init --recursive
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel ninja
```

Install PyTorch with CUDA 13 support. Use the newest stable CUDA 13 wheel available for your Python version; for example:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

Then install the rest of the project dependencies without replacing that PyTorch build:

```bash
python - <<'PY'
from pathlib import Path

skip_prefixes = ("torch==", "torchvision==", "torchaudio==")
lines = Path("requirements.txt").read_text().splitlines()
kept = [
    line for line in lines
    if line.strip() and not line.startswith(skip_prefixes)
]
Path("/tmp/atml-requirements-no-torch.txt").write_text("\n".join(kept) + "\n")
PY
python -m pip install -r /tmp/atml-requirements-no-torch.txt
```

Confirm PyTorch sees CUDA and the GB10:

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("arch list:", torch.cuda.get_arch_list())
props = torch.cuda.get_device_properties(0)
print("device:", props.name)
print("compute capability:", f"{props.major}.{props.minor}")
PY
```

## 3. Build The Authors' Extension

Build from the extension directory used by this repository's Python wrapper:

```bash
cd libs/laplacianformer
if [ -z "${CUDA_HOME:-}" ] && command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
fi
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="12.1"
python setup.py build_ext --inplace
cd ../..
```

Before rerunning the build, this quick check should print non-empty `CUDA_HOME` and `nvcc` paths:

```bash
python - <<'PY'
import shutil
import torch
from torch.utils.cpp_extension import CUDA_HOME

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("CUDA_HOME:", CUDA_HOME)
print("nvcc:", shutil.which("nvcc"))
PY
```

If `12.1` is rejected by your installed CUDA/PyTorch toolchain, upgrade to a CUDA 13.x toolkit and a CUDA 13 PyTorch wheel first. GB10 is `sm_121`, so building with an older toolkit is the most likely source of architecture errors.

## 4. Verify The CUDA Backend

Run this from the repository root:

```bash
python - <<'PY'
import torch
from src.models.laplacian_cuda_ops import (
    extension_diagnostics,
    is_laplacian_cuda_available,
    laplacian_kernel_cuda,
    newton_inverse_cuda,
)

print(extension_diagnostics())
assert is_laplacian_cuda_available()

q = torch.randn(2, 4, 16, 32, device="cuda", dtype=torch.float16, requires_grad=True)
k = torch.randn(2, 4, 8, 32, device="cuda", dtype=torch.float16, requires_grad=True)
out = laplacian_kernel_cuda(q, k, lambda_scale=4.0)
loss = out.float().mean()
loss.backward()
print("kernel:", out.shape, out.dtype, torch.isfinite(out).all().item())

w = torch.eye(16, device="cuda", dtype=torch.float16).view(1, 1, 16, 16).repeat(2, 4, 1, 1)
inv = newton_inverse_cuda(w, num_iters=5)
print("inverse:", inv.shape, inv.dtype, torch.isfinite(inv).all().item())
PY
```

Then smoke-test the model:

```bash
python - <<'PY'
import torch
from src.models.pvt import PyramidVisionBackbone

model = PyramidVisionBackbone(
    attn_type="laplacian",
    laplacian_backend="cuda",
    laplacian_fallback_to_torch=False,
).cuda().eval()

x = torch.randn(1, 3, 224, 224, device="cuda")
with torch.autocast("cuda", dtype=torch.float16):
    y = model(x)

print(y.shape, y.dtype, torch.isfinite(y).all().item())
PY
```

## 5. Run Training

Important: the authors' Laplace subtraction kernel supports `float16`, `float32`, and `float64`, but not `bfloat16`. Start with `16-mixed` or `32-true`, not `bf16-mixed`.

Recommended first run:

```bash
python train.py \
  task=cv_classification \
  model=laplacian_pvt_tiny_cuda \
  datamodule=cifar100 \
  trainer.precision=16-mixed \
  trainer.compile=false
```

After that works, try `trainer.compile=true`. If `torch.compile` causes graph breaks or custom-op issues, keep `trainer.compile=false` for CUDA-backend runs.

For the larger model:

```bash
python train.py \
  task=cv_classification \
  model=laplacian_pvt_small_cuda \
  datamodule=cifar100 \
  trainer.precision=16-mixed \
  trainer.compile=false
```

## 6. Compare Against The PyTorch Backend

Use the same training settings and switch only the model config:

```bash
python train.py \
  task=cv_classification \
  model=laplacian_pvt_tiny \
  datamodule=cifar100 \
  trainer.precision=16-mixed \
  trainer.compile=false

python train.py \
  task=cv_classification \
  model=laplacian_pvt_tiny_cuda \
  datamodule=cifar100 \
  trainer.precision=16-mixed \
  trainer.compile=false
```

For quick timing, add `trainer.max_epochs=1` and keep the same batch size in both runs.

## Troubleshooting

- `CUDA_HOME environment variable is not set`: load the server CUDA toolkit module or export `CUDA_HOME` to the toolkit root before building. `nvcc --version` must work in the same activated environment where you run `python setup.py build_ext --inplace`.
- `No module named Laplace_subtraction_cuda`: build the extension in `libs/laplacianformer` with `python setup.py build_ext --inplace`.
- `no kernel image is available`: rebuild with `TORCH_CUDA_ARCH_LIST="12.1"` on the GB10 server.
- `Got torch.bfloat16`: use `trainer.precision=16-mixed` or `trainer.precision=32-true`.
- `NewtonInverse CUDA path does not support N=...`: the authors' kernel supports landmark matrices up to `N <= 128`; the provided PVT configs stay within that.
- `torch.compile` errors: rerun with `trainer.compile=false`.
