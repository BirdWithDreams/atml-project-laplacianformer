#!/usr/bin/env bash

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DEFAULT_MODELS=(
  vanilla_1d_small
  vanilla_1d_medium
  laplacian_1d_cuda_small
  laplacian_1d_cuda_medium
)

DEFAULT_OPTIMIZERS=(
  adamw_text_default
)

DEFAULT_DATASETS=(
  wikitext2
  seq2seq
)

# Laplacian-specific defaults
DEFAULT_LAMBDAS=(4 8)
DEFAULT_POOL_RATIOS=(2 4)

# Hardware & Training Defaults
ACCELERATOR="${ACCELERATOR:-gpu}"
DEVICES="${DEVICES:-1}"            # Single GPU is safer for custom CUDA kernels
PRECISION="${PRECISION:-32}"        # Laplacian CUDA requires 32-bit
COMPILE="${COMPILE:-false}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"       # OOM-safe default for text models
WANDB_PROJECT="${WANDB_PROJECT:-text-gen-model-matrix}"
SEED="${SEED:-42}"

# --- Environment Overrides ---
if [ -n "${MODELS:-}" ]; then read -r -a MODEL_LIST <<< "${MODELS}"; else MODEL_LIST=("${DEFAULT_MODELS[@]}"); fi
if [ -n "${OPTIMIZERS:-}" ]; then read -r -a OPTIMIZER_LIST <<< "${OPTIMIZERS}"; else OPTIMIZER_LIST=("${DEFAULT_OPTIMIZERS[@]}"); fi
if [ -n "${DATASETS:-}" ]; then read -r -a DATASET_LIST <<< "${DATASETS}"; else DATASET_LIST=("${DEFAULT_DATASETS[@]}"); fi
if [ -n "${LAMBDAS:-}" ]; then read -r -a LAMBDA_LIST <<< "${LAMBDAS}"; else LAMBDA_LIST=("${DEFAULT_LAMBDAS[@]}"); fi
if [ -n "${POOLS:-}" ]; then read -r -a POOL_LIST <<< "${POOLS}"; else POOL_LIST=("${DEFAULT_POOL_RATIOS[@]}"); fi

if [ -n "${TRAIN_CMD:-}" ]; then
  read -r -a TRAIN_CMD_LIST <<< "${TRAIN_CMD}"
elif command -v uv >/dev/null 2>&1; then
  TRAIN_CMD_LIST=(uv run python)
else
  TRAIN_CMD_LIST=(python)
fi

EXTRA_ARGS=("$@")
FAILED_RUNS=()
GRAD_ACCUM_ARGS=()
if [ -n "${ACCUMULATE_GRAD_BATCHES}" ]; then
  GRAD_ACCUM_ARGS=(trainer.accumulate_grad_batches="${ACCUMULATE_GRAD_BATCHES}")
fi

for dataset in "${DATASET_LIST[@]}"; do
  for optimizer in "${OPTIMIZER_LIST[@]}"; do
    for model in "${MODEL_LIST[@]}"; do
      
      # Determine stability parameters
      if [[ "${model}" == *"laplacian"* ]]; then
        CURRENT_LAMBDAS=("${LAMBDA_LIST[@]}")
        CURRENT_POOLS=("${POOL_LIST[@]}")
        # Stability Fixes for custom CUDA kernels
        STABILITY_ARGS=("++optimizer.lr=5e-5" "+trainer.gradient_clip_val=1.0")
      else
        CURRENT_LAMBDAS=("N/A")
        CURRENT_POOLS=("N/A")
        STABILITY_ARGS=()
      fi

      for lambd in "${CURRENT_LAMBDAS[@]}"; do
        for pool in "${CURRENT_POOLS[@]}"; do
          
          MODEL_ARGS=(model="${model}")
          run_name="gen_${dataset}_${model}_${optimizer}"

          # Append Laplacian args with proper Hydra '++' overrides
          if [[ "${model}" == *"laplacian"* ]]; then
            MODEL_ARGS+=("++model.lambda_scale=${lambd}" "++model.pool_ratio=${pool}")
            run_name="gen_${dataset}_${model}_L${lambd}_P${pool}_${optimizer}"
          fi

          echo "================================================================"
          echo "Starting ${run_name}"
          echo "================================================================"

          if ! "${TRAIN_CMD_LIST[@]}" train.py \
            task=generation_nlp \
            datamodule="${dataset}" \
            datamodule.batch_size="${BATCH_SIZE}" \
            "${MODEL_ARGS[@]}" \
            "${STABILITY_ARGS[@]}" \
            optimizer="${optimizer}" \
            trainer.accelerator="${ACCELERATOR}" \
            trainer.devices="${DEVICES}" \
            trainer.precision="${PRECISION}" \
            trainer.compile="${COMPILE}" \
            "${GRAD_ACCUM_ARGS[@]}" \
            logger.project="${WANDB_PROJECT}" \
            logger.name="${run_name}" \
            seed="${SEED}" \
            "${EXTRA_ARGS[@]}"; then
            echo "Run failed: ${run_name}"
            FAILED_RUNS+=("${run_name}")
          fi

        done
      done
    done
  done
done

if [ "${#FAILED_RUNS[@]}" -gt 0 ]; then
  echo
  echo "Completed with failures:"
  printf '  - %s\n' "${FAILED_RUNS[@]}"
  exit 1
fi

echo
echo "All Text Generation runs completed successfully."