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
)

# Laplacian-specific defaults[cite: 2]
DEFAULT_LAMBDAS=(4 8)
DEFAULT_POOL_RATIOS=(2 4)

ACCELERATOR="${ACCELERATOR:-gpu}"
DEVICES="${DEVICES:-1}"
PRECISION="${PRECISION:-32}"
WANDB_PROJECT="${WANDB_PROJECT:-text-gen-model-matrix}"
COMPILE="${COMPILE:-false}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-}"

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
      
      # Determine if we need to sweep Laplacian parameters[cite: 1, 2]
      if [[ "${model}" == *"laplacian"* ]]; then
        CURRENT_LAMBDAS=("${LAMBDA_LIST[@]}")
        CURRENT_POOLS=("${POOL_LIST[@]}")
      else
        # For vanilla models, run exactly once without lambda/pool sweeps
        CURRENT_LAMBDAS=("N/A")
        CURRENT_POOLS=("N/A")
      fi

      for lambd in "${CURRENT_LAMBDAS[@]}"; do
        for pool in "${CURRENT_POOLS[@]}"; do
          
          MODEL_ARGS=(model="${model}")
          run_name="gen_${dataset}_${model}_${optimizer}"

          # Append Laplacian args only if applicable
          if [[ "${model}" == *"laplacian"* ]]; then
            MODEL_ARGS+=(model.lambda_scale="${lambd}" model.pool_ratio="${pool}")
            run_name="gen_${dataset}_${model}_L${lambd}_P${pool}_${optimizer}"
          fi

          echo "================================================================"
          echo "Starting ${run_name}"
          echo "================================================================"

          if ! "${TRAIN_CMD_LIST[@]}" train.py \
            task=nlp_generation \
            datamodule="${dataset}" \
            "${MODEL_ARGS[@]}" \
            optimizer="${optimizer}" \
            trainer.accelerator="${ACCELERATOR}" \
            trainer.devices="${DEVICES}" \
            trainer.precision="${PRECISION}" \
            trainer.compile="${COMPILE}" \
            "${GRAD_ACCUM_ARGS[@]}" \
            logger.project="${WANDB_PROJECT}" \
            logger.name="${run_name}" \
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