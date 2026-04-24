#!/usr/bin/env bash

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODELS=(
  vanilla_tiny
  vanilla_small
  vanilla_medium
  laplacian_tiny
  laplacian_small
  laplacian_medium
)

OPTIMIZERS=(
  adamw_text_default
  adamw_text_high_lr
  adam_text_baseline
)

DATASETS=(
  conll2003
  ontonotes5
)

MAX_EPOCHS="${MAX_EPOCHS:-5}"
ACCELERATOR="${ACCELERATOR:-gpu}"
DEVICES="${DEVICES:-1}"
PRECISION="${PRECISION:-bf16-mixed}"
WANDB_PROJECT="${WANDB_PROJECT:-ner-model-matrix}"
PYTHON_BIN="${PYTHON_BIN:-python}"

EXTRA_ARGS=("$@")
FAILED_RUNS=()

for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    for optimizer in "${OPTIMIZERS[@]}"; do
      run_name="ner_${dataset}_${model}_${optimizer}"
      echo "================================================================"
      echo "Starting ${run_name}"
      echo "================================================================"

      if ! "${PYTHON_BIN}" train.py \
        task=ner_task \
        datamodule="${dataset}" \
        model="${model}" \
        optimizer="${optimizer}" \
        trainer.max_epochs="${MAX_EPOCHS}" \
        trainer.accelerator="${ACCELERATOR}" \
        trainer.devices="${DEVICES}" \
        trainer.precision="${PRECISION}" \
        logger.project="${WANDB_PROJECT}" \
        logger.name="${run_name}" \
        "${EXTRA_ARGS[@]}"; then
        echo "Run failed: ${run_name}"
        FAILED_RUNS+=("${run_name}")
      fi
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
echo "All NER runs completed successfully."
