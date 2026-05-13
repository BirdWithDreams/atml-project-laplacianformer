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
  adamw_text_scratch
)

DEFAULT_DATASETS=(
  ag_news
  sst2
)

ACCELERATOR="${ACCELERATOR:-gpu}"
DEVICES="${DEVICES:-1}"
PRECISION="${PRECISION:-32}"
COMPILE="${COMPILE:-false}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-2}"
WANDB_PROJECT="${WANDB_PROJECT:-nlp-classification-model-matrix}"
DRY_RUN="${DRY_RUN:-false}"
SEED="${SEED:-42}"
SKIP_FIRST_N=0
EXTRA_ARGS=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --skip)
      if [ "$#" -lt 2 ]; then
        echo "--skip requires a non-negative integer argument"
        exit 1
      fi
      SKIP_FIRST_N="$2"
      shift 2
      ;;
    --skip=*)
      SKIP_FIRST_N="${1#--skip=}"
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if ! [[ "${SKIP_FIRST_N}" =~ ^[0-9]+$ ]]; then
  echo "--skip must be a non-negative integer, got: ${SKIP_FIRST_N}"
  exit 1
fi

if [ -n "${MODELS:-}" ]; then
  read -r -a MODEL_LIST <<< "${MODELS}"
else
  MODEL_LIST=("${DEFAULT_MODELS[@]}")
fi

if [ -n "${OPTIMIZERS:-}" ]; then
  read -r -a OPTIMIZER_LIST <<< "${OPTIMIZERS}"
else
  OPTIMIZER_LIST=("${DEFAULT_OPTIMIZERS[@]}")
fi

if [ -n "${DATASETS:-}" ]; then
  read -r -a DATASET_LIST <<< "${DATASETS}"
else
  DATASET_LIST=("${DEFAULT_DATASETS[@]}")
fi

if [ -n "${TRAIN_CMD:-}" ]; then
  read -r -a TRAIN_CMD_LIST <<< "${TRAIN_CMD}"
elif command -v uv >/dev/null 2>&1; then
  TRAIN_CMD_LIST=(uv run python)
else
  TRAIN_CMD_LIST=(python)
fi

TRAINER_ARGS=(
  trainer.accelerator="${ACCELERATOR}"
  trainer.devices="${DEVICES}"
  trainer.precision="${PRECISION}"
  trainer.compile="${COMPILE}"
  trainer.max_epochs="${MAX_EPOCHS}"
)

if [ -n "${ACCUMULATE_GRAD_BATCHES}" ]; then
  TRAINER_ARGS+=(trainer.accumulate_grad_batches="${ACCUMULATE_GRAD_BATCHES}")
fi

DATAMODULE_ARGS=()
if [ -n "${BATCH_SIZE:-}" ]; then
  DATAMODULE_ARGS+=(datamodule.batch_size="${BATCH_SIZE}")
fi
if [ -n "${MAX_LENGTH:-}" ]; then
  DATAMODULE_ARGS+=(datamodule.max_length="${MAX_LENGTH}")
fi
if [ -n "${VALIDATION_SIZE:-}" ]; then
  DATAMODULE_ARGS+=(datamodule.validation_size="${VALIDATION_SIZE}")
fi

FAILED_RUNS=()
RUN_INDEX=0

for dataset in "${DATASET_LIST[@]}"; do
  for optimizer in "${OPTIMIZER_LIST[@]}"; do
    for model in "${MODEL_LIST[@]}"; do
      RUN_INDEX=$((RUN_INDEX + 1))
      if [ "${RUN_INDEX}" -le "${SKIP_FIRST_N}" ]; then
        echo "Skipping ${RUN_INDEX}: ${dataset}_${model}_${optimizer}"
        continue
      fi

      run_name="nlp_${dataset}_${model}_${optimizer}"
      echo "================================================================"
      echo "Starting ${run_name}"
      echo "================================================================"

      CMD=(
        "${TRAIN_CMD_LIST[@]}" train.py
        task=nlp_classification
        datamodule="${dataset}"
        model="${model}"
        optimizer="${optimizer}"
        seed="${SEED}"
        datamodule.subset_seed="${SEED}"
        "${TRAINER_ARGS[@]}"
        "${DATAMODULE_ARGS[@]}"
        logger.project="${WANDB_PROJECT}"
        logger.name="${run_name}"
        "${EXTRA_ARGS[@]}"
      )

      if [ "${DRY_RUN}" = "true" ]; then
        printf 'Dry run:'
        printf ' %q' "${CMD[@]}"
        printf '\n'
        continue
      fi

      if ! "${CMD[@]}"; then
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
echo "All NLP classification runs completed successfully."
