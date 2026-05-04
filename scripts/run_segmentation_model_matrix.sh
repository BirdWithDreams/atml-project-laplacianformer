#!/usr/bin/env bash

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DEFAULT_MODELS=(
  vanilla_pvt_small
  vanilla_pvt_medium
  laplacian_pvt_small_cuda
  laplacian_pvt_medium_cuda
)

DEFAULT_OPTIMIZERS=(
  adamw_cv_default
)

DEFAULT_DATASETS=(
  voc2012_segmentation
  coco_segmentation
)

ACCELERATOR="${ACCELERATOR:-gpu}"
DEVICES="${DEVICES:-1}"
PRECISION="${PRECISION:-32}"
COMPILE="${COMPILE:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-segmentation-model-matrix}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-8}"
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

EXTRA_ARGS=("$@")
FAILED_RUNS=()
RUN_INDEX=0
GRAD_ACCUM_ARGS=()
if [ -n "${ACCUMULATE_GRAD_BATCHES}" ]; then
  GRAD_ACCUM_ARGS=(trainer.accumulate_grad_batches="${ACCUMULATE_GRAD_BATCHES}")
fi

for dataset in "${DATASET_LIST[@]}"; do
  for optimizer in "${OPTIMIZER_LIST[@]}"; do
    for model in "${MODEL_LIST[@]}"; do
      RUN_INDEX=$((RUN_INDEX + 1))
      if [ "${RUN_INDEX}" -le "${SKIP_FIRST_N}" ]; then
        echo "Skipping ${RUN_INDEX}: ${dataset}_${model}_${optimizer}"
        continue
      fi

      run_name="seg_${dataset}_${model}_${optimizer}"
      echo "================================================================"
      echo "Starting ${run_name}"
      echo "================================================================"

      if ! "${TRAIN_CMD_LIST[@]}" train.py \
        task=semantic_segmentation \
        datamodule="${dataset}" \
        model="${model}" \
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

if [ "${#FAILED_RUNS[@]}" -gt 0 ]; then
  echo
  echo "Completed with failures:"
  printf '  - %s\n' "${FAILED_RUNS[@]}"
  exit 1
fi

echo
echo "All segmentation runs completed successfully."
