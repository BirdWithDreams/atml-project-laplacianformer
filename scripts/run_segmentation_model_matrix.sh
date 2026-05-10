#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

to_csv() {
  local value="$1"
  value="${value//,/ }"
  read -r -a items <<< "${value}"
  local IFS=,
  echo "${items[*]}"
}

MODELS="$(to_csv "${MODELS:-laplacian_pvt_medium_cuda laplacian_pvt_small_cuda vanilla_pvt_small vanilla_pvt_medium}")"
OPTIMIZERS="$(to_csv "${OPTIMIZERS:-adamw_segmentation_poly}")"
DATASETS="$(to_csv "${DATASETS:-cityscapes_segmentation voc2012_segmentation}")"

ACCELERATOR="${ACCELERATOR:-gpu}"
DEVICES="${DEVICES:-1}"
PRECISION="${PRECISION:-32}"
COMPILE="${COMPILE:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-segmentation-model-matrix}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-2}"

if [ -n "${TRAIN_CMD:-}" ]; then
  read -r -a TRAIN_CMD_LIST <<< "${TRAIN_CMD}"
elif command -v uv >/dev/null 2>&1; then
  TRAIN_CMD_LIST=(uv run python)
else
  TRAIN_CMD_LIST=(python)
fi

GRAD_ACCUM_ARGS=()
if [ -n "${ACCUMULATE_GRAD_BATCHES}" ]; then
  GRAD_ACCUM_ARGS=(trainer.accumulate_grad_batches="${ACCUMULATE_GRAD_BATCHES}")
fi

"${TRAIN_CMD_LIST[@]}" train.py -m \
  task=semantic_segmentation \
  datamodule="${DATASETS}" \
  model="${MODELS}" \
  optimizer="${OPTIMIZERS}" \
  trainer.accelerator="${ACCELERATOR}" \
  trainer.devices="${DEVICES}" \
  trainer.precision="${PRECISION}" \
  trainer.compile="${COMPILE}" \
  "${GRAD_ACCUM_ARGS[@]}" \
  logger.project="${WANDB_PROJECT}" \
  'logger.name=seg_${hydra:runtime.choices.datamodule}_${hydra:runtime.choices.model}_${hydra:runtime.choices.optimizer}' \
  "$@"
