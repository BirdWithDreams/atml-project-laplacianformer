#!/usr/bin/env bash

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DEFAULT_DATASETS=(
  sst2
  ag_news
)

BEST_CONFIGS=(
  "nlp_sst2_laplacian_medium_stable|7e-5|0.10|0.05"
  "nlp_sst2_vanilla_medium_stable|1e-4|0.10|0.05"
)

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

DRY_RUN="${DRY_RUN:-false}"
SEED="${SEED:-42}"
WANDB_PROJECT="${WANDB_PROJECT:-nlp-classification-best-stability}"
MAX_EPOCHS="${MAX_EPOCHS:-12}"
RUN_PREFIX="${RUN_PREFIX:-stability}"
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

FAILED_RUNS=()
RUN_INDEX=0

for dataset in "${DATASET_LIST[@]}"; do
  for config in "${BEST_CONFIGS[@]}"; do
    IFS="|" read -r experiment lr dropout label_smoothing <<< "${config}"
    RUN_INDEX=$((RUN_INDEX + 1))

    lr_tag="${lr//./p}"
    lr_tag="${lr_tag//-/m}"
    dropout_tag="${dropout//./p}"
    label_smoothing_tag="${label_smoothing//./p}"
    run_name="${RUN_PREFIX}_${dataset}_${experiment}_lr${lr_tag}_drop${dropout_tag}_ls${label_smoothing_tag}_seed${SEED}"

    if [ "${RUN_INDEX}" -le "${SKIP_FIRST_N}" ]; then
      echo "Skipping ${RUN_INDEX}: ${run_name}"
      continue
    fi

    echo "================================================================"
    echo "Starting ${run_name}"
    echo "================================================================"

    CMD=(
      "${TRAIN_CMD_LIST[@]}" train.py
      "+experiment=${experiment}"
      datamodule="${dataset}"
      optimizer.lr="${lr}"
      model.dropout="${dropout}"
      task.label_smoothing="${label_smoothing}"
      trainer.max_epochs="${MAX_EPOCHS}"
      seed="${SEED}"
      datamodule.subset_seed="${SEED}"
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

if [ "${#FAILED_RUNS[@]}" -gt 0 ]; then
  echo
  echo "Completed with failures:"
  printf '  - %s\n' "${FAILED_RUNS[@]}"
  exit 1
fi

echo
echo "All best NLP stability configs completed successfully."
