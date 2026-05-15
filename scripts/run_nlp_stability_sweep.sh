#!/usr/bin/env bash

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DEFAULT_EXPERIMENTS=(
  nlp_sst2_vanilla_medium_stable
  nlp_sst2_laplacian_medium_stable
)

LR_LIST_DEFAULT="7e-5 1e-4"
DROPOUT_LIST_DEFAULT="0.10"
LABEL_SMOOTHING_LIST_DEFAULT="0.00 0.05"
DATASETS_DEFAULT="sst2 ag_news"

if [ -n "${EXPERIMENTS:-}" ]; then
  read -r -a EXPERIMENT_LIST <<< "${EXPERIMENTS}"
else
  EXPERIMENT_LIST=("${DEFAULT_EXPERIMENTS[@]}")
fi

read -r -a LR_LIST <<< "${LR_LIST:-${LR_LIST_DEFAULT}}"
read -r -a DROPOUT_LIST <<< "${DROPOUT_LIST:-${DROPOUT_LIST_DEFAULT}}"
read -r -a LABEL_SMOOTHING_LIST <<< "${LABEL_SMOOTHING_LIST:-${LABEL_SMOOTHING_LIST_DEFAULT}}"
read -r -a DATASET_LIST <<< "${DATASETS:-${DATASETS_DEFAULT}}"

if [ -n "${TRAIN_CMD:-}" ]; then
  read -r -a TRAIN_CMD_LIST <<< "${TRAIN_CMD}"
elif command -v uv >/dev/null 2>&1; then
  TRAIN_CMD_LIST=(uv run python)
else
  TRAIN_CMD_LIST=(python)
fi

DRY_RUN="${DRY_RUN:-false}"
SEED="${SEED:-42}"
WANDB_PROJECT="${WANDB_PROJECT:-nlp-classification-recommended}"
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
  for experiment in "${EXPERIMENT_LIST[@]}"; do
    for lr in "${LR_LIST[@]}"; do
      for dropout in "${DROPOUT_LIST[@]}"; do
        for label_smoothing in "${LABEL_SMOOTHING_LIST[@]}"; do
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
echo "All NLP stability sweep runs completed successfully."
