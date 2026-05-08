#!/usr/bin/env bash

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DEFAULT_EXPERIMENTS=(
  nlp_sst2_vanilla_medium_stable
  nlp_sst2_laplacian_medium_stable
  nlp_sst2_vanilla_large_regularized
  nlp_ag_news_laplacian_medium_stable
)

if [ -n "${EXPERIMENTS:-}" ]; then
  read -r -a EXPERIMENT_LIST <<< "${EXPERIMENTS}"
else
  EXPERIMENT_LIST=("${DEFAULT_EXPERIMENTS[@]}")
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
WANDB_PROJECT="${WANDB_PROJECT:-nlp-classification-recommended}"
RUN_PREFIX="${RUN_PREFIX:-recommended}"
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

COMMON_ARGS=(
  seed="${SEED}"
  datamodule.subset_seed="${SEED}"
  logger.project="${WANDB_PROJECT}"
)

if [ -n "${MAX_EPOCHS:-}" ]; then
  COMMON_ARGS+=(trainer.max_epochs="${MAX_EPOCHS}")
fi
if [ -n "${ACCUMULATE_GRAD_BATCHES:-}" ]; then
  COMMON_ARGS+=(trainer.accumulate_grad_batches="${ACCUMULATE_GRAD_BATCHES}")
fi
if [ -n "${DEVICES:-}" ]; then
  COMMON_ARGS+=(trainer.devices="${DEVICES}")
fi
if [ -n "${ACCELERATOR:-}" ]; then
  COMMON_ARGS+=(trainer.accelerator="${ACCELERATOR}")
fi

FAILED_RUNS=()
RUN_INDEX=0

for experiment in "${EXPERIMENT_LIST[@]}"; do
  RUN_INDEX=$((RUN_INDEX + 1))
  if [ "${RUN_INDEX}" -le "${SKIP_FIRST_N}" ]; then
    echo "Skipping ${RUN_INDEX}: ${experiment}"
    continue
  fi

  run_name="${RUN_PREFIX}_${experiment}_seed${SEED}"
  echo "================================================================"
  echo "Starting ${run_name}"
  echo "================================================================"

  CMD=(
    "${TRAIN_CMD_LIST[@]}" train.py
    "+experiment=${experiment}"
    "${COMMON_ARGS[@]}"
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

if [ "${#FAILED_RUNS[@]}" -gt 0 ]; then
  echo
  echo "Completed with failures:"
  printf '  - %s\n' "${FAILED_RUNS[@]}"
  exit 1
fi

echo
echo "All recommended NLP runs completed successfully."
