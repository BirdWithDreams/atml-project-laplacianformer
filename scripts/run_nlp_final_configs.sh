#!/usr/bin/env bash

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DEFAULT_EXPERIMENTS=(
  nlp_sst2_final_vanilla
  nlp_sst2_final_laplacian
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
WANDB_PROJECT="${WANDB_PROJECT:-nlp-classification-model-matrix}"
RUN_PREFIX="${RUN_PREFIX:-final}"
EXTRA_ARGS=("$@")
FAILED_RUNS=()

for experiment in "${EXPERIMENT_LIST[@]}"; do
  run_name="${RUN_PREFIX}_${experiment}_seed${SEED}"

  echo "================================================================"
  echo "Starting ${run_name}"
  echo "================================================================"

  CMD=(
    "${TRAIN_CMD_LIST[@]}" train.py
    "+experiment=${experiment}"
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

if [ "${#FAILED_RUNS[@]}" -gt 0 ]; then
  echo
  echo "Completed with failures:"
  printf '  - %s\n' "${FAILED_RUNS[@]}"
  exit 1
fi

echo
echo "All final NLP configs completed successfully."
