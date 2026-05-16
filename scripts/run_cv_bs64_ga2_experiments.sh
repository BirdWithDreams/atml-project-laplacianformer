#!/usr/bin/env bash

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DEFAULT_EXPERIMENTS=(
  cv_cifar100_vanilla_tiny_bs64_ga2_lr1e_4_wd1e_2
  cv_imagenet_vanilla_tiny_bs64_ga2_lr1e_4_wd1e_2
  cv_cifar100_laplacian_tiny_bs64_ga2_adamw_cv_default
  cv_imagenet_laplacian_tiny_bs64_ga2_lr4e_4_wd1e_2
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
WANDB_PROJECT="${WANDB_PROJECT:-}"
RUN_PREFIX="${RUN_PREFIX:-}"
EXTRA_ARGS=("$@")
FAILED_RUNS=()

for experiment in "${EXPERIMENT_LIST[@]}"; do
  LOGGER_ARGS=()
  if [ -n "${WANDB_PROJECT}" ]; then
    LOGGER_ARGS+=(logger.project="${WANDB_PROJECT}")
  fi

  if [ -n "${RUN_PREFIX}" ]; then
    run_name="${RUN_PREFIX}_${experiment}_seed${SEED}"
    LOGGER_ARGS+=(logger.name="${run_name}")
  else
    run_name="${experiment}_seed${SEED}"
  fi

  echo "================================================================"
  echo "Starting ${run_name}"
  echo "================================================================"

  CMD=(
    "${TRAIN_CMD_LIST[@]}" train.py
    "+experiment=${experiment}"
    seed="${SEED}"
    "${LOGGER_ARGS[@]}"
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
echo "All CV bs64/ga2 experiments completed successfully."
