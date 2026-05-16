#!/usr/bin/env bash

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DEFAULT_DATASETS=(
  conll2003
  ontonotes5
)

DEFAULT_MODELS=(
  vanilla_1d_tiny
  vanilla_1d_small
  laplacian_1d_cuda_tiny
  laplacian_1d_cuda_small
)

if [ -n "${DATASETS:-}" ]; then
  read -r -a DATASET_LIST <<< "${DATASETS}"
else
  DATASET_LIST=("${DEFAULT_DATASETS[@]}")
fi

if [ -n "${MODELS:-}" ]; then
  read -r -a MODEL_LIST <<< "${MODELS}"
else
  MODEL_LIST=("${DEFAULT_MODELS[@]}")
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
WANDB_PROJECT="${WANDB_PROJECT:-ner-final}"
RUN_PREFIX="${RUN_PREFIX:-final_ner}"
EXTRA_ARGS=("$@")
FAILED_RUNS=()

base_experiment_for_dataset() {
  case "$1" in
    conll2003)
      printf 'ner_conll2003_final'
      ;;
    ontonotes5)
      printf 'ner_ontonotes5_final'
      ;;
    *)
      echo "Unknown final NER dataset: $1" >&2
      return 1
      ;;
  esac
}

model_type_for_model() {
  local model="$1"
  if [[ "${model}" == vanilla_* ]]; then
    printf 'vanilla'
  elif [[ "${model}" == laplacian_* ]]; then
    printf 'laplacian'
  else
    printf 'unknown'
  fi
}

model_size_for_model() {
  local model="$1"
  if [[ "${model}" == *"_tiny"* ]]; then
    printf 'tiny'
  elif [[ "${model}" == *"_small"* ]]; then
    printf 'small'
  elif [[ "${model}" == *"_medium"* ]]; then
    printf 'medium'
  else
    printf 'unknown'
  fi
}

for dataset in "${DATASET_LIST[@]}"; do
  if ! experiment="$(base_experiment_for_dataset "${dataset}")"; then
    FAILED_RUNS+=("${dataset}_unknown")
    continue
  fi

  for model in "${MODEL_LIST[@]}"; do
    model_type="$(model_type_for_model "${model}")"
    model_size="$(model_size_for_model "${model}")"
    run_name="${RUN_PREFIX}_${dataset}_${model}_seed${SEED}"

    echo "================================================================"
    echo "Starting ${run_name}"
    echo "================================================================"

    CMD=(
      "${TRAIN_CMD_LIST[@]}" train.py
      "+experiment=${experiment}"
      model="${model}"
      seed="${SEED}"
      logger.project="${WANDB_PROJECT}"
      logger.name="${run_name}"
      "logger.extra_tags=['final:ner','dataset:${dataset}','model_type:${model_type}','model_size:${model_size}']"
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
echo "All final NER configs completed successfully."
