#!/usr/bin/env bash

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DEFAULT_EXPERIMENTS=(
  vanilla_deep_medium
  vanilla_large_width
  vanilla_large_deep
  laplacian_lambda8_pool2
  laplacian_lambda16_pool2
  laplacian_lambda8_pool1
)

DEFAULT_DATASETS=(
  conll2003
  ontonotes5
)

ACCELERATOR="${ACCELERATOR:-gpu}"
DEVICES="${DEVICES:-1}"
PRECISION="${PRECISION:-32}"
WANDB_PROJECT="${WANDB_PROJECT:-ner-model-matrix}"
GEN_TAG="${GEN_TAG:-gen_2}"
COMPILE="${COMPILE:-false}"
VANILLA_EPOCHS="${VANILLA_EPOCHS:-40}"
LAPLACIAN_CONLL_EPOCHS="${LAPLACIAN_CONLL_EPOCHS:-50}"
LAPLACIAN_ONTONOTES_EPOCHS="${LAPLACIAN_ONTONOTES_EPOCHS:-60}"

if [ -n "${EXPERIMENTS:-}" ]; then
  read -r -a EXPERIMENT_LIST <<< "${EXPERIMENTS}"
else
  EXPERIMENT_LIST=("${DEFAULT_EXPERIMENTS[@]}")
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

for dataset in "${DATASET_LIST[@]}"; do
  for experiment in "${EXPERIMENT_LIST[@]}"; do
    datamodule="${dataset}"
    optimizer="adam_text_baseline"
    max_epochs="${VANILLA_EPOCHS}"
    overrides=()

    case "${experiment}" in
      vanilla_deep_medium)
        model="vanilla_1d_medium_d8"
        optimizer="adam_text_lr7e_5"
        ;;
      vanilla_large_width)
        model="vanilla_1d_large_1024"
        optimizer="adam_text_lr7e_5"
        overrides=(datamodule.batch_size=128)
        ;;
      vanilla_large_deep)
        model="vanilla_1d_large_1024_d8"
        optimizer="adam_text_lr5e_5"
        overrides=(datamodule.batch_size=64)
        ;;
      laplacian_lambda8_pool2)
        model="laplacian_1d_cuda_medium_lambda8_pool2_ns8"
        if [ "${dataset}" = "ontonotes5" ]; then
          max_epochs="${LAPLACIAN_ONTONOTES_EPOCHS}"
        else
          max_epochs="${LAPLACIAN_CONLL_EPOCHS}"
        fi
        ;;
      laplacian_lambda16_pool2)
        model="laplacian_1d_cuda_medium_lambda16_pool2_ns8"
        if [ "${dataset}" = "ontonotes5" ]; then
          max_epochs="${LAPLACIAN_ONTONOTES_EPOCHS}"
        else
          max_epochs="${LAPLACIAN_CONLL_EPOCHS}"
        fi
        ;;
      laplacian_lambda8_pool1)
        model="laplacian_1d_cuda_medium_lambda8_pool1_ns8"
        overrides=(datamodule.batch_size=128)
        if [ "${dataset}" = "ontonotes5" ]; then
          max_epochs="${LAPLACIAN_ONTONOTES_EPOCHS}"
        else
          max_epochs="${LAPLACIAN_CONLL_EPOCHS}"
        fi
        ;;
      *)
        echo "Unknown experiment: ${experiment}"
        FAILED_RUNS+=("${dataset}_${experiment}")
        continue
        ;;
    esac

    run_name="ner_${GEN_TAG}_${dataset}_${experiment}"
    echo "================================================================"
    echo "Starting ${run_name}"
    echo "================================================================"

    if ! "${TRAIN_CMD_LIST[@]}" train.py \
      task=ner_task \
      datamodule="${datamodule}" \
      model="${model}" \
      optimizer="${optimizer}" \
      trainer.accelerator="${ACCELERATOR}" \
      trainer.devices="${DEVICES}" \
      trainer.precision="${PRECISION}" \
      trainer.compile="${COMPILE}" \
      trainer.max_epochs="${max_epochs}" \
      logger.project="${WANDB_PROJECT}" \
      logger.name="${run_name}" \
      logger.extra_tags="[${GEN_TAG}]" \
      "${overrides[@]}" \
      "${EXTRA_ARGS[@]}"; then
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
echo "All gen-2 NER runs completed successfully."
