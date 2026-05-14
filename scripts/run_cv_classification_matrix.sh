#!/usr/bin/env bash

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DEFAULT_MODELS=(
  vanilla_pvt_small
  laplacian_pvt_small_cuda
)

DEFAULT_OPTIMIZERS=(
  adamw_cv_default
)

DEFAULT_DATASETS=(
  cifar100
  imagenet
)

ACCELERATOR="${ACCELERATOR:-gpu}"
DEVICES="${DEVICES:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-cv-classification-recommended}"
COMPILE="${COMPILE:-false}"

# --- Environment Overrides ---
if [ -n "${MODELS:-}" ]; then read -r -a MODEL_LIST <<< "${MODELS}"; else MODEL_LIST=("${DEFAULT_MODELS[@]}"); fi
if [ -n "${OPTIMIZERS:-}" ]; then read -r -a OPTIMIZER_LIST <<< "${OPTIMIZERS}"; else OPTIMIZER_LIST=("${DEFAULT_OPTIMIZERS[@]}"); fi
if [ -n "${DATASETS:-}" ]; then read -r -a DATASET_LIST <<< "${DATASETS}"; else DATASET_LIST=("${DEFAULT_DATASETS[@]}"); fi

if [ -n "${TRAIN_CMD:-}" ]; then
  read -r -a TRAIN_CMD_LIST <<< "${TRAIN_CMD}"
elif command -v uv >/dev/null 2>&1; then
  TRAIN_CMD_LIST=(uv run python)
else
  TRAIN_CMD_LIST=(python)
fi

EXTRA_ARGS=("$@")
FAILED_RUNS=()

dataset_args() {
  case "$1" in
    cifar100)
      DATASET_ARGS=(
        datamodule=cifar100
        datamodule.batch_size="${CIFAR_BATCH_SIZE:-32}"
        datamodule.num_workers="${CIFAR_NUM_WORKERS:-4}"
        datamodule.num_classes=100
        model.img_size="${CIFAR_IMG_SIZE:-320}"
        optimizer.lr="${CIFAR_LR:-3e-4}"
        optimizer.weight_decay="${CIFAR_WEIGHT_DECAY:-0.05}"
        trainer.max_epochs="${CIFAR_MAX_EPOCHS:-80}"
        trainer.accumulate_grad_batches="${CIFAR_ACCUMULATE_GRAD_BATCHES:-4}"
        logger.extra_tags="[cv,cifar100,small,recommended]"
      )
      ;;
    imagenet)
      DATASET_ARGS=(
        datamodule=imagenet
        ++datamodule.data_dir="${IMAGENET_DATA_DIR:-data/imagenet_subset}"
        datamodule.batch_size="${IMAGENET_BATCH_SIZE:-32}"
        datamodule.num_workers="${IMAGENET_NUM_WORKERS:-8}"
        datamodule.num_classes="${IMAGENET_NUM_CLASSES:-100}"
        model.img_size="${IMAGENET_IMG_SIZE:-320}"
        optimizer.lr="${IMAGENET_LR:-2e-4}"
        optimizer.weight_decay="${IMAGENET_WEIGHT_DECAY:-0.05}"
        trainer.max_epochs="${IMAGENET_MAX_EPOCHS:-60}"
        trainer.accumulate_grad_batches="${IMAGENET_ACCUMULATE_GRAD_BATCHES:-4}"
        logger.extra_tags="[cv,imagenet_subset,small,recommended]"
      )
      ;;
    *)
      echo "Unknown CV dataset profile: $1" >&2
      return 1
      ;;
  esac
}

model_args() {
  local dataset="$1"
  local model="$2"
  local config_name="$3"

  MODEL_ARGS=(model="${model}")
  MODEL_CONFIG_LABEL="base"

  if [[ "${model}" != *"laplacian"* ]]; then
    MODEL_ARGS+=(trainer.precision="${PRECISION:-16}")
    return 0
  fi

  MODEL_ARGS+=(
    trainer.precision=32
    ++optimizer.lr="${LAPLACIAN_LR:-1e-5}"
    +trainer.gradient_clip_val="${LAPLACIAN_GRADIENT_CLIP_VAL:-0.1}"
  )

  case "${dataset}:${config_name}" in
    cifar100:stable)
      MODEL_CONFIG_LABEL="L4_P8421_NS5"
      MODEL_ARGS+=(model.lambda_scale=4.0 model.ns_iters=5 'model.pool_ratios=[8,4,2,1]')
      ;;
    cifar100:fine)
      MODEL_CONFIG_LABEL="L8_P4211_NS5"
      MODEL_ARGS+=(model.lambda_scale=8.0 model.ns_iters=5 'model.pool_ratios=[4,2,1,1]')
      ;;
    imagenet:stable)
      MODEL_CONFIG_LABEL="L4_P8421_NS5"
      MODEL_ARGS+=(model.lambda_scale=4.0 model.ns_iters=5 'model.pool_ratios=[8,4,2,1]')
      ;;
    imagenet:wide)
      MODEL_CONFIG_LABEL="L8_P8421_NS6"
      MODEL_ARGS+=(model.lambda_scale=8.0 model.ns_iters=6 'model.pool_ratios=[8,4,2,1]')
      ;;
    *)
      echo "Unknown Laplacian config profile: ${dataset}:${config_name}" >&2
      return 1
      ;;
  esac
}

laplacian_configs() {
  case "$1" in
    cifar100)
      LAPLACIAN_CONFIG_LIST=(stable fine)
      ;;
    imagenet)
      LAPLACIAN_CONFIG_LIST=(stable wide)
      ;;
    *)
      echo "Unknown CV dataset profile: $1" >&2
      return 1
      ;;
  esac
}

for dataset in "${DATASET_LIST[@]}"; do
  if ! dataset_args "${dataset}"; then
    FAILED_RUNS+=("${dataset}_profile")
    continue
  fi

  if ! laplacian_configs "${dataset}"; then
    FAILED_RUNS+=("${dataset}_laplacian_profiles")
    continue
  fi

  for optimizer in "${OPTIMIZER_LIST[@]}"; do
    for model in "${MODEL_LIST[@]}"; do
      if [[ "${model}" == *"laplacian"* ]]; then
        CONFIG_LIST=("${LAPLACIAN_CONFIG_LIST[@]}")
      else
        CONFIG_LIST=(base)
      fi

      for config_name in "${CONFIG_LIST[@]}"; do
        if ! model_args "${dataset}" "${model}" "${config_name}"; then
          FAILED_RUNS+=("${dataset}_${model}_${config_name}_profile")
          continue
        fi

        run_name="cv_${dataset}_${model}_${MODEL_CONFIG_LABEL}_${optimizer}"

        echo "================================================================"
        echo "Starting ${run_name}"
        echo "================================================================"

        if ! "${TRAIN_CMD_LIST[@]}" train.py \
          task=cv_classification \
          optimizer="${optimizer}" \
          "${DATASET_ARGS[@]}" \
          "${MODEL_ARGS[@]}" \
          trainer.accelerator="${ACCELERATOR}" \
          trainer.devices="${DEVICES}" \
          trainer.compile="${COMPILE}" \
          logger.project="${WANDB_PROJECT}" \
          logger.name="${run_name}" \
          "${EXTRA_ARGS[@]}"; then
          echo "Run failed: ${run_name}"
          FAILED_RUNS+=("${run_name}")
        fi

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
echo "All CV runs completed successfully."
