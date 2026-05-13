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
  adamw_segmentation_poly
)

DEFAULT_DATASETS=(
  voc2012_segmentation
  cityscapes_segmentation
)

ACCELERATOR="${ACCELERATOR:-gpu}"
DEVICES="${DEVICES:-1}"
PRECISION="${PRECISION:-32}"
COMPILE="${COMPILE:-false}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-4}"
MAX_STEPS="${MAX_STEPS:-}"
WANDB_PROJECT="${WANDB_PROJECT:-segmentation-model-matrix}"
DRY_RUN="${DRY_RUN:-false}"
SEED="${SEED:-42}"

AUTO_BACKBONE_CHECKPOINTS="${AUTO_BACKBONE_CHECKPOINTS:-true}"
CV_BACKBONE_ROOT="${CV_BACKBONE_ROOT:-results/cv-model-matrix}"
CV_BACKBONE_DATASET="${CV_BACKBONE_DATASET:-imagenet}"
CV_BACKBONE_OPTIMIZER="${CV_BACKBONE_OPTIMIZER:-adamw_cv_default}"
CV_BACKBONE_CHECKPOINT_FILENAME="${CV_BACKBONE_CHECKPOINT_FILENAME:-last.ckpt}"
LAPLACIAN_CV_LAMBDA="${LAPLACIAN_CV_LAMBDA:-4}"
LAPLACIAN_CV_POOL="${LAPLACIAN_CV_POOL:-2}"
REQUIRE_PRETRAINED_BACKBONE="${REQUIRE_PRETRAINED_BACKBONE:-false}"

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

TRAINER_ARGS=(
  trainer.accelerator="${ACCELERATOR}"
  trainer.devices="${DEVICES}"
  trainer.precision="${PRECISION}"
  trainer.compile="${COMPILE}"
)

if [ -n "${ACCUMULATE_GRAD_BATCHES}" ]; then
  TRAINER_ARGS+=(trainer.accumulate_grad_batches="${ACCUMULATE_GRAD_BATCHES}")
fi
if [ -n "${MAX_STEPS}" ]; then
  TRAINER_ARGS+=(trainer.max_steps="${MAX_STEPS}")
  TRAINER_ARGS+=(optimizer.max_iters="${MAX_STEPS}")
fi

DATAMODULE_ARGS=()
if [ -n "${BATCH_SIZE:-}" ]; then
  DATAMODULE_ARGS+=(datamodule.batch_size="${BATCH_SIZE}")
fi
if [ -n "${NUM_WORKERS:-}" ]; then
  DATAMODULE_ARGS+=(datamodule.num_workers="${NUM_WORKERS}")
fi
if [ -n "${MAX_TRAIN_SAMPLES:-}" ]; then
  DATAMODULE_ARGS+=(datamodule.max_train_samples="${MAX_TRAIN_SAMPLES}")
fi
if [ -n "${MAX_VAL_SAMPLES:-}" ]; then
  DATAMODULE_ARGS+=(datamodule.max_val_samples="${MAX_VAL_SAMPLES}")
fi
if [ -n "${MAX_TEST_SAMPLES:-}" ]; then
  DATAMODULE_ARGS+=(datamodule.max_test_samples="${MAX_TEST_SAMPLES}")
fi

manual_checkpoint_for_model() {
  local model="$1"
  local checkpoint=""

  if [[ "${model}" == vanilla_* ]]; then
    checkpoint="${VANILLA_BACKBONE_CHECKPOINT_PATH:-${BACKBONE_CHECKPOINT_PATH:-}}"
  elif [[ "${model}" == laplacian_* ]]; then
    checkpoint="${LAPLACIAN_BACKBONE_CHECKPOINT_PATH:-${BACKBONE_CHECKPOINT_PATH:-}}"
  fi

  if [ -n "${checkpoint}" ]; then
    printf '%s' "${checkpoint}"
  fi
}

is_truthy() {
  case "${1,,}" in
    1|true|yes|on)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

cv_run_name_for_model() {
  local model="$1"

  if [[ "${model}" == vanilla_* ]]; then
    printf 'cv_%s_%s_%s' "${CV_BACKBONE_DATASET}" "${model}" "${CV_BACKBONE_OPTIMIZER}"
    return 0
  fi

  if [[ "${model}" == laplacian_* ]]; then
    printf 'cv_%s_%s_L%s_P%s_%s' \
      "${CV_BACKBONE_DATASET}" \
      "${model}" \
      "${LAPLACIAN_CV_LAMBDA}" \
      "${LAPLACIAN_CV_POOL}" \
      "${CV_BACKBONE_OPTIMIZER}"
    return 0
  fi

  return 1
}

latest_cv_checkpoint_for_run() {
  local run_name="$1"
  local run_dir="${CV_BACKBONE_ROOT}/${run_name}"
  local candidates=()
  local latest=""
  local candidate

  if [ ! -d "${run_dir}" ]; then
    return 1
  fi

  shopt -s nullglob
  candidates=("${run_dir}"/*/checkpoints/"${CV_BACKBONE_CHECKPOINT_FILENAME}")
  shopt -u nullglob

  if [ "${#candidates[@]}" -eq 0 ]; then
    return 1
  fi

  for candidate in "${candidates[@]}"; do
    if [[ -z "${latest}" || "${candidate}" > "${latest}" ]]; then
      latest="${candidate}"
    fi
  done

  printf '%s' "${latest}"
}

checkpoint_path_for_model() {
  local model="$1"
  local checkpoint=""
  local run_name=""

  checkpoint="$(manual_checkpoint_for_model "${model}")"
  if [ -n "${checkpoint}" ]; then
    printf '%s' "${checkpoint}"
    return 0
  fi

  if ! is_truthy "${AUTO_BACKBONE_CHECKPOINTS}"; then
    return 0
  fi

  if ! run_name="$(cv_run_name_for_model "${model}")"; then
    return 0
  fi

  if checkpoint="$(latest_cv_checkpoint_for_run "${run_name}")"; then
    printf '%s' "${checkpoint}"
    return 0
  fi

  return 0
}

pvt_model_requires_checkpoint() {
  local model="$1"

  [[ "${model}" == vanilla_* || "${model}" == laplacian_* ]]
}

FAILED_RUNS=()
RUN_INDEX=0

for dataset in "${DATASET_LIST[@]}"; do
  for optimizer in "${OPTIMIZER_LIST[@]}"; do
    for model in "${MODEL_LIST[@]}"; do
      RUN_INDEX=$((RUN_INDEX + 1))
      if [ "${RUN_INDEX}" -le "${SKIP_FIRST_N}" ]; then
        echo "Skipping ${RUN_INDEX}: ${dataset}_${model}_${optimizer}"
        continue
      fi

      run_name="seg_${dataset}_${model}_${optimizer}"
      manual_checkpoint="$(manual_checkpoint_for_model "${model}")"
      checkpoint_path="$(checkpoint_path_for_model "${model}")"

      CMD=(
        "${TRAIN_CMD_LIST[@]}" train.py
        task=semantic_segmentation
        datamodule="${dataset}"
        model="${model}"
        optimizer="${optimizer}"
        seed="${SEED}"
        datamodule.subset_seed="${SEED}"
        "${TRAINER_ARGS[@]}"
        "${DATAMODULE_ARGS[@]}"
        logger.project="${WANDB_PROJECT}"
        logger.name="${run_name}"
      )

      echo "================================================================"
      echo "Starting ${run_name}"
      echo "================================================================"

      if [ -n "${checkpoint_path}" ]; then
        echo "Backbone checkpoint: ${checkpoint_path}"
        CMD+=("model.backbone_checkpoint_path=${checkpoint_path}")
        if [[ "${model}" == laplacian_* ]] && [ -z "${manual_checkpoint}" ] && is_truthy "${AUTO_BACKBONE_CHECKPOINTS}"; then
          CMD+=("model.lambda_scale=${LAPLACIAN_CV_LAMBDA}")
        fi
      elif pvt_model_requires_checkpoint "${model}" && is_truthy "${REQUIRE_PRETRAINED_BACKBONE}"; then
        echo "Missing pretrained CV backbone checkpoint for model=${model} under ${CV_BACKBONE_ROOT}." >&2
        echo "Set BACKBONE_CHECKPOINT_PATH, a model-specific checkpoint path, or disable REQUIRE_PRETRAINED_BACKBONE." >&2
        exit 3
      elif pvt_model_requires_checkpoint "${model}" && is_truthy "${AUTO_BACKBONE_CHECKPOINTS}"; then
        echo "Backbone checkpoint: none found; running from scratch"
      fi

      CMD+=("${EXTRA_ARGS[@]}")

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

if [ "${#FAILED_RUNS[@]}" -gt 0 ]; then
  echo
  echo "Completed with failures:"
  printf '  - %s\n' "${FAILED_RUNS[@]}"
  exit 1
fi

echo
echo "All segmentation runs completed."
