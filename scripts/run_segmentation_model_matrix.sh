#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

usage() {
  cat <<'EOF'
Run staged semantic-segmentation baseline experiments.

Default RUN_SET=baseline runs:
  1. VOC torchvision DeepLabV3 pretrained sanity baseline
  2. VOC vanilla small baseline, auto-pretrained when a matching CV checkpoint exists
  3. VOC Laplacian small comparison, auto-pretrained when a matching CV checkpoint exists
  4. VOC vanilla medium capacity baseline
  5. VOC Laplacian medium capacity comparison
  6. Cityscapes vanilla small transfer check
  7. Cityscapes Laplacian small comparison

Environment overrides:
  RUN_SET=smoke|baseline|full
  TRAIN_CMD="uv run python"
  ACCELERATOR=gpu DEVICES=1 PRECISION=32 COMPILE=false
  WANDB_PROJECT=segmentation-baselines
  ACCUMULATE_GRAD_BATCHES=4
  MAX_STEPS=80000
  AUTO_BACKBONE_CHECKPOINTS=true
  CV_BACKBONE_ROOT=results/cv-model-matrix
  CV_BACKBONE_DATASET=imagenet
  CV_BACKBONE_OPTIMIZER=adamw_cv_default
  CV_BACKBONE_CHECKPOINT_FILENAME=last.ckpt
  LAPLACIAN_CV_LAMBDA=4
  LAPLACIAN_CV_POOL=2
  REQUIRE_PRETRAINED_BACKBONE=false
  BACKBONE_CHECKPOINT_PATH=/path/to/matching_pvt.ckpt
  VANILLA_BACKBONE_CHECKPOINT_PATH=/path/to/vanilla_pvt.ckpt
  LAPLACIAN_BACKBONE_CHECKPOINT_PATH=/path/to/laplacian_pvt.ckpt
  EXTRA_OVERRIDES='logger.entity=null trainer.log_every_n_steps=20'

Script flags:
  --skip N      Skip the first N planned experiments
  --limit N     Run at most N experiments after skipping
  --dry-run     Print commands without executing
  --help        Show this message

Any remaining arguments are appended to every train.py invocation as Hydra overrides.
EOF
}

SKIP=0
LIMIT=0
DRY_RUN=false
HYDRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip)
      SKIP="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      HYDRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [ -n "${TRAIN_CMD:-}" ]; then
  read -r -a TRAIN_CMD_LIST <<< "${TRAIN_CMD}"
elif command -v uv >/dev/null 2>&1; then
  TRAIN_CMD_LIST=(uv run python)
else
  TRAIN_CMD_LIST=(python)
fi

RUN_SET="${RUN_SET:-baseline}"
ACCELERATOR="${ACCELERATOR:-gpu}"
DEVICES="${DEVICES:-1}"
PRECISION="${PRECISION:-32}"
COMPILE="${COMPILE:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-segmentation-baselines}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-4}"
MAX_STEPS="${MAX_STEPS:-}"
AUTO_BACKBONE_CHECKPOINTS="${AUTO_BACKBONE_CHECKPOINTS:-true}"
CV_BACKBONE_ROOT="${CV_BACKBONE_ROOT:-results/cv-model-matrix}"
CV_BACKBONE_DATASET="${CV_BACKBONE_DATASET:-imagenet}"
CV_BACKBONE_OPTIMIZER="${CV_BACKBONE_OPTIMIZER:-adamw_cv_default}"
CV_BACKBONE_CHECKPOINT_FILENAME="${CV_BACKBONE_CHECKPOINT_FILENAME:-last.ckpt}"
LAPLACIAN_CV_LAMBDA="${LAPLACIAN_CV_LAMBDA:-4}"
LAPLACIAN_CV_POOL="${LAPLACIAN_CV_POOL:-2}"
REQUIRE_PRETRAINED_BACKBONE="${REQUIRE_PRETRAINED_BACKBONE:-false}"

COMMON_OVERRIDES=(
  task=semantic_segmentation
  optimizer=adamw_segmentation_poly
  trainer.accelerator="${ACCELERATOR}"
  trainer.devices="${DEVICES}"
  trainer.precision="${PRECISION}"
  trainer.compile="${COMPILE}"
  trainer.accumulate_grad_batches="${ACCUMULATE_GRAD_BATCHES}"
  logger.project="${WANDB_PROJECT}"
)

if [ -n "${MAX_STEPS}" ]; then
  COMMON_OVERRIDES+=(trainer.max_steps="${MAX_STEPS}" optimizer.max_iters="${MAX_STEPS}")
fi

if [ -n "${EXTRA_OVERRIDES:-}" ]; then
  read -r -a EXTRA_OVERRIDE_LIST <<< "${EXTRA_OVERRIDES}"
  COMMON_OVERRIDES+=("${EXTRA_OVERRIDE_LIST[@]}")
fi

EXPERIMENTS=()
add_experiment() {
  local name="$1"
  local dataset="$2"
  local model="$3"
  EXPERIMENTS+=("${name}|${dataset}|${model}")
}

case "${RUN_SET}" in
  smoke)
    add_experiment "voc_torchvision_deeplabv3_pretrained_smoke" "voc2012_segmentation" "torchvision_deeplabv3_resnet50"
    add_experiment "voc_vanilla_small_smoke" "voc2012_segmentation" "vanilla_pvt_small"
    add_experiment "voc_laplacian_small_smoke" "voc2012_segmentation" "laplacian_pvt_small_cuda"
    ;;
  baseline)
    add_experiment "voc_torchvision_deeplabv3_pretrained_sanity" "voc2012_segmentation" "torchvision_deeplabv3_resnet50"
    add_experiment "voc_vanilla_small_baseline" "voc2012_segmentation" "vanilla_pvt_small"
    add_experiment "voc_laplacian_small_baseline" "voc2012_segmentation" "laplacian_pvt_small_cuda"
    add_experiment "voc_vanilla_medium_capacity" "voc2012_segmentation" "vanilla_pvt_medium"
    add_experiment "voc_laplacian_medium_capacity" "voc2012_segmentation" "laplacian_pvt_medium_cuda"
    add_experiment "cityscapes_vanilla_small_baseline" "cityscapes_segmentation" "vanilla_pvt_small"
    add_experiment "cityscapes_laplacian_small_baseline" "cityscapes_segmentation" "laplacian_pvt_small_cuda"
    ;;
  full)
    add_experiment "voc_torchvision_deeplabv3_pretrained_sanity" "voc2012_segmentation" "torchvision_deeplabv3_resnet50"
    add_experiment "voc_torchvision_fcn_pretrained_sanity" "voc2012_segmentation" "torchvision_fcn_resnet50"
    add_experiment "voc_vanilla_small_baseline" "voc2012_segmentation" "vanilla_pvt_small"
    add_experiment "voc_laplacian_small_baseline" "voc2012_segmentation" "laplacian_pvt_small_cuda"
    add_experiment "voc_vanilla_medium_capacity" "voc2012_segmentation" "vanilla_pvt_medium"
    add_experiment "voc_laplacian_medium_capacity" "voc2012_segmentation" "laplacian_pvt_medium_cuda"
    add_experiment "cityscapes_vanilla_small_baseline" "cityscapes_segmentation" "vanilla_pvt_small"
    add_experiment "cityscapes_laplacian_small_baseline" "cityscapes_segmentation" "laplacian_pvt_small_cuda"
    add_experiment "cityscapes_vanilla_medium_capacity" "cityscapes_segmentation" "vanilla_pvt_medium"
    add_experiment "cityscapes_laplacian_medium_capacity" "cityscapes_segmentation" "laplacian_pvt_medium_cuda"
    ;;
  *)
    echo "Unknown RUN_SET='${RUN_SET}'. Expected smoke, baseline, or full." >&2
    exit 2
    ;;
esac

checkpoint_override_for_model() {
  local model="$1"
  local checkpoint=""

  if [[ "${model}" == vanilla_* ]]; then
    checkpoint="${VANILLA_BACKBONE_CHECKPOINT_PATH:-${BACKBONE_CHECKPOINT_PATH:-}}"
  elif [[ "${model}" == laplacian_* ]]; then
    checkpoint="${LAPLACIAN_BACKBONE_CHECKPOINT_PATH:-${BACKBONE_CHECKPOINT_PATH:-}}"
  fi

  if [ -n "${checkpoint}" ]; then
    printf 'model.backbone_checkpoint_path=%s' "${checkpoint}"
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
  local checkpoint_override=""
  local run_name=""

  checkpoint_override="$(checkpoint_override_for_model "${model}")"
  if [ -n "${checkpoint_override}" ]; then
    printf '%s' "${checkpoint_override#model.backbone_checkpoint_path=}"
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

run_experiment() {
  local index="$1"
  local total="$2"
  local name="$3"
  local dataset="$4"
  local model="$5"
  local explicit_checkpoint_override
  local checkpoint_path
  explicit_checkpoint_override="$(checkpoint_override_for_model "${model}")"
  checkpoint_path="$(checkpoint_path_for_model "${model}")"

  local command=(
    "${TRAIN_CMD_LIST[@]}" train.py
    "${COMMON_OVERRIDES[@]}"
    datamodule="${dataset}"
    model="${model}"
    logger.name="seg_${name}"
  )

  if [ -n "${checkpoint_path}" ]; then
    command+=("model.backbone_checkpoint_path=${checkpoint_path}")
    if [[ "${model}" == laplacian_* ]] && [ -z "${explicit_checkpoint_override}" ] && is_truthy "${AUTO_BACKBONE_CHECKPOINTS}"; then
      command+=("model.lambda_scale=${LAPLACIAN_CV_LAMBDA}")
    fi
  elif pvt_model_requires_checkpoint "${model}" && is_truthy "${REQUIRE_PRETRAINED_BACKBONE}"; then
    echo "Missing pretrained CV backbone checkpoint for model=${model} under ${CV_BACKBONE_ROOT}." >&2
    echo "Set BACKBONE_CHECKPOINT_PATH, a model-specific checkpoint path, or disable REQUIRE_PRETRAINED_BACKBONE." >&2
    exit 3
  fi

  command+=("${HYDRA_ARGS[@]}")

  echo "[$((index + 1))/${total}] ${name}: datamodule=${dataset} model=${model}"
  if [ -n "${checkpoint_path}" ]; then
    echo "  backbone checkpoint: ${checkpoint_path}"
  elif pvt_model_requires_checkpoint "${model}" && is_truthy "${AUTO_BACKBONE_CHECKPOINTS}"; then
    echo "  backbone checkpoint: none found; running from scratch"
  fi
  if [ "${DRY_RUN}" = true ]; then
    printf '  '
    printf '%q ' "${command[@]}"
    printf '\n'
  else
    "${command[@]}"
  fi
}

total=${#EXPERIMENTS[@]}
ran=0
for index in "${!EXPERIMENTS[@]}"; do
  if (( index < SKIP )); then
    continue
  fi
  if (( LIMIT > 0 && ran >= LIMIT )); then
    break
  fi

  IFS='|' read -r name dataset model <<< "${EXPERIMENTS[$index]}"
  run_experiment "${index}" "${total}" "${name}" "${dataset}" "${model}"
  ran=$((ran + 1))
done

echo "Completed ${ran} experiment(s) from RUN_SET=${RUN_SET}."
