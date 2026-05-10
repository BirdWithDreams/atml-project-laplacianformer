#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

usage() {
  cat <<'EOF'
Run staged semantic-segmentation baseline experiments.

Default RUN_SET=baseline runs:
  1. VOC vanilla small sanity baseline
  2. VOC Laplacian small comparison
  3. VOC vanilla medium capacity baseline
  4. VOC Laplacian medium comparison
  5. Cityscapes vanilla small transfer check
  6. Cityscapes Laplacian small comparison

Environment overrides:
  RUN_SET=smoke|baseline|full
  TRAIN_CMD="uv run python"
  ACCELERATOR=gpu DEVICES=1 PRECISION=32 COMPILE=false
  WANDB_PROJECT=segmentation-baselines
  ACCUMULATE_GRAD_BATCHES=4
  MAX_STEPS=80000
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
    add_experiment "voc_vanilla_small_smoke" "voc2012_segmentation" "vanilla_pvt_small"
    add_experiment "voc_laplacian_small_smoke" "voc2012_segmentation" "laplacian_pvt_small_cuda"
    ;;
  baseline)
    add_experiment "voc_vanilla_small_baseline" "voc2012_segmentation" "vanilla_pvt_small"
    add_experiment "voc_laplacian_small_baseline" "voc2012_segmentation" "laplacian_pvt_small_cuda"
    add_experiment "voc_vanilla_medium_capacity" "voc2012_segmentation" "vanilla_pvt_medium"
    add_experiment "voc_laplacian_medium_capacity" "voc2012_segmentation" "laplacian_pvt_medium_cuda"
    add_experiment "cityscapes_vanilla_small_baseline" "cityscapes_segmentation" "vanilla_pvt_small"
    add_experiment "cityscapes_laplacian_small_baseline" "cityscapes_segmentation" "laplacian_pvt_small_cuda"
    ;;
  full)
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

run_experiment() {
  local index="$1"
  local total="$2"
  local name="$3"
  local dataset="$4"
  local model="$5"
  local checkpoint_override
  checkpoint_override="$(checkpoint_override_for_model "${model}")"

  local command=(
    "${TRAIN_CMD_LIST[@]}" train.py
    "${COMMON_OVERRIDES[@]}"
    datamodule="${dataset}"
    model="${model}"
    logger.name="seg_${name}"
  )

  if [ -n "${checkpoint_override}" ]; then
    command+=("${checkpoint_override}")
  fi

  command+=("${HYDRA_ARGS[@]}")

  echo "[$((index + 1))/${total}] ${name}: datamodule=${dataset} model=${model}"
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
