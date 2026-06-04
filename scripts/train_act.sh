#!/usr/bin/env bash
#SBATCH --job-name=act_so101
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

set -euo pipefail

# Train an ACT (Action Chunking Transformer) policy on SO101 EE-space demonstrations.
# ACT trains from scratch — no pretrained base model or LoRA needed.
#
# Submit on Slurm:
#   sbatch scripts/train_act.sh
#
# Submit with overrides:
#   sbatch --export=ALL,STEPS=200000,BATCH_SIZE=64 scripts/train_act.sh
#
# Run interactively for debugging:
#   bash scripts/train_act.sh

ROOT_DIR="/home/r84368868/lerobot-so101-data"
cd "$ROOT_DIR"

DATASET_NAME=so101_bowl_placement
DATASET_REPO_ID="${DATASET_REPO_ID:-Refinath/$DATASET_NAME}"
DATASET_ROOT="${DATASET_ROOT:-}"
DATASET_REVISION="${DATASET_REVISION:-main}"
DATASET_STREAMING="${DATASET_STREAMING:-false}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$ROOT_DIR}"
HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/train/act_$DATASET_NAME}"
JOB_NAME="${JOB_NAME:-act_so101}"
POLICY_REPO_ID="${POLICY_REPO_ID:-}"

# ACT hyperparameters
STEPS="${STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float32}"
CHUNK_SIZE="${CHUNK_SIZE:-100}"
LR="${LR:-1e-5}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"

# Set HF_TOKEN in your environment before running, e.g.:
#   export HF_TOKEN=hf_...
: "${HF_TOKEN:?HF_TOKEN must be set in the environment}"
export HF_LEROBOT_HOME HF_HOME HF_DATASETS_CACHE

export PATH="/home/r84368868/miniconda3/bin:$PATH"
source /home/r84368868/miniconda3/bin/activate /home/r84368868/envs/lerobot/
export WANDB_MODE=offline

if ! command -v lerobot-train >/dev/null 2>&1; then
  echo "lerobot-train was not found."
  echo "Install LeRobot with feetech extras, for example:"
  echo '  pip install "lerobot[feetech]@git+https://github.com/huggingface/lerobot.git"'
  exit 1
fi

args=(
  --dataset.repo_id="$DATASET_REPO_ID"
  --dataset.revision="$DATASET_REVISION"
  --dataset.streaming="$DATASET_STREAMING"
  --policy.type=act
  --policy.chunk_size="$CHUNK_SIZE"
  --policy.n_action_steps="$CHUNK_SIZE"
  --policy.vision_backbone=resnet18
  --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"MEAN_STD"}'
  --policy.device="$DEVICE"
  --policy.optimizer_lr="$LR"
  --policy.optimizer_lr_backbone="$LR"
  --output_dir="$OUTPUT_DIR"
  --job_name="$JOB_NAME"
  --steps="$STEPS"
  --batch_size="$BATCH_SIZE"
  --wandb.enable="$WANDB_ENABLE"
  --policy.push_to_hub="$PUSH_TO_HUB"
)

if [[ -n "$DATASET_ROOT" ]]; then
  if [[ ! -d "$DATASET_ROOT" ]]; then
    echo "Dataset root does not exist: $DATASET_ROOT"
    exit 1
  fi
  args+=(--dataset.root="$DATASET_ROOT")
fi

if [[ -n "$POLICY_REPO_ID" ]]; then
  args+=(--policy.repo_id="$POLICY_REPO_ID")
fi

echo "Starting ACT training"
echo "  host:    $(hostname)"
echo "  date:    $(date -Is)"
echo "  python:  $(command -v python || true)"
echo "  train:   $(command -v lerobot-train || true)"
echo "  slurm:   ${SLURM_JOB_ID:-not running under Slurm}"
echo "  dataset: $DATASET_REPO_ID"
if [[ -n "$DATASET_ROOT" ]]; then
  echo "  root:    $DATASET_ROOT"
else
  echo "  root:    LeRobot default cache"
fi
echo "  output:  $OUTPUT_DIR"
echo "  device:  $DEVICE"
echo "  steps:   $STEPS"
echo "  batch:   $BATCH_SIZE"
echo "  chunk:   $CHUNK_SIZE"
echo "  lr:      $LR"

exec lerobot-train "${args[@]}" "$@"
