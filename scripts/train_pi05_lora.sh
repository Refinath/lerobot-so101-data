#!/usr/bin/env bash
#SBATCH --job-name=pi05_lora_so101
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1


set -euo pipefail

# Fine-tune a Pi0.5 policy with LoRA on the demonstrations collected in this
# repository. This intentionally delegates training to LeRobot's built-in
# lerobot-train CLI and PEFT support.
#
# Submit on Slurm:
#   sbatch scripts/train_pi05_lora.sh
#
# Submit with overrides:
#   sbatch --export=ALL,HF_TOKEN=hf_xxx,STEPS=5000,BATCH_SIZE=4 scripts/train_pi05_lora.sh
#
# Run interactively for debugging:
#   bash scripts/train_pi05_lora.sh

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

BASE_POLICY="${BASE_POLICY:-lerobot/pi05_base}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/train/pi05_$DATASET_NAME}"
JOB_NAME="${JOB_NAME:-pi05_lora_spatial}"
POLICY_REPO_ID="${POLICY_REPO_ID:-}"

STEPS="${STEPS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
LR="${LR:-1e-3}"
SCHEDULER_DECAY_LR="${SCHEDULER_DECAY_LR:-1e-4}"
LORA_R="${LORA_R:-64}"
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
  echo "Install LeRobot with Pi/PEFT extras, for example:"
  echo '  pip install "lerobot[pi,peft]@git+https://github.com/huggingface/lerobot.git"'
  echo "Or set LEROBOT_PATH=/path/to/lerobot if you use a local checkout."
  exit 1
fi

args=(
  --dataset.repo_id="$DATASET_REPO_ID"
  --dataset.revision="$DATASET_REVISION"
  --dataset.streaming="$DATASET_STREAMING"
  --policy.type=pi05
  --policy.pretrained_path="$BASE_POLICY"
  --policy.compile_model=true
  --policy.gradient_checkpointing=true
  --policy.dtype="$DTYPE"
  --policy.device="$DEVICE"
  --policy.freeze_vision_encoder=false
  --policy.train_expert_only=false
  --policy.optimizer_lr="$LR"
  --policy.scheduler_decay_lr="$SCHEDULER_DECAY_LR"
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}'
  --output_dir="$OUTPUT_DIR"
  --job_name="$JOB_NAME"
  --steps="$STEPS"
  --batch_size="$BATCH_SIZE"
  --peft.method_type=LORA
  --peft.r="$LORA_R"
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

echo "Starting Pi0.5 LoRA fine-tuning"
echo "  host:    $(hostname)"
echo "  date:    $(date -Is)"
echo "  python:  $(command -v python || true)"
echo "  train:   $(command -v lerobot-train || true)"
echo "  slurm:   ${SLURM_JOB_ID:-not running under Slurm}"
echo "  dataset: $DATASET_REPO_ID"
echo "  revision: $DATASET_REVISION"
echo "  streaming: $DATASET_STREAMING"
if [[ -n "$DATASET_ROOT" ]]; then
  echo "  root:    $DATASET_ROOT"
else
  echo "  root:    LeRobot default cache"
fi
echo "  base:    $BASE_POLICY"
echo "  output:  $OUTPUT_DIR"
echo "  device:  $DEVICE"

exec lerobot-train "${args[@]}" "$@"
