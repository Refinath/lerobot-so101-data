#!/usr/bin/env bash
set -euo pipefail

# Fine-tune a Pi0.5 policy with LoRA on the demonstrations collected in this
# repository. This intentionally delegates training to LeRobot's built-in
# lerobot-train CLI and PEFT support.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATASET_REPO_ID="${DATASET_REPO_ID:-Refinath/so101_bowl_placement}"
DATASET_ROOT="${DATASET_ROOT:-}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$ROOT_DIR}"

BASE_POLICY="${BASE_POLICY:-lerobot/pi05_base}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/train/pi05_lora_spatial}"
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

#if [[ -n "${LEROBOT_PATH:-}" ]]; then
#  export PYTHONPATH="$LEROBOT_PATH/src:$LEROBOT_PATH:${PYTHONPATH:-}"
#  export PATH="$LEROBOT_PATH/.venv/bin:$PATH"
#fi

if ! command -v lerobot-train >/dev/null 2>&1; then
  echo "lerobot-train was not found."
  echo "Install LeRobot with Pi/PEFT extras, for example:"
  echo '  pip install "lerobot[pi,peft]@git+https://github.com/huggingface/lerobot.git"'
  echo "Or set LEROBOT_PATH=/path/to/lerobot if you use a local checkout."
  exit 1
fi

export HF_LEROBOT_HOME

args=(
  --dataset.repo_id="$DATASET_REPO_ID"
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
echo "  dataset: $DATASET_REPO_ID"
if [[ -n "$DATASET_ROOT" ]]; then
  echo "  root:    $DATASET_ROOT"
else
  echo "  root:    LeRobot default cache"
fi
echo "  base:    $BASE_POLICY"
echo "  output:  $OUTPUT_DIR"
echo "  device:  $DEVICE"

exec lerobot-train "${args[@]}" "$@"
