#!/usr/bin/env bash
#SBATCH --job-name=smolvla_so101
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=agent-long

set -euo pipefail

# Fine-tune SmolVLA (SmolVLM2-500M backbone, ~6x smaller than Pi0.5) with LoRA
# on the SO101 EE-space bowl-placement demonstrations.
#
# Starting from lerobot/smolvla_base (pretrained SmolVLA checkpoint) and applying
# LoRA to the action-expert q/v projections keeps memory and compute low while
# giving the model a strong initialisation.
#
# Submit on Slurm:
#   sbatch scripts/train_smolvla.sh
#
# Submit with overrides:
#   sbatch --export=ALL,STEPS=30000,BATCH_SIZE=32 scripts/train_smolvla.sh

ROOT_DIR="${ROOT_DIR:-/home/r84368868/lerobot-so101-data}"
cd "$ROOT_DIR"

DATASET_NAME=so101_bowl_placement
DATASET_REPO_ID="${DATASET_REPO_ID:-Refinath/$DATASET_NAME}"
RAW_DATASET_ROOT="${RAW_DATASET_ROOT:-$ROOT_DIR/dataset/pick-the-black-bowl-from-the-top-of-the-drawer-and-place-it-on-the-table}"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-${RAW_DATASET_ROOT}-preprocessed}"
DATASET_ROOT="${DATASET_ROOT:-}"
DATASET_REVISION="${DATASET_REVISION:-main}"
DATASET_STREAMING="${DATASET_STREAMING:-false}"
SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$ROOT_DIR}"
HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

BASE_POLICY="${BASE_POLICY:-lerobot/smolvla_base}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/train/smolvla_$DATASET_NAME}"
JOB_NAME="${JOB_NAME:-smolvla_so101}"
POLICY_REPO_ID="${POLICY_REPO_ID:-}"

STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
LR="${LR:-1e-4}"
SCHEDULER_DECAY_LR="${SCHEDULER_DECAY_LR:-2.5e-6}"
LORA_R="${LORA_R:-32}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN is not set. Continuing with local cache only."
fi
export HF_TOKEN="${HF_TOKEN:-}"
export HF_LEROBOT_HOME HF_HOME HF_DATASETS_CACHE

CONDA_BIN_DIR="${CONDA_BIN_DIR:-/home/r84368868/miniconda3/bin}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/home/r84368868/envs/lerobot/}"
source "$CONDA_BIN_DIR/../etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_PATH"
export PATH="${CONDA_ENV_PATH%/}/bin:$PATH"
export WANDB_MODE=offline

python - <<'PYEOF'
import torch, sys
if not torch.cuda.is_available():
    print(f"ERROR: CUDA not available on {__import__('socket').gethostname()} — driver too old or GPU absent.")
    sys.exit(2)
print(f"CUDA OK: {torch.cuda.get_device_name(0)} on {__import__('socket').gethostname()}")
PYEOF
if [[ $? -ne 0 ]]; then exit 2; fi

if ! command -v lerobot-train >/dev/null 2>&1; then
  echo "lerobot-train was not found. Install lerobot with smolvla extras:"
  echo '  pip install -e ".[smolvla]"'
  exit 1
fi

# ── Preprocessing: unwrap ee.wy + drop depth camera ──────────────────────────
if [[ -n "$RAW_DATASET_ROOT" && -d "$RAW_DATASET_ROOT" ]]; then
  if [[ "$SKIP_PREPROCESS" == "1" && -d "$PREPROCESSED_ROOT" ]]; then
    echo "Skipping preprocessing (SKIP_PREPROCESS=1), using: $PREPROCESSED_ROOT"
  else
    echo "Preprocessing dataset (unwrap ee.wy, drop depth camera)..."
    python "$ROOT_DIR/scripts/preprocess_dataset.py" \
      --input-dataset  "$RAW_DATASET_ROOT" \
      --output-dataset "$PREPROCESSED_ROOT" \
      --force
  fi
  DATASET_ROOT="$PREPROCESSED_ROOT"
fi

args=(
  --policy.path="$BASE_POLICY"
  --policy.device="$DEVICE"
  --policy.dtype="$DTYPE"
  --policy.load_vlm_weights=true
  --policy.freeze_vision_encoder=true
  --policy.train_expert_only=true
  --policy.optimizer_lr="$LR"
  --policy.scheduler_decay_lr="$SCHEDULER_DECAY_LR"
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}'
  --dataset.repo_id="$DATASET_REPO_ID"
  --dataset.revision="$DATASET_REVISION"
  --dataset.streaming="$DATASET_STREAMING"
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

echo "Starting SmolVLA LoRA fine-tuning"
echo "  host:     $(hostname)"
echo "  date:     $(date -Is)"
echo "  python:   $(command -v python || true)"
echo "  train:    $(command -v lerobot-train || true)"
echo "  slurm:    ${SLURM_JOB_ID:-not running under Slurm}"
echo "  dataset:  $DATASET_REPO_ID"
echo "  root:     ${DATASET_ROOT:-LeRobot default cache}"
echo "  base:     $BASE_POLICY"
echo "  output:   $OUTPUT_DIR"
echo "  steps:    $STEPS  batch: $BATCH_SIZE  lr: $LR  lora_r: $LORA_R"
echo "  device:   $DEVICE  dtype: $DTYPE"

exec lerobot-train "${args[@]}" "$@"
