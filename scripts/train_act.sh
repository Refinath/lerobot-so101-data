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

# HF_TOKEN is needed to download models/datasets from HuggingFace.
# When running fully local (preprocessed dataset + cached weights) it can be left unset.
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN is not set. Continuing with local cache only."
fi
export HF_TOKEN="${HF_TOKEN:-}"
export HF_LEROBOT_HOME HF_HOME HF_DATASETS_CACHE

CONDA_BIN_DIR="${CONDA_BIN_DIR:-/home/r84368868/miniconda3/bin}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/home/r84368868/envs/lerobot/}"
source "$CONDA_BIN_DIR/../etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_PATH"
# Force the env's bin/ to the front of PATH — conda activate's PATH ordering
# is unreliable across compute nodes (base conda's bin can shadow it, causing
# `python` to resolve without torch installed). This guarantees correctness.
export PATH="${CONDA_ENV_PATH%/}/bin:$PATH"
export WANDB_MODE=offline

# ── Fast CUDA check — exit code 2 signals broken GPU node to the monitor script
python - <<'PYEOF'
import torch, sys
if not torch.cuda.is_available():
    print(f"ERROR: CUDA not available on {__import__('socket').gethostname()} — driver too old or GPU absent.")
    sys.exit(2)
print(f"CUDA OK: {torch.cuda.get_device_name(0)} on {__import__('socket').gethostname()}")
PYEOF
if [[ $? -ne 0 ]]; then exit 2; fi

if ! command -v lerobot-train >/dev/null 2>&1; then
  echo "lerobot-train was not found."
  echo "Install LeRobot with feetech extras, for example:"
  echo '  pip install "lerobot[feetech]@git+https://github.com/huggingface/lerobot.git"'
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
