#!/usr/bin/env bash
#SBATCH --job-name=eval_act_openloop
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --partition=agent-long

set -euo pipefail

# One-shot open-loop evaluation: replay recorded episodes through the trained
# ACT checkpoint and compare its predicted EE-space actions against the
# human-teleoperated ground-truth trajectory. No real robot / sim required.
#
# Submit:
#   sbatch scripts/eval_act_openloop.sh
# Override episodes/checkpoint:
#   sbatch --export=ALL,EPISODES="0 36 71",CHECKPOINT=outputs/train/.../checkpoints/last/pretrained_model scripts/eval_act_openloop.sh

ROOT_DIR="${ROOT_DIR:-/home/r84368868/lerobot-so101-data}"
cd "$ROOT_DIR"

DATASET_NAME=so101_bowl_placement
DATASET_REPO_ID="${DATASET_REPO_ID:-Refinath/$DATASET_NAME}"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-$ROOT_DIR/dataset/pick-the-black-bowl-from-the-top-of-the-drawer-and-place-it-on-the-table-preprocessed}"
CHECKPOINT="${CHECKPOINT:-$ROOT_DIR/outputs/train/act_so101_bowl_placement/checkpoints/100000/pretrained_model}"
EPISODES="${EPISODES:-0 36 71}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/eval/act_openloop}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$ROOT_DIR}"
HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_LEROBOT_HOME HF_HOME HF_DATASETS_CACHE
export HF_TOKEN="${HF_TOKEN:-}"

CONDA_BIN_DIR="${CONDA_BIN_DIR:-/home/r84368868/miniconda3/bin}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/home/r84368868/envs/lerobot/}"
source "$CONDA_BIN_DIR/../etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_PATH"
export PATH="${CONDA_ENV_PATH%/}/bin:$PATH"

echo "host:       $(hostname)"
echo "checkpoint: $CHECKPOINT"
echo "dataset:    $PREPROCESSED_ROOT"
echo "episodes:   $EPISODES"
echo "output:     $OUTPUT_DIR"

# shellcheck disable=SC2086
python "$ROOT_DIR/scripts/eval_policy_openloop.py" \
  --policy-path "$CHECKPOINT" \
  --repo-id "$DATASET_REPO_ID" \
  --root "$PREPROCESSED_ROOT" \
  --episodes $EPISODES \
  --output-dir "$OUTPUT_DIR"
