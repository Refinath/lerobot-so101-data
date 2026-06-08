#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper: deploy the Pi0.5 LoRA checkpoint via deploy_ee_real_robot.py.
# For ACT use the same script with --policy-path pointing to the ACT checkpoint.
#
# NOTE: Do NOT use lerobot-rollout directly for EE-space policies. The CLI
# uses identity robot processors by default, so EE predictions would be sent
# as raw joint commands — the correct FK/IK wrappers live in deploy_ee_real_robot.py.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PATH="/home/r84368868/miniconda3/bin:$PATH"
source /home/r84368868/miniconda3/bin/activate /home/r84368868/envs/lerobot/

POLICY_PATH="${POLICY_PATH:-$ROOT_DIR/outputs/train/pi05_so101_bowl_placement/checkpoints/last/pretrained_model}"
TASK="${TASK:-My task description}"
DURATION="${DURATION:-60}"
FPS="${FPS:-30}"

FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/ttyACM0}"
FOLLOWER_ID="${FOLLOWER_ID:-follower_arm}"

WRIST_CAMERA_PATH="${WRIST_CAMERA_PATH:-/dev/video2}"
AGENT_CAMERA_PATH="${AGENT_CAMERA_PATH:-/dev/video8}"

echo "Deploying Pi0.5 policy via deploy_ee_real_robot.py"
echo "  python:     $(command -v python || true)"
echo "  policy:     $POLICY_PATH"
echo "  task:       $TASK"
echo "  duration:   ${DURATION}s"

exec python "$ROOT_DIR/scripts/deploy_ee_real_robot.py" \
  --policy-path="$POLICY_PATH" \
  --task="$TASK" \
  --duration="$DURATION" \
  --fps="$FPS" \
  --follower-port="$FOLLOWER_PORT" \
  --follower-id="$FOLLOWER_ID" \
  --wrist-cam="$WRIST_CAMERA_PATH" \
  --agent-cam="$AGENT_CAMERA_PATH" \
  "$@"
