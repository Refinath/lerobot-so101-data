#!/usr/bin/env bash
set -euo pipefail

# Deploy a fine-tuned Pi0.5 LoRA policy on the real SO101 follower arm using
# LeRobot's built-in rollout command.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

POLICY_PATH="${POLICY_PATH:-$ROOT_DIR/outputs/train/pi05_lora_spatial/checkpoints/last/pretrained_model}"
TASK="${TASK:-My task description}"
DURATION="${DURATION:-60}"
FPS="${FPS:-30}"

ROBOT_TYPE="${ROBOT_TYPE:-so101_follower}"
FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/ttyACM1}"
FOLLOWER_ID="${FOLLOWER_ID:-follower_arm}"

WRIST_CAMERA_PATH="${WRIST_CAMERA_PATH:-/dev/video0}"
AGENT_CAMERA_PATH="${AGENT_CAMERA_PATH:-/dev/video2}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"

ROBOT_CAMERAS="${ROBOT_CAMERAS:-{ wrist: {type: opencv, index_or_path: $WRIST_CAMERA_PATH, width: $CAMERA_WIDTH, height: $CAMERA_HEIGHT, fps: $FPS}, agent_view: {type: opencv, index_or_path: $AGENT_CAMERA_PATH, width: $CAMERA_WIDTH, height: $CAMERA_HEIGHT, fps: $FPS}}}"
YES="${YES:-false}"

if [[ -n "${LEROBOT_PATH:-}" ]]; then
  export PYTHONPATH="$LEROBOT_PATH/src:$LEROBOT_PATH:${PYTHONPATH:-}"
  export PATH="$LEROBOT_PATH/.venv/bin:$PATH"
fi

if ! command -v lerobot-rollout >/dev/null 2>&1; then
  echo "lerobot-rollout was not found."
  echo "Install LeRobot with Pi support, for example:"
  echo '  pip install "lerobot[pi]@git+https://github.com/huggingface/lerobot.git"'
  echo "Or set LEROBOT_PATH=/path/to/lerobot if you use a local checkout."
  exit 1
fi

if [[ "$POLICY_PATH" != */* && ! -d "$POLICY_PATH" ]]; then
  echo "POLICY_PATH should be a local checkpoint directory or a Hugging Face model id."
  echo "Current value: $POLICY_PATH"
  exit 1
fi

echo "About to run a real-robot Pi0.5 rollout."
echo "  policy:   $POLICY_PATH"
echo "  robot:    $ROBOT_TYPE on $FOLLOWER_PORT ($FOLLOWER_ID)"
echo "  cameras:  $ROBOT_CAMERAS"
echo "  task:     $TASK"
echo "  duration: ${DURATION}s"

if [[ "$YES" != "true" ]]; then
  read -r -p "Place the robot in a safe start pose, clear the workspace, then type RUN: " confirm
  if [[ "$confirm" != "RUN" ]]; then
    echo "Aborted."
    exit 1
  fi
fi

exec lerobot-rollout \
  --strategy.type=base \
  --policy.path="$POLICY_PATH" \
  --robot.type="$ROBOT_TYPE" \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id="$FOLLOWER_ID" \
  --robot.cameras="$ROBOT_CAMERAS" \
  --task="$TASK" \
  --duration="$DURATION" \
  "$@"
