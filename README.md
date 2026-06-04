# LeRobot SO101 — Data Collection, Training & Deployment

End-to-end workflow for the [SO101](https://github.com/TheRobotStudio/SO-ARM100) 6-joint robot arm using [LeRobot](https://github.com/huggingface/lerobot).
Covers teleoperated data collection, training two complementary policies (Pi0.5 and ACT), and real-robot deployment.

---

## Repository Layout

```
record_data.py                    Teleoperated data collection (leader → follower).
scripts/
  train_pi05_lora.sh              Fine-tune Pi0.5 with LoRA (VLM-based, language-conditioned).
  train_act.sh                    Train ACT from scratch (ResNet backbone, fast inference).
  deploy_pi05_real_robot.sh       Deploy Pi0.5 checkpoint on the real robot.
  deploy_ee_real_robot.py         Shared Python deploy for any EE-space policy (ACT or Pi0.5).
  validate_dataset.py             Validate a collected LeRobot dataset.
  replay_dataset.py               Replay an episode in MuJoCo or as a joint-trajectory plot.
  check_task_instruction.py       Print task instruction metadata from a dataset.
SO101/                            URDF, MuJoCo scene, and SO101 robot assets.
necessary_commands.txt            Quick-reference hardware commands.
```

---

## Robot & Action Space

The SO101 has **6 motors**:

| ID | Joint | Role |
|----|-------|------|
| 1 | `shoulder_pan` | Base rotation |
| 2 | `shoulder_lift` | Shoulder pitch |
| 3 | `elbow_flex` | Elbow pitch |
| 4 | `wrist_flex` | Wrist pitch |
| 5 | `wrist_roll` | Wrist roll (continuous) |
| 6 | `gripper` | Open / close (0–100 %) |

**Why 7-D actions?** Data is stored in **end-effector (EE) space** rather than joint space:

| Dimension | Meaning |
|-----------|---------|
| `ee.x`, `ee.y`, `ee.z` | EE position (metres) |
| `ee.wx`, `ee.wy`, `ee.wz` | EE orientation (axis-angle) |
| `ee.gripper_pos` | Gripper (0–100 %) |

The recording pipeline converts 6 raw joint angles → 7-D EE via **forward kinematics** at collection time.
At deployment a matching **inverse kinematics** step converts the policy's 7-D EE prediction back to 6 joint commands.
The 6 DOF (degrees of freedom) and 7-D EE representation are consistent: 5-joint arm chain → 6-D wrist pose (3 position + 3 orientation) plus 1-D gripper.

---

## Setup

```bash
# Activate the LeRobot conda environment
export PATH="/home/r84368868/miniconda3/bin:$PATH"
source /home/r84368868/miniconda3/bin/activate /home/r84368868/envs/lerobot/

# Set your HuggingFace token
export HF_TOKEN=hf_...
```

Before recording, confirm hardware constants at the top of `record_data.py`:

| Constant | Default |
|----------|---------|
| `FOLLOWER_PORT` | `/dev/ttyACM0` |
| `LEADER_PORT` | `/dev/ttyACM1` |
| `WRIST_CAMERA_PATH` | `/dev/video2` |
| `AGENT_CAMERA_PATH` | `/dev/video8` |
| `AGENT_DEPTH_CAMERA_PATH` | `/dev/video6` |
| `NUM_EPISODES` | `50` |
| `EPISODE_TIME_SEC` | `60` |
| `RESET_TIME_SEC` | `30` |

For port/camera discovery commands see `necessary_commands.txt`.

---

## 1 — Calibrate & Teleoperate

```bash
# Find ports
lerobot-find-port

# Calibrate follower then leader
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=follower_arm
lerobot-calibrate --teleop.type=so101_leader  --teleop.port=/dev/ttyACM1 --teleop.id=leader_arm

# Test teleoperation
lerobot-teleoperate \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=follower_arm \
  --teleop.type=so101_leader  --teleop.port=/dev/ttyACM1 --teleop.id=leader_arm
```

---

## 2 — Collect Data

```bash
# Create a new dataset for a task
python record_data.py --prompt "place the black bowl on top of the stove"

# Resume an existing dataset
python record_data.py --prompt "place the black bowl on top of the stove" --resume

# Start fresh
python record_data.py --prompt "place the black bowl on top of the stove" --overwrite

# Save PNG frames instead of MP4
python record_data.py --prompt "place the black bowl on top of the stove" --save-images
```

### Keyboard controls

| Key | Action |
|-----|--------|
| `Space` | Start the next episode |
| `Enter` | End the current episode / exit reset loop early |
| `Backspace` | Discard the current episode and re-record |
| `Esc` | Stop the whole session |

### Dataset layout

```
dataset/<prompt-slug>/
  data/chunk-000/file-000.parquet    # joint states, EE actions, timestamps
  meta/info.json
  meta/stats.json
  meta/tasks.parquet
  videos/observation.images.wrist/...
  videos/observation.images.agent_view/...
  videos/observation.images.agent_view_depth/...
```

---

## 3 — Validate & Replay

```bash
# Validate dataset integrity
python scripts/validate_dataset.py \
  --repo-id dataset/place-the-black-bowl-on-top-of-the-stove \
  --root    dataset/place-the-black-bowl-on-top-of-the-stove

# Replay episode 0 in MuJoCo
python scripts/replay_dataset.py \
  --repo-id dataset/place-the-black-bowl-on-top-of-the-stove \
  --root    dataset/place-the-black-bowl-on-top-of-the-stove \
  --episode 0

# Replay as a joint-trajectory plot (no MuJoCo needed)
python scripts/replay_dataset.py \
  --repo-id dataset/place-the-black-bowl-on-top-of-the-stove \
  --root    dataset/place-the-black-bowl-on-top-of-the-stove \
  --episode 0 --backend plot
```

---

## 4 — Train

Both policies train on the same 7-D EE-space dataset. Choose based on your needs:

| | **Pi0.5 (LoRA)** | **ACT** |
|---|---|---|
| Base model | PaliGemma 3B VLM | ResNet-18 + Transformer (scratch) |
| Language conditioning | Yes — uses task description as text input | No — task is implicit in the data |
| GPU RAM (min) | ~24 GB | ~8 GB |
| Training steps | 3 000–10 000 | 100 000+ |
| Inference speed | Slower (large VLM) | Fast (~10 ms/step) |
| Best for | Language-generalisation, few-shot tasks | Speed-critical, well-defined single tasks |

### Train Pi0.5 with LoRA

```bash
export HF_TOKEN=hf_...
sbatch scripts/train_pi05_lora.sh

# Or interactively
bash scripts/train_pi05_lora.sh

# Override defaults
STEPS=5000 BATCH_SIZE=4 bash scripts/train_pi05_lora.sh
```

Output: `outputs/train/pi05_so101_bowl_placement/`

### Train ACT

```bash
export HF_TOKEN=hf_...
sbatch scripts/train_act.sh

# Or interactively
bash scripts/train_act.sh

# Override defaults
STEPS=200000 BATCH_SIZE=64 CHUNK_SIZE=100 bash scripts/train_act.sh
```

Output: `outputs/train/act_so101_bowl_placement/`

### Key training parameters

| Parameter | Pi0.5 | ACT |
|-----------|-------|-----|
| `STEPS` | 3 000 (default) | 100 000 (default) |
| `BATCH_SIZE` | 8 | 32 |
| `LR` | 1e-3 | 1e-5 |
| `LORA_R` (Pi0.5 only) | 64 | — |
| `CHUNK_SIZE` (ACT only) | — | 100 |

---

## 5 — Deploy on the Real Robot

> **Important — EE-space deploy requires FK/IK wrappers.**
> Both policies predict 7-D EE actions. The robot motors speak joint space.
> `lerobot-rollout` uses identity processors by default and will **not** do the
> EE → joint conversion — always deploy via `deploy_ee_real_robot.py` or the
> shell wrappers below.

### Deploy Pi0.5

```bash
# Default — uses the Pi0.5 checkpoint path
bash scripts/deploy_pi05_real_robot.sh

# Override policy path or task
POLICY_PATH=outputs/train/pi05_so101_bowl_placement/checkpoints/last/pretrained_model \
TASK="place the black bowl on the stove" \
bash scripts/deploy_pi05_real_robot.sh
```

### Deploy ACT

```bash
python scripts/deploy_ee_real_robot.py \
  --policy-path outputs/train/act_so101_bowl_placement/checkpoints/last/pretrained_model \
  --task "place the black bowl on the stove" \
  --duration 60
```

### Common deploy options (`deploy_ee_real_robot.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--policy-path` | (required) | Checkpoint dir or HF model id |
| `--task` | (required) | Natural-language task string |
| `--duration` | `60` | Episode length in seconds |
| `--fps` | `30` | Control frequency |
| `--follower-port` | `/dev/ttyACM0` | Follower USB port |
| `--wrist-cam` | `/dev/video2` | Wrist camera |
| `--agent-cam` | `/dev/video8` | Agent-view camera |
| `--agent-depth-cam` | `/dev/video6` | Depth camera |

---

## Troubleshooting

**Motors not found on port**
- Check `sudo chmod 666 /dev/ttyACM0` and that USB cables are seated.
- Ensure motors are powered before running the script.

**Dataset already exists error**
```bash
python record_data.py --prompt "..." --resume    # append episodes
python record_data.py --prompt "..." --overwrite # start fresh
```

**Training: `HF_TOKEN` not set**
```bash
export HF_TOKEN=hf_...
```

**Training: `relative_actions_processor` not found**
This is a registry-name mismatch between `pi05_base` and the local lerobot version.
The fix (alias registration) is already applied in
`lerobot/processor/relative_action_processor.py`.

**Deploy: robot receives zeros / arm does not move**
You are likely calling `lerobot-rollout` directly. Use `deploy_ee_real_robot.py`
instead — it wires the FK/IK processors that convert EE predictions to joint commands.
