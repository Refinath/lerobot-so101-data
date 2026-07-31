# SO101 Real-Robot Deployment Guide

Describes the full deployment workflow for the SO101 arm on an **Ubuntu laptop**.
Covers environment setup, running campaigns, single-episode commands, camera
identification, control-loop timing, and troubleshooting.

---

## 0. TL;DR quick start

```bash
cd /home/r84368868/lerobot-so101-data
source /home/r84368868/miniconda3/bin/activate /home/r84368868/envs/lerobot/

# Run the full 10-episode campaign for a task:
python scripts/campaign_runner.py \
    --task cube_on_top_bbox_to_drawer \
    --policy pi05
```

The campaign runner handles everything: shows each episode's condition and
instruction, launches the correct deploy script, prompts for outcomes, and
writes results to `experiments/results/trials.csv`.

---

## 1. Hardware

| Device | Default path |
|---|---|
| Follower arm serial | `/dev/ttyACM0` |
| Wrist camera | `/dev/video2` |
| Agent-view camera | `/dev/video8` |

On Linux, device paths are stable across sessions (assigned by udev). Verify
they are correct after replugging with `ls /dev/ttyACM* /dev/video*`.

Both cameras are driven by **OpenCV** (`/dev/videoN` path). This matches the
training data collection setup — do not use the librealsense backend even if
the agent-view camera is a RealSense physically.

Override defaults with:
```bash
python scripts/campaign_runner.py --task <task> --policy pi05 \
    --follower-port /dev/ttyACM1 \
    --wrist-cam /dev/video4 \
    --agent-cam-opencv /dev/video6
```

---

## 2. Environment setup (once per terminal session)

```bash
cd /home/r84368868/lerobot-so101-data
source /home/r84368868/miniconda3/bin/activate /home/r84368868/envs/lerobot/
```

---

## 3. Identifying camera devices (Linux)

On Linux, camera paths are stable but it is worth confirming which `/dev/videoN`
corresponds to which physical camera after replugging:

```bash
v4l2-ctl --list-devices
```

This shows each camera with its device nodes. Alternatively, capture a test frame:

```bash
python - <<'EOF'
import cv2
for dev in ["/dev/video2", "/dev/video4", "/dev/video6", "/dev/video8"]:
    cap = cv2.VideoCapture(dev)
    if not cap.isOpened():
        print(f"{dev}: not available"); continue
    ok, frame = cap.read()
    if ok:
        cv2.imwrite(f"/tmp/cam_{dev.split('/')[-1]}.jpg", frame)
        print(f"{dev}: OK — saved /tmp/cam_{dev.split('/')[-1]}.jpg")
    cap.release()
EOF
```

Open the saved frames:
- Wrist cam → shows the gripper close-up (teal fingers)
- Agent-view cam → shows the full workspace (arm + objects)

---

## 4. Pre-flight checklist (every session)

1. **Arm check** — verify motors are live and temperatures are safe:
   ```bash
   python scripts/check_arm.py --follower-port /dev/ttyACM0
   ```
   This reads joint positions, temperatures, and does a 5° wrist micro-move.
   If a servo is missing (`no status packet`), power-cycle the arm and reseat
   the daisy-chain cable near that motor.

2. **Camera devices** — confirm `/dev/video2` (wrist) and `/dev/video8`
   (agent-view) are present: `ls /dev/video*`.

3. **Scene arrangement** — set up objects for the task. Exact positions are
   randomised per env-setup, but the correct objects must be present.

4. **Start pose** — place the arm in a raised, mid-workspace ready pose
   (similar to where teleop episodes started). Tucked or edge poses are
   out-of-distribution and will cause an immediate lurch on first action.

5. **Home** — home the arm before each episode.

---

## 5. Running a campaign (normal workflow)

The campaign runner executes 10 episodes for one task in the fixed Option-B order,
prompts for outcomes after each, and appends to `experiments/results/trials.csv`.

```bash
# Pi0.5 policy:
python scripts/campaign_runner.py --task <task_key> --policy pi05

# SmolVLA policy:
python scripts/campaign_runner.py --task <task_key> --policy smolvla

# Resume after interruption (0-indexed; to resume from episode 4 use --start-episode 3):
python scripts/campaign_runner.py --task <task_key> --policy pi05 --start-episode 3
```

### Episode schedule (Option B)

| Ep | Condition | What the policy receives |
|---|---|---|
| 1 | C0 | Full instruction (baseline) |
| 2 | scene_graph | C0 + Gemini VLM scene-graph prefix |
| 3 | C1 | Empty string |
| 4 | C2 | Fixed garbage words |
| 5 | C3 | Shuffled words (seed=42, fixed per task) |
| 6 | C4 | Coherent instruction for a different task |
| 7 | C5 | Action verb only |
| 8 | C6 | Objects only (no verb, no spatial relation) |
| 9 | C7 | Wrong object (one key noun swapped) |
| 10 | covgate | C0 instruction + CovGate K=4 variance reduction |

All instruction strings are in `experiments/ablations.json`.

### Valid task keys

| Task key | Full task name |
|---|---|
| `bowl_next_to_blue_box_to_drawer` | pick-up-the-black-bowl-next-to-the-blue-box-and-place-it-on-the-drawer |
| `bowl_next_to_yellow_rect_to_bbox` | pick-up-the-black-bowl-next-to-the-yellow-rectangle-and-place-it-on-top-of-the-blue-box |
| `bowl_on_top_of_bbox_to_drawer` | pick-up-the-black-bowl-on-top-of-the-blue-box-and-place-it-on-the-drawer |
| `bowl_on_top_of_cookies_to_bbox` | pick-up-the-black-bowl-on-top-of-the-cookies-and-place-it-on-top-of-the-blue-box |
| `bowl_on_top_drawer_to_yellow_rect` | pick-up-the-black-bowl-on-top-the-drawer-and-place-it-on-top-of-the-yellow-rectangle |
| `cube_between_bbox_and_rect_to_drawer` | pick-up-the-black-cube-between-the-blue-box-and-the-yellow-rectangle-and-place-it-inside-the-drawer |
| `cube_inside_drawer_to_bbox` | pick-up-the-black-cube-inside-the-drawer-and-place-it-on-top-of-the-blue-box |
| `cube_next_to_bbox_to_rect` | pick-up-the-black-cube-next-to-the-blue-box-and-place-it-on-top-of-the-yellow-rectangle |
| `cube_next_to_cookies_to_rect` | pick-up-the-black-cube-next-to-the-cookies-and-place-it-on-top-of-the-yellow-rectangle |
| `cube_on_top_bbox_to_drawer` | pick-up-the-black-cube-on-top-of-the-blue-box-and-place-it-inside-the-drawer |
| `cube_on_top_cookies_to_drawer` | pick-up-the-black-cube-on-top-of-the-cookies-and-place-it-inside-the-drawer |
| `cube_on_top_rect_to_drawer` | pick-up-the-black-cube-on-top-of-the-yellow-rectangle-and-place-it-inside-the-drawer |

---

## 6. Running a single episode manually

If you need to run one condition outside the campaign loop:

```bash
# Standard / ablation conditions (C0–C8):
python scripts/run_episode_mac.py \
    --policy-path outputs/train/pi05_<full-task-name>/checkpoints/last/pretrained_model \
    --task cube_on_top_bbox_to_drawer \
    --instruction "Pick up the black cube on top of the blue box and place it inside the drawer" \
    --condition C0 \
    --context-mode standard \
    --duration 30

# scene_graph (Gemini augments the instruction at episode start):
python scripts/run_episode_mac.py \
    --policy-path outputs/train/pi05_<full-task-name>/checkpoints/last/pretrained_model \
    --task cube_on_top_bbox_to_drawer \
    --instruction "Pick up the black cube on top of the blue box and place it inside the drawer" \
    --condition scene_graph \
    --context-mode scene_graph \
    --duration 30

# CovGate (K=4 independent samples, eigenvalue-weighted gating):
python scripts/deploy_ee_covgate.py \
    --policy-path outputs/train/pi05_<full-task-name>/checkpoints/last/pretrained_model \
    --task "Pick up the black cube on top of the blue box and place it inside the drawer" \
    --duration 30 \
    --k-samples 4
```

### Dry run (no robot, no cameras — tests policy loading and logic only)

```bash
python scripts/run_episode_mac.py \
    --policy-path outputs/train/pi05_<task>/checkpoints/last/pretrained_model \
    --task cube_on_top_bbox_to_drawer \
    --instruction "..." \
    --condition C0 \
    --dry-run --dry-seconds 6
```

---

## 7. Checkpoint paths

Pattern: `outputs/train/<policy>_<full-task-name>/checkpoints/last/pretrained_model`

Example:
```
outputs/train/pi05_pick-up-the-black-cube-on-top-of-the-blue-box-and-place-it-inside-the-drawer/checkpoints/last/pretrained_model
outputs/train/smolvla_pick-up-the-black-cube-on-top-of-the-blue-box-and-place-it-inside-the-drawer/checkpoints/last/pretrained_model
```

The campaign runner resolves checkpoint paths automatically from
`experiments/ablations.json` — you only need the task key and policy name.

---

## 8. Control loop timing

SmolVLA and Pi0.5 produce action **chunks** (50 steps each) at 30 Hz. 49 of
every 50 steps pop a pre-computed action (~3–5 ms); the 50th step re-runs the
model synchronously (~50–200 ms on GPU). This creates one brief hitch per chunk
(every ~1.7 s) rather than a uniform slowdown.

Run at `--fps 30` (the default). Warnings like *"loop slower than target"*
appearing once every ~1–2 s are expected and harmless.

**CovGate** draws K=4 samples per re-plan, so the hitch is ~4× longer. All
inter-hitch steps still run at full 30 Hz.

---

## 9. Result recording

The campaign runner appends one row per episode to `experiments/results/trials.csv`.

Schema: `timestamp, task, task_key, policy, condition, context_mode, episode_idx,
env_idx, instruction, approached, grasped, task_success, failure_mode, ttc_s,
media_dir, notes`.

Do not edit this file manually — the runner appends atomically.

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Timed out waiting for frame from camera` | Wrong camera device path | Check `ls /dev/video*`; verify with camera probe (§3) |
| Camera opens but returns 0 frames | Device path correct but wrong physical camera | Swap wrist-cam and agent-cam paths; verify with probe (§3) |
| `Failed to write 'Lock' on id_=N … no status packet` | Servo dropped off motor bus | Power-cycle arm, reseat daisy-chain cable near that motor |
| `Permission denied: /dev/ttyACM0` | User not in `dialout` group | `sudo usermod -aG dialout $USER` then log out/in |
| `Permission denied: /dev/video*` | User not in `video` group | `sudo usermod -aG video $USER` then log out/in |
| Arm lurches immediately on first action | Out-of-distribution start pose | Start from raised, mid-workspace ready pose (§4) |
| Arm approaches but misses grasp | Control rate too low | Ensure `--fps 30`; verify GPU is used (`nvidia-smi`) |
| Warning every ~1–2 s, motion otherwise smooth | Normal periodic re-plan hitch | Expected — see §8 |
| `GOOGLE_API_KEY not set` | scene_graph condition needs Gemini key | `export GOOGLE_API_KEY=<key>` before running |
