# SO101 Real-Robot Deployment Guide

Describes the full deployment workflow for the SO101 arm on a **macOS laptop
(Apple Silicon)**. Covers environment setup, running campaigns, single-episode
commands, camera identification, control-loop timing, and troubleshooting.

---

## 0. TL;DR quick start

```bash
cd /path/to/lerobot-so101-data
source /Users/refinath/work/envs/lerobot/bin/activate   # or conda activate …

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

| Device | Default value |
|---|---|
| Machine | Apple Silicon Mac (M1 Pro), macOS |
| Python env | `/Users/refinath/work/envs/lerobot` |
| lerobot | editable checkout at `/Users/refinath/work/lerobot` |
| Robot serial port | `/dev/tty.usbmodem5B140303851` |
| Wrist camera | USB webcam — **OpenCV integer index** (re-verify each session) |
| Agent-view camera | USB webcam — **OpenCV integer index** (re-verify each session) |

Both cameras are driven by OpenCV regardless of hardware (this matches the
training data collection setup). Camera indices reshuffle on macOS whenever
USB devices are plugged/unplugged — always verify at the start of a session
(see §3).

Override defaults with:
```bash
python scripts/campaign_runner.py --task <task> --policy pi05 \
    --follower-port /dev/tty.usbmodemXXXX \
    --wrist-cam 1 \
    --agent-cam-opencv 2
```

---

## 2. Environment setup (once per terminal session)

```bash
cd /path/to/lerobot-so101-data
source /Users/refinath/work/envs/lerobot/bin/activate
# If using conda:
# conda activate /Users/refinath/work/envs/lerobot
export PYTORCH_ENABLE_MPS_FALLBACK=1   # required on Apple Silicon
```

---

## 3. Finding camera indices (macOS)

Camera OpenCV indices are assigned by AVFoundation and **change when USB devices
are plugged or unplugged**. Run this probe at the start of each session:

```python
import cv2, time
for i in range(6):
    cap = cv2.VideoCapture(i)
    if not cap.isOpened():
        cap.release(); continue
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    time.sleep(0.5)
    ok, frame = 0, None
    for _ in range(6):
        r, f = cap.read(); ok += r; time.sleep(0.05)
    if ok and f is not None:
        cv2.imwrite(f"/tmp/cam_{i}.jpg", f)
    print(f"index {i}: frames_ok={ok}/6")
    cap.release()
```

Open `/tmp/cam_*.jpg`:
- Wrist cam → shows the gripper close-up (teal fingers)
- Agent-view cam → shows the full workspace (arm + objects)

Both cameras must be accessed via OpenCV — this is what the training data was
collected with. Do **not** use the librealsense backend even if the agent-view
camera is a RealSense physically; use `--agent-cam-opencv <index>` with the
OpenCV index found above.

---

## 4. Pre-flight checklist (every session)

1. **Arm power** — confirm motors are powered; check with:
   ```bash
   python scripts/check_arm.py --follower-port /dev/tty.usbmodemXXXX
   ```
   This reads joint positions, temperatures, and does a 5° wrist micro-move.
   If a servo is missing (`no status packet`), power-cycle and reseat the
   daisy-chain cable near that motor.

2. **Camera indices** — run the probe above; plug order changes indices.

3. **Scene arrangement** — set up objects for the task. Exact positions are
   randomised per env-setup, but the objects must be the right ones for the task.

4. **Start pose** — place the arm in a raised, mid-workspace ready pose (similar
   to where teleop episodes started). Tucked or edge poses are out-of-distribution
   and will cause an immediate lurch.

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

# Resume after interruption (0-indexed; resume from episode 4 = --start-episode 3):
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
# Standard / ablation conditions (C0–C8) or scene_graph:
python scripts/run_episode_mac.py \
    --policy-path outputs/train/pi05_<full-task-name>/checkpoints/last/pretrained_model \
    --task cube_on_top_bbox_to_drawer \
    --instruction "Pick up the black cube on top of the blue box and place it inside the drawer" \
    --condition C0 \
    --context-mode standard \
    --duration 30 \
    --wrist-cam 0 \
    --agent-cam-opencv 2

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

### Dry run (no robot, no cameras — tests the policy loading and logic only)

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

SmolVLA and Pi0.5 are trained at 30 Hz but produce action **chunks** (50 steps
each). 49 of every 50 steps pop a pre-computed action (~3–5 ms); the 50th step
re-runs the model synchronously (~200–400 ms on MPS).

This creates a periodic hitch every ~1.7 s rather than a uniform slowdown. At
`--fps 30` the loop runs at the full trained rate between hitches, which is
visibly faster and more decisive than running at a lower fps.

Run at `--fps 30` (the default). Warnings like *"loop slower than target"* appearing
once every ~1–2 s are expected and harmless.

**CovGate** draws K=4 samples per re-plan, so the hitch is ~4× longer
(~800–1500 ms on MPS). Everything else is identical.

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
| `Timed out waiting for frame from camera` | Wrong wrist index | Re-run camera probe (§3); indices reshuffle on USB plug/unplug |
| Camera opens but returns 0 frames | OpenCV index is wrong | Re-run camera probe (§3); use `--agent-cam-opencv`, not a serial number |
| `Failed to write 'Lock' on id_=N … no status packet` | Servo dropped off motor bus | Power-cycle arm, reseat daisy-chain cable near that motor |
| `Device 'cuda' is not available` | Checkpoint hard-codes cuda | Set `--device mps` or leave blank (auto-selects mps on Apple Silicon) |
| Arm lurches immediately | Out-of-distribution start pose | Start from raised, mid-workspace ready pose (§4) |
| Arm approaches but misses grasp | Control rate / timing | Ensure `--fps 30`; for highest success rate use Linux + GPU |
| Warning every ~1–2 s, motion otherwise smooth | Normal periodic re-plan hitch | Expected — see §8 |
| `EXIT=139` (segfault) after camera connect | `cv2`/`av` libavdevice conflict on macOS | Retry — it's an intermittent crash (~50% rate); keep only needed cameras attached |
| `GOOGLE_API_KEY not set` | scene_graph needs Gemini key | `export GOOGLE_API_KEY=<key>` before running |
