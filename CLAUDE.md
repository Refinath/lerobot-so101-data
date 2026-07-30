# SO101 Robot Deployment — Claude Code Instructions

This file tells Claude Code (running on the **personal laptop**) how to run
real-robot deployment experiments for the SO101 arm.

---

## Hardware

| Device | Default path / index |
|---|---|
| Follower arm serial | `/dev/ttyACM0` |
| Wrist camera | `/dev/video2` |
| Agent-view camera | `/dev/video8` |

Override with `--follower-port`, `--wrist-cam`, `--agent-cam` if the device
paths differ on the day.

---

## Environment setup (run once per terminal session)

```bash
cd /home/r84368868/lerobot-so101-data
source /home/r84368868/miniconda3/bin/activate /home/r84368868/envs/lerobot/
```

---

## Running a campaign (the normal workflow)

The **campaign runner** executes all 10 ablation episodes for one task in order,
prompts for outcomes after each, and writes results to `experiments/results/trials.csv`.

```bash
# Pi0.5 policy:
python scripts/campaign_runner.py --task <task_key> --policy pi05

# SmolVLA policy:
python scripts/campaign_runner.py --task <task_key> --policy smolvla

# Resume from episode 4 (0-indexed) after an interruption:
python scripts/campaign_runner.py --task <task_key> --policy pi05 --start-episode 3
```

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

## Episode schedule (Option B)

Each of the 10 env setups runs a different condition, in this order:

| Ep | Condition | Description |
|---|---|---|
| 1 | C0 | Standard full instruction |
| 2 | scene_graph | Gemini VLM augments the instruction with object positions |
| 3 | C1 | Empty instruction string |
| 4 | C2 | Garbage words (fixed nonsense, same for all runs) |
| 5 | C3 | Shuffled words (seed=42, fixed per task) |
| 6 | C4 | Cross-task instruction (coherent but wrong task) |
| 7 | C5 | Action verb only |
| 8 | C6 | Objects only (no verb, no spatial relation) |
| 9 | C7 | Wrong object (one key noun swapped) |
| 10 | covgate | Standard C0 instruction + CovGate K=4 variance reduction |

The exact instruction strings for each condition and task are in
`experiments/ablations.json`.

---

## Questions to ask after each episode

The campaign runner prompts automatically, but for reference:

1. **Approached?** — Did the arm move toward the target object and get close enough?
2. **Grasped?** — Did the gripper successfully grab the object?
3. **Task success?** — Was the full placement task completed?
4. **Failure mode** (if not success): `approach_fail`, `grasp_fail`, `transport_fail`, `placement_fail`, `wrong_object`, `no_motion`
5. **Notes** — free text (e.g. "slipped at final placement", "hit edge of box")

---

## Result recording

All outcomes are written to `experiments/results/trials.csv` (created automatically).
Schema: `timestamp, task, task_key, policy, condition, context_mode, episode_idx,
env_idx, instruction, approached, grasped, task_success, failure_mode, ttc_s,
media_dir, notes`.

Do **not** manually edit this file; the campaign runner appends atomically.

---

## Running a single episode manually

If you need to run one condition without the full campaign loop:

```bash
# Standard / ablation conditions (C0-C8, scene_graph):
python scripts/run_episode_mac.py \
    --policy-path outputs/train/pi05_<full-task-name>/checkpoints/last/pretrained_model \
    --task "<C0 instruction>" \
    --duration 30 \
    --context-mode standard     # or scene_graph

# CovGate:
python scripts/deploy_ee_covgate.py \
    --policy-path outputs/train/pi05_<full-task-name>/checkpoints/last/pretrained_model \
    --task "<C0 instruction>" \
    --duration 30 \
    --k-samples 4
```

---

## Checkpoint paths

Pattern: `outputs/train/<policy>_<full-task-name>/checkpoints/last/pretrained_model`

Example:
```
outputs/train/pi05_pick-up-the-black-cube-on-top-of-the-blue-box-and-place-it-inside-the-drawer/checkpoints/last/pretrained_model
outputs/train/smolvla_pick-up-the-black-cube-on-top-of-the-blue-box-and-place-it-inside-the-drawer/checkpoints/last/pretrained_model
```

The campaign runner resolves these automatically from `experiments/ablations.json`.

---

## Safety notes

- Always home the arm before each episode.
- Keep the robot base bolted / clamped — the drawer task involves significant reach.
- The campaign runner pauses between episodes and waits for ENTER before launching.
- Press Ctrl+C to abort an in-progress episode; the runner will prompt before continuing.
- CovGate takes ~4× as long to compute the first action chunk; this is expected.
