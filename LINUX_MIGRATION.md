# Linux Migration & Handoff

Migrating the SO101 SmolVLA deployment + language/semantic experiment campaign from
macOS (M1 Pro) to a Linux machine. This doc lets a fresh Claude Code session on Linux
resume instantly.

## Why Linux (the macOS blocker)

On macOS the Intel RealSense **D455** (serial `234222301106`) does **not** stream color
reliably: librealsense must `sudo`-seize the color interface from the kernel UVC driver,
it contends with the OpenCV/AVFoundation wrist cam, and after a full episode the color
interface **hardware-wedges** — recoverable only by a physical USB replug (software
`hardware_reset` does not clear it). This made a multi-episode campaign impractical.

On **Linux** the RealSense is a plain V4L2/OpenCV device: **no sudo, no seize, no wedging,
no warmup saga.** This is the setup that ran fine "for days".

## Where the campaign stands (as of 2026-07-09)

- **Task piloting:** `cube_to_drawer`, env 1.
- **Checkpoint:** `outputs/train/smolvla_so101_cube_to_drawer/checkpoints/last/pretrained_model`
- **Protocol:** per env, 3 experiment blocks run sequentially on the same env (user resets
  between): **(a) C0** normal instruction, **(b) scene_graph** (Gemini-augmented), **(c)
  ablations C1–C8** (no scene graph). 10 trials/task, video per episode from agent cam,
  ask user grasp/success after each, log via `exp_log.py`, live table.
- **Logged so far:** C0 env1 → **grasp_fail** (approach ✓, grasp ✗).
  Video: `experiments/results/media/cube_to_drawer/C0/trial_20260709_140837/episode.mp4`
- **Next:** scene_graph env1 (was blocked only by the macOS RealSense wedge).
- Ablation strings live in `experiments/ablations.json` (C0–C8 approved, duration_s=30).

## Set up on Linux

1. **Code** (this repo, branch `dev_refi`):
   ```
   git clone https://github.com/Refinath/lerobot-so101-data.git
   cd lerobot-so101-data && git checkout dev_refi && git pull
   ```
2. **Checkpoints** (gitignored, ~1.2 GB each — rsync from the Mac; cube_to_drawer is enough
   to resume the pilot):
   ```
   rsync -avP <mac-user>@<mac-host>:~/work/lerobot-so101-data/outputs/train/smolvla_so101_cube_to_drawer ./outputs/train/
   ```
   (Add the other 3 task dirs when scaling to all tasks.)
3. **Dataset** (only if training/eval needs it, ~4.7 GB):
   ```
   rsync -avP <mac-user>@<mac-host>:~/work/lerobot-so101-data/dataset ./
   ```
4. **Memory** (auto-loaded by Claude Code — place under the Linux project's memory dir;
   the exact path prints on first session start as `~/.claude/projects/<hash>/memory/`):
   ```
   rsync -avP <mac-user>@<mac-host>:~/.claude/projects/-Users-refinath-work-lerobot-so101-data/memory/ <linux-memory-dir>/
   ```
5. **lerobot env** — recreate the conda/venv (`lerobot`) on Linux and `pip install -e .`
   the lerobot package. RealSense: `pip install pyrealsense2` (Linux wheels are stable;
   no from-source build needed unlike the Mac).

## Find camera device paths on Linux

```
v4l2-ctl --list-devices      # or: ls /dev/video*
rs-enumerate-devices | grep -i serial   # confirm RealSense serial 234222301106
```
Wrist cam = the USB webcam `/dev/videoN`. Agent view = RealSense (by serial, or its
`/dev/videoN` color node).

## Proven run command (Linux variant)

The camera-order and warmup hacks are macOS-only; on Linux the defaults work. RealSense by
serial still works (`--agent-cam-serial 234222301106`), OR use `--agent-cam /dev/videoN`:

```
python scripts/run_episode_mac.py \
  --policy-path outputs/train/smolvla_so101_cube_to_drawer/checkpoints/last/pretrained_model \
  --task cube_to_drawer --condition C0 --context-mode standard --env 1 \
  --instruction "Pick up the black cube on top of the blue box and place it inside the drawer" \
  --duration 30 --fps 30 \
  --wrist-cam /dev/video0 --agent-cam-serial 234222301106 \
  --follower-port /dev/ttyACM0
```
(No `sudo` needed on Linux for the RealSense. Follower port is `/dev/ttyACM*` on Linux, not
`/dev/tty.usbmodem*`. Drop the reset+retry loop — that was a macOS-wedge workaround.)

For scene_graph: `--context-mode scene_graph` (needs Gemini API key; see
`EmbodimentSemantic/vlm_benchmarking/.env` → `GEMINI_API_KEY`).

## Scripts reference

- `scripts/run_episode_mac.py` — campaign runner (records episode.mp4, motion check, logging,
  scene-graph thread). Despite the name it's the campaign driver; works on Linux.
- `scripts/deploy_ee_real_robot.py` — bare EE-space deploy (FK/IK processors, no recording).
- `scripts/exp_log.py` — logs to `experiments/results/trials.csv` + `live_table.md`.
- `scripts/check_arm.py` — pre-episode motor health gate.
- macOS-only helpers (ignore on Linux): `reset_realsense.py`, `find_agent_cam.py`,
  `preflight_cameras.py`.

## Resuming the Claude conversation (optional)

The full transcript is `~/.claude/projects/-Users-refinath-work-lerobot-so101-data/<session>.jsonl`
(~16 MB). Cross-machine `--resume` is unreliable (sessions are keyed to the project's
absolute path, which differs on Linux). **Recommended:** start a fresh session on Linux —
the memory files + this doc restore full context. To attempt resume anyway, copy the
`.jsonl` into the Linux project's `~/.claude/projects/<linux-hash>/` dir and run
`claude --resume`.
