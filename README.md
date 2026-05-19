# LeRobot SO101 Data Collection

This repo contains a small SO101 data-collection workflow built on top of
[LeRobot](https://github.com/huggingface/lerobot). The main entry point is
`record_data.py`, which records teleoperated demonstrations from a SO100/SO101
leader arm to a follower arm and stores each task instruction in its own
LeRobot dataset folder.

## What This Repo Is For

- Record real-robot demonstrations with joint states and camera observations.
- Save datasets by natural-language instruction prompt.
- Resume existing prompt datasets without overwriting collected episodes.
- Overwrite a prompt dataset when you intentionally want to start over.
- Choose camera storage as MP4 videos, or PNG frames for new datasets.
- Validate or replay collected datasets with helper scripts.

## Repository Layout

```text
record_data.py              Main recording script.
dataset/                    Prompt-specific LeRobot datasets.
calibration/                Robot and teleoperator calibration files.
scripts/validate_dataset.py Dataset validation helper.
scripts/replay_dataset.py   Dataset replay helper.
SO101/                      URDF, MuJoCo scene, and robot assets.
```

`record_data.py` currently sets `HF_LEROBOT_HOME=./`, so collected datasets are
stored relative to this repo. With the default namespace `dataset`, a prompt like
`"place black bowl in front of the drawer"` is saved under:

```text
dataset/place-black-bowl-in-front-of-the-drawer/
```

## Setup

Run commands from the repo root:

```bash
cd /home/refinath/lerobot-so101-data
source /home/refinath/envs/lerobot/bin/activate
```

Check the recording script help:

```bash
python record_data.py --help
```

Before recording, confirm the hardware constants at the top of `record_data.py`:

- `FOLLOWER_PORT`
- `LEADER_PORT`
- `WRIST_CAMERA_PATH`
- `AGENT_CAMERA_PATH`
- `AGENT_DEPTH_CAMERA_PATH`
- `NUM_EPISODES`
- `EPISODE_TIME_SEC`
- `RESET_TIME_SEC`

For USB serial ports, prefer stable paths from:

```bash
ls -l /dev/serial/by-id/
```

## Recording Data

Create a new dataset for a prompt:

```bash
python record_data.py --prompt "place black bowl in front of the drawer"
```

Resume an existing dataset for the same prompt:

```bash
python record_data.py --prompt "place black bowl in front of the drawer" --resume
```

Overwrite an existing dataset and start fresh:

```bash
python record_data.py --prompt "place black bowl in front of the drawer" --overwrite
```

By default, camera observations are stored as MP4 videos under `videos/`. To
store PNG frames for a newly created dataset, use:

```bash
python record_data.py \
  --prompt "place black bowl in front of the drawer" \
  --save-images
```

To replace an existing video-backed dataset with image-backed storage:

```bash
python record_data.py \
  --prompt "place black bowl in front of the drawer" \
  --overwrite \
  --save-images
```

Note: `--save-images` only applies when creating a new dataset. When using
`--resume`, the existing dataset schema is reused.

## Dataset Modes

| Mode | Purpose |
| --- | --- |
| Default | Create a new dataset. Fails if the prompt dataset already exists. |
| `--resume` | Append new episodes to the existing dataset for that prompt. |
| `--overwrite` | Delete the existing dataset for that prompt and start fresh. |
| `--save-images` | Store camera observations as PNG frames instead of MP4 videos for new datasets. |

## Keyboard Controls

| Key | Purpose |
| --- | --- |
| Space | Start the next episode. |
| Enter | End the current episode or reset loop early. |
| Left arrow | Discard the current episode and record it again. |
| Esc | Stop the whole recording session. |

The script connects the robot first, then waits for Space before recording the
first episode. It waits for Space again before each later episode.

## Dataset Contents

A video-backed dataset typically looks like:

```text
dataset/<prompt-slug>/
  data/chunk-000/file-000.parquet
  meta/info.json
  meta/stats.json
  meta/tasks.parquet
  meta/episodes/chunk-000/file-000.parquet
  videos/observation.images.wrist/chunk-000/file-000.mp4
  videos/observation.images.agent_view/chunk-000/file-000.mp4
  videos/observation.images.agent_view_depth/chunk-000/file-000.mp4
```

The parquet files contain joint states, actions, timestamps, episode indexes,
task indexes, and media references. Camera pixels are stored in `videos/` by
default, or in `images/` when the dataset is created with `--save-images`.

## Validation And Replay

Validate a local dataset:

```bash
python scripts/validate_dataset.py \
  --repo-id dataset/place-black-bowl-in-front-of-the-drawer \
  --root dataset/place-black-bowl-in-front-of-the-drawer
```

Replay one episode in MuJoCo:

```bash
python scripts/replay_dataset.py \
  --repo-id dataset/place-black-bowl-in-front-of-the-drawer \
  --root dataset/place-black-bowl-in-front-of-the-drawer \
  --episode 0
```

If MuJoCo is not available, use the plot backend:

```bash
python scripts/replay_dataset.py \
  --repo-id dataset/place-black-bowl-in-front-of-the-drawer \
  --root dataset/place-black-bowl-in-front-of-the-drawer \
  --episode 0 \
  --backend plot
```

## Troubleshooting

If recording fails while connecting to a motor, check:

- Follower and leader USB ports are not swapped.
- The gripper motor is connected and powered.
- Motor IDs match the SO101 mapping expected by LeRobot.
- The daisy-chain cable is seated correctly.
- The arm is powered before running the script.

If a prompt dataset already exists and the script refuses to start, choose one:

```bash
python record_data.py --prompt "..." --resume
python record_data.py --prompt "..." --overwrite
```

If you expected PNG files but see MP4 videos, the dataset was created without
`--save-images`. Create a new prompt dataset or rerun with `--overwrite
--save-images`.
