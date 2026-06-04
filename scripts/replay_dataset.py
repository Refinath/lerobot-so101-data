#!/usr/bin/env python3
"""Replay one episode from a LeRobot dataset in a local simulator.

This script never connects to the real SO101 robot. By default it visualizes the
recorded follower joint observations in the MuJoCo scene included in this repo.
If MuJoCo is not installed, use ``--backend plot`` for a simple joint trajectory
viewer.
"""

from __future__ import annotations

import os


import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_REPO_ID = "dataset/spatial_dataset"
DEFAULT_FPS = 30
DEFAULT_SCENE = Path(__file__).resolve().parents[1] / "SO101" / "scene.xml"
DEFAULT_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def import_lerobot_dataset():
    try:
        from lerobot.datasets import LeRobotDataset
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Could not import lerobot. Install LeRobot in this environment first, "
            "then rerun this script."
        ) from exc
    return LeRobotDataset


def tensor_to_list(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (int, float)):
        return [float(value)]
    return [float(item) for item in value]


def feature_names(dataset: Any, key: str) -> list[str]:
    feature = dataset.features.get(key, {})
    names = feature.get("names") or []
    return list(names)


def infer_joint_keys(dataset: Any, preferred: str) -> tuple[str, list[str]]:
    if preferred != "auto":
        names = feature_names(dataset, preferred)
        if not names:
            raise ValueError(f"Feature `{preferred}` does not define joint names.")
        return preferred, names

    candidates = [
        "observation.state",
        "observation.joints",
        "observation.motors",
        "action",
    ]
    for key in candidates:
        names = feature_names(dataset, key)
        if names and any(name in names for name in DEFAULT_JOINTS):
            return key, names

    feature_keys = ", ".join(sorted(dataset.features.keys()))
    raise ValueError(
        "Could not infer a joint feature to replay. Pass --joint-feature explicitly. "
        f"Available features: {feature_keys}"
    )


def load_episode(args: argparse.Namespace) -> tuple[Any, str, list[str], int]:
    LeRobotDataset = import_lerobot_dataset()
    dataset = LeRobotDataset(args.repo_id, root=args.root, episodes=[args.episode])
    episode_frames = dataset.hf_dataset.filter(lambda row: row["episode_index"] == args.episode)
    if len(episode_frames) == 0:
        raise ValueError(f"Episode {args.episode} was not found in {args.repo_id}.")

    joint_feature, names = infer_joint_keys(dataset, args.joint_feature)
    fps = args.fps or dataset.fps or DEFAULT_FPS
    return episode_frames, joint_feature, names, fps


def make_joint_vector(row: dict[str, Any], joint_feature: str, names: list[str], radians: bool) -> dict[str, float]:
    values = tensor_to_list(row[joint_feature])
    joints: dict[str, float] = {}
    for joint_name in DEFAULT_JOINTS:
        if joint_name not in names:
            continue
        value = values[names.index(joint_name)]
        joints[joint_name] = value if radians else math.radians(value)
    return joints


def replay_mujoco(args: argparse.Namespace, episode_frames: Any, joint_feature: str, names: list[str], fps: int) -> None:
    try:
        import mujoco
        import mujoco.viewer
    except ModuleNotFoundError as exc:
        raise SystemExit("MuJoCo is not installed. Install `mujoco` or rerun with `--backend plot`.") from exc

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    joint_ids = {
        joint_name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        for joint_name in DEFAULT_JOINTS
    }

    print(f"Replaying episode {args.episode} in MuJoCo at {fps} FPS from `{joint_feature}`")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for idx in range(len(episode_frames)):
            if not viewer.is_running():
                break
            start_t = time.perf_counter()
            joints = make_joint_vector(episode_frames[idx], joint_feature, names, args.radians)
            for joint_name, value in joints.items():
                joint_id = joint_ids[joint_name]
                if joint_id < 0:
                    continue
                qpos_addr = model.jnt_qposadr[joint_id]
                data.qpos[qpos_addr] = value
            mujoco.mj_forward(model, data)
            viewer.sync()
            elapsed = time.perf_counter() - start_t
            time.sleep(max(1.0 / fps - elapsed, 0.0))


def replay_plot(args: argparse.Namespace, episode_frames: Any, joint_feature: str, names: list[str], fps: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit("matplotlib is not installed. Install it or use `--backend mujoco`.") from exc

    timestamps = []
    series = {joint_name: [] for joint_name in DEFAULT_JOINTS if joint_name in names}
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title(f"Episode {args.episode} joint replay")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("joint position")
    lines = {name: ax.plot([], [], label=name)[0] for name in series}
    ax.legend(loc="upper right")

    print(f"Replaying episode {args.episode} as joint plot at {fps} FPS from `{joint_feature}`")
    for idx in range(len(episode_frames)):
        start_t = time.perf_counter()
        row = episode_frames[idx]
        timestamp = float(row["timestamp"]) if "timestamp" in row else idx / fps
        joints = make_joint_vector(row, joint_feature, names, args.radians)
        timestamps.append(timestamp)
        for joint_name in series:
            series[joint_name].append(joints.get(joint_name, float("nan")))
            lines[joint_name].set_data(timestamps, series[joint_name])
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        elapsed = time.perf_counter() - start_t
        time.sleep(max(1.0 / fps - elapsed, 0.0))

    plt.ioff()
    plt.show()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="LeRobot dataset repo id.")
    parser.add_argument("--root", type=Path, default=None, help="Local dataset root. Omit to use HF cache/Hub.")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to replay.")
    parser.add_argument("--fps", type=int, default=None, help="Override replay FPS. Defaults to dataset FPS.")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE, help="MuJoCo scene XML.")
    parser.add_argument(
        "--backend",
        choices=("mujoco", "plot"),
        default="mujoco",
        help="Simulation/viewer backend. `mujoco` uses SO101/scene.xml; `plot` shows joint trajectories.",
    )
    parser.add_argument(
        "--joint-feature",
        default="auto",
        help="Dataset feature containing follower joint positions. Use `auto` to infer it.",
    )
    parser.add_argument(
        "--radians",
        action="store_true",
        help="Treat dataset joint values as radians. By default values are treated as degrees.",
    )
    args = parser.parse_args()

    if args.backend == "mujoco" and not args.scene.exists():
        raise SystemExit(f"MuJoCo scene not found: {args.scene}")

    episode_frames, joint_feature, names, fps = load_episode(args)
    if args.backend == "mujoco":
        replay_mujoco(args, episode_frames, joint_feature, names, fps)
    else:
        replay_plot(args, episode_frames, joint_feature, names, fps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
