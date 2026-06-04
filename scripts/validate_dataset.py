#!/usr/bin/env python3
"""Validate a LeRobot dataset recorded for the SO101 setup.

This script intentionally starts from the standard validation path: loading the
dataset through ``LeRobotDataset``. That exercises LeRobot's metadata, parquet,
timestamp, and video-decoding checks. It then adds a few practical audits that
are useful before training on real robot data.
"""
from __future__ import annotations

import os

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_REPO_ID = "dataset/spatial_dataset"


def add_lerobot_path(path: Path | None) -> None:
    if path is None:
        return

    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise SystemExit(f"LeRobot path does not exist: {resolved}")

    src_path = resolved / "src"
    if (src_path / "lerobot").is_dir():
        import_path = src_path
    elif (resolved / "lerobot").is_dir():
        import_path = resolved
    else:
        raise SystemExit(
            "Could not find a `lerobot` package at the provided path. Pass either "
            "the LeRobot repo root or a directory that directly contains `lerobot/`."
        )

    sys.path.insert(0, str(import_path))


def parse_episode_list(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def import_lerobot_dataset(lerobot_path: Path | None):
    add_lerobot_path(lerobot_path)
    try:
        from lerobot.datasets import LeRobotDataset
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Could not import lerobot. Install LeRobot in this environment first, "
            "or pass a local checkout with `--lerobot-path /path/to/lerobot`."
        ) from exc
    return LeRobotDataset


def is_numeric_sequence(value: Any) -> bool:
    if hasattr(value, "shape"):
        return True
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, list):
        return all(is_numeric_sequence(item) for item in value)
    return False


def flatten_numbers(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        out: list[float] = []
        for item in value:
            out.extend(flatten_numbers(item))
        return out
    return []


def frame_has_bad_numeric(frame: dict[str, Any], keys: list[str]) -> list[str]:
    bad_keys: list[str] = []
    for key in keys:
        value = frame.get(key)
        if not is_numeric_sequence(value):
            continue
        numbers = flatten_numbers(value)
        if any(not math.isfinite(number) for number in numbers):
            bad_keys.append(key)
    return bad_keys


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metadata_summary(dataset: Any) -> dict[str, Any]:
    meta = getattr(dataset, "meta", None)
    return {
        "repo_id": getattr(dataset, "repo_id", None),
        "root": str(getattr(dataset, "root", "")),
        "fps": getattr(dataset, "fps", None),
        "frames": len(dataset),
        "episodes": getattr(dataset, "num_episodes", None),
        "features": sorted(getattr(dataset, "features", {}).keys()),
        "video_keys": sorted(getattr(meta, "video_keys", []) or []),
    }


def audit_dataset(dataset: Any, sample_frames: int, strict_fps: bool) -> list[str]:
    issues: list[str] = []
    hf_dataset = dataset.hf_dataset
    columns = set(getattr(hf_dataset, "column_names", []) or [])
    required = {"episode_index", "frame_index", "timestamp", "action"}
    missing = sorted(required - columns)
    if missing:
        issues.append(f"Missing expected column(s): {', '.join(missing)}")

    total_frames = len(dataset)
    if total_frames == 0:
        issues.append("Dataset contains no frames.")
        return issues

    action_names = dataset.features.get("action", {}).get("names", [])
    if not action_names:
        issues.append("Feature `action` has no named dimensions.")

    step = max(total_frames // max(sample_frames, 1), 1)
    sampled_indices = list(range(0, total_frames, step))[:sample_frames]
    bad_numeric: Counter[str] = Counter()
    unreadable: list[int] = []
    for idx in sampled_indices:
        try:
            frame = dataset[idx]
        except Exception as exc:  # noqa: BLE001 - report all dataset read failures.
            unreadable.append(idx)
            issues.append(f"Could not read frame {idx}: {exc}")
            continue
        for key in frame_has_bad_numeric(frame, list(frame.keys())):
            bad_numeric[key] += 1
    for key, count in bad_numeric.items():
        issues.append(f"Found non-finite numeric values in `{key}` for {count} sampled frame(s).")
    if unreadable:
        return issues

    if {"episode_index", "frame_index", "timestamp"} <= columns:
        rows = hf_dataset.select_columns(["episode_index", "frame_index", "timestamp"])
        per_episode: dict[int, list[tuple[int, float]]] = {}
        for row in rows:
            per_episode.setdefault(int(row["episode_index"]), []).append(
                (int(row["frame_index"]), float(row["timestamp"]))
            )

        expected_dt = 1.0 / float(dataset.fps)
        tolerance = 1e-3 if strict_fps else max(1e-3, expected_dt * 0.20)
        for episode_index, values in sorted(per_episode.items()):
            frame_indices = [item[0] for item in values]
            timestamps = [item[1] for item in values]
            expected_frame_indices = list(range(len(frame_indices)))
            if frame_indices != expected_frame_indices:
                issues.append(f"Episode {episode_index} has non-contiguous frame_index values.")
            if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
                issues.append(f"Episode {episode_index} timestamps are not strictly increasing.")
            for left, right in zip(timestamps, timestamps[1:]):
                if abs((right - left) - expected_dt) > tolerance:
                    issues.append(
                        f"Episode {episode_index} timestamp step differs from fps "
                        f"(expected {expected_dt:.6f}s, saw {right - left:.6f}s)."
                    )
                    break

    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lerobot-path",
        type=Path,
        default=None,
        help="Local LeRobot repo root or import parent. Prepended to PYTHONPATH before import.",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="LeRobot dataset repo id.")
    parser.add_argument("--root", type=Path, default=None, help="Local dataset root. Omit to use HF cache/Hub.")
    parser.add_argument("--episodes", default=None, help="Comma-separated episode indices to validate.")
    parser.add_argument("--sample-frames", type=int, default=64, help="Number of frames to decode and audit.")
    parser.add_argument(
        "--no-download-videos",
        action="store_true",
        help="Skip video downloads when validating a Hub dataset.",
    )
    parser.add_argument(
        "--strict-fps",
        action="store_true",
        help="Use a 1 ms timestamp tolerance instead of a relaxed 20%% tolerance.",
    )
    args = parser.parse_args()

    LeRobotDataset = import_lerobot_dataset(args.lerobot_path)
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.root,
        episodes=parse_episode_list(args.episodes),
        download_videos=not args.no_download_videos,
    )

    print("Dataset summary:")
    for key, value in metadata_summary(dataset).items():
        print(f"  {key}: {value}")

    info = load_json(Path(dataset.root) / "meta" / "info.json")
    if info is not None:
        print(f"  codebase_version: {info.get('codebase_version', 'unknown')}")

    issues = audit_dataset(dataset, args.sample_frames, args.strict_fps)
    if issues:
        print("\nValidation failed:")
        for issue in issues:
            print(f"  - {issue}")
        return 1

    print("\nValidation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
