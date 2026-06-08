#!/usr/bin/env python3
"""Preprocess SO101 dataset before training.

Two transforms are applied:
  1. ee.wy unwrap — axis-angle wrap events (~2π jumps) in the wrist-pitch
     dimension are corrected per episode using numpy.unwrap. Without this,
     the policy sees artificial 360° wrist snaps in 42% of episodes.
  2. Drop agent_view_depth — only wrist + agent_view RGB cameras are kept.

The result is written to OUTPUT_DATASET. The source is never modified.
Video files for kept cameras are symlinked (not copied) to save disk space.
Do not move or delete the source dataset after running this script.

Usage:
    python scripts/preprocess_dataset.py \\
        --input-dataset  dataset/pick-the-black-bowl-from-the-top-of-the-drawer-and-place-it-on-the-table \\
        --output-dataset dataset/pick-the-black-bowl-from-the-top-of-the-drawer-and-place-it-on-the-table-preprocessed

    python scripts/preprocess_dataset.py \\
        --input-dataset  dataset/pick-... \\
        --output-dataset dataset/pick-...-preprocessed \\
        --force          # overwrite existing output
"""
from __future__ import annotations

import argparse
import json
import shutil
import pathlib

import numpy as np
import pandas as pd


EE_WY_DIM = 4  # index of ee.wy inside the 7-D action/state vector
DEPTH_CAM_KEY = "observation.images.agent_view_depth"
KEEP_CAM_KEYS = ["observation.images.wrist", "observation.images.agent_view"]
STAT_KEYS = ["min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"]


# ── helpers ──────────────────────────────────────────────────────────────────

def unwrap_ee_wy(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with ee.wy unwrapped per episode in 'action'."""
    df = df.copy()
    for ep_idx, grp in df.groupby("episode_index"):
        idx = grp.index
        actions = np.stack(grp["action"].values)       # (N, 7)
        actions[:, EE_WY_DIM] = np.unwrap(actions[:, EE_WY_DIM])
        for i, row_idx in enumerate(idx):
            df.at[row_idx, "action"] = actions[i].astype(np.float32)
    return df


def compute_stats(arr: np.ndarray) -> dict:
    """Compute the same stat keys LeRobot uses, for a (N, D) float array."""
    return {
        "min":   arr.min(axis=0).tolist(),
        "max":   arr.max(axis=0).tolist(),
        "mean":  arr.mean(axis=0).tolist(),
        "std":   arr.std(axis=0).tolist(),
        "count": int(len(arr)),
        "q01":   np.quantile(arr, 0.01, axis=0).tolist(),
        "q10":   np.quantile(arr, 0.10, axis=0).tolist(),
        "q50":   np.quantile(arr, 0.50, axis=0).tolist(),
        "q90":   np.quantile(arr, 0.90, axis=0).tolist(),
        "q99":   np.quantile(arr, 0.99, axis=0).tolist(),
    }


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input-dataset",  required=True, help="Path to source LeRobot dataset.")
    p.add_argument("--output-dataset", required=True, help="Path to write preprocessed dataset.")
    p.add_argument("--force", action="store_true", help="Delete and overwrite output if it exists.")
    return p.parse_args()


def main():
    args = parse_args()
    src = pathlib.Path(args.input_dataset)
    dst = pathlib.Path(args.output_dataset)

    if not src.exists():
        raise FileNotFoundError(f"Input dataset not found: {src}")

    if dst.exists():
        if args.force:
            print(f"Removing existing output: {dst}")
            shutil.rmtree(dst)
        else:
            print(f"Output already exists: {dst}")
            print("  Pass --force to overwrite.")
            return

    # ── 1. load & unwrap data parquets ───────────────────────────────────────
    src_data_dir = src / "data" / "chunk-000"
    data_parts = sorted(src_data_dir.glob("*.parquet"))
    print(f"Loading {len(data_parts)} data parquet(s)...")
    dfs = [pd.read_parquet(p) for p in data_parts]

    print("Unwrapping ee.wy per episode...")
    dfs = [unwrap_ee_wy(df) for df in dfs]

    full_df = pd.concat(dfs, ignore_index=True)
    all_actions = np.stack(full_df["action"].values)          # (N, 7)

    # per-episode action arrays (for episodes parquet stat update)
    ep_action_arrs = {
        int(ep): np.stack(grp["action"].values)
        for ep, grp in full_df.groupby("episode_index")
    }

    # ── 2. write modified data parquets ──────────────────────────────────────
    out_data_dir = dst / "data" / "chunk-000"
    out_data_dir.mkdir(parents=True)
    for part, df in zip(data_parts, dfs):
        df.to_parquet(out_data_dir / part.name, index=False)
    print(f"  Wrote {len(data_parts)} data parquet(s)")

    # ── 3. meta/ — tasks.parquet (unchanged) ─────────────────────────────────
    out_meta = dst / "meta"
    out_meta.mkdir(parents=True)
    shutil.copy2(src / "meta" / "tasks.parquet", out_meta / "tasks.parquet")

    # ── 4. meta/ — info.json (remove depth camera feature) ───────────────────
    info = json.loads((src / "meta" / "info.json").read_text())
    info["features"].pop(DEPTH_CAM_KEY, None)
    (out_meta / "info.json").write_text(json.dumps(info, indent=2))
    print("  Updated info.json (removed agent_view_depth feature)")

    # ── 5. meta/ — stats.json (recompute action stats, drop depth) ───────────
    stats = json.loads((src / "meta" / "stats.json").read_text())
    stats["action"] = compute_stats(all_actions)
    stats.pop(DEPTH_CAM_KEY, None)
    (out_meta / "stats.json").write_text(json.dumps(stats, indent=2))
    print("  Updated stats.json (recomputed action stats, removed depth camera)")

    # ── 6. meta/episodes/ — drop depth cols, recompute action stats ──────────
    src_ep_dir = src / "meta" / "episodes" / "chunk-000"
    out_ep_dir = out_meta / "episodes" / "chunk-000"
    out_ep_dir.mkdir(parents=True)

    for ep_path in sorted(src_ep_dir.glob("*.parquet")):
        ep_df = pd.read_parquet(ep_path)

        # drop all columns that reference agent_view_depth
        drop_cols = [c for c in ep_df.columns if "agent_view_depth" in c]
        ep_df = ep_df.drop(columns=drop_cols)

        # recompute per-episode action stats for the unwrapped data
        for stat_key in STAT_KEYS:
            col = f"stats/action/{stat_key}"
            if col not in ep_df.columns:
                continue
            new_vals = []
            for _, row in ep_df.iterrows():
                ep_idx = int(row["episode_index"])
                s = compute_stats(ep_action_arrs[ep_idx])
                new_vals.append(s[stat_key])
            ep_df[col] = new_vals

        ep_df.to_parquet(out_ep_dir / ep_path.name, index=False)

    n_ep_parts = len(list(src_ep_dir.glob("*.parquet")))
    print(f"  Wrote {n_ep_parts} episodes parquet(s) with updated action stats")

    # ── 7. symlink video files (wrist + agent_view only) ─────────────────────
    for cam_key in KEEP_CAM_KEYS:
        src_vdir = src / "videos" / cam_key / "chunk-000"
        out_vdir = dst / "videos" / cam_key / "chunk-000"
        out_vdir.mkdir(parents=True)
        for mp4 in sorted(src_vdir.glob("*.mp4")):
            (out_vdir / mp4.name).symlink_to(mp4.resolve())
    print(f"  Symlinked videos: {KEEP_CAM_KEYS}")
    print(f"  (depth camera videos NOT included)")

    print(f"\nDone — preprocessed dataset at: {dst}")
    print(f"  Episodes : {full_df['episode_index'].nunique()}")
    print(f"  Frames   : {len(full_df)}")
    print(f"  ee.wy    : unwrapped  (range now {all_actions[:, EE_WY_DIM].min():.2f}..{all_actions[:, EE_WY_DIM].max():.2f})")
    print(f"  Cameras  : wrist, agent_view  (depth dropped)")


if __name__ == "__main__":
    main()
