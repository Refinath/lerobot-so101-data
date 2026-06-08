#!/usr/bin/env python3
"""Open-loop evaluation of a trained EE-space policy against recorded demonstrations.

This feeds the policy the *real* recorded observations (images + state) frame by
frame from chosen episodes — exactly as it would see them at deployment — and
compares its predicted EE-space actions against the actions the human
teleoperator actually took at those same instants. It never touches the real
robot or a simulator; it is a pure sanity check that the policy has learned to
reproduce trajectories similar to the demonstrations before risking a real-robot
rollout.

Usage:
    python scripts/eval_policy_openloop.py \\
        --policy-path outputs/train/pi05_so101_bowl_placement/checkpoints/010000/pretrained_model \\
        --root dataset/pick-the-black-bowl-from-the-top-of-the-drawer-and-place-it-on-the-table-preprocessed \\
        --episodes 0 36 71 \\
        --output-dir outputs/eval/pi05_openloop
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy-path", required=True, help="Local checkpoint dir (pretrained_model).")
    parser.add_argument("--repo-id", default="Refinath/so101_bowl_placement")
    parser.add_argument("--root", required=True, help="Local dataset root (preprocessed dataset).")
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 36, 71], help="Episode indices to replay.")
    parser.add_argument("--output-dir", default="outputs/eval/openloop")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int, default=None, help="Optionally cap frames per episode.")
    parser.add_argument("--compile", action="store_true", help="Keep torch.compile enabled (slow warmup).")
    return parser.parse_args()


def main():
    args = parse_args()

    from lerobot.configs import PreTrainedConfig
    from lerobot.configs.types import FeatureType
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.utils.utils import init_logging

    init_logging()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    policy_config = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_config.pretrained_path = args.policy_path
    policy_config.device = args.device
    if not args.compile:
        policy_config.compile_model = False

    image_keys = [k for k, ft in policy_config.input_features.items() if ft.type == FeatureType.VISUAL]
    print(f"Policy:     {policy_config.type}  ({args.policy_path})")
    print(f"Image keys: {image_keys}")

    summary = {}

    for ep in args.episodes:
        print(f"\n=== Episode {ep} ===")
        dataset = LeRobotDataset(args.repo_id, root=args.root, episodes=[ep])
        action_names = dataset.features["action"]["names"]
        task_str = dataset[0]["task"]
        n = len(dataset) if args.max_frames is None else min(len(dataset), args.max_frames)
        print(f"task: {task_str!r}  |  frames: {n}")

        policy = make_policy(policy_config, ds_meta=dataset.meta)
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_config,
            pretrained_path=args.policy_path,
            dataset_stats=dataset.meta.stats,
            preprocessor_overrides={"device_processor": {"device": args.device}},
        )
        policy.eval()
        policy.reset()

        gt_actions, pred_actions, gt_states = [], [], []
        for i in range(n):
            item = dataset[i]
            batch = {
                "observation.state": item["observation.state"].unsqueeze(0),
                "task": [item["task"]],
            }
            for key in image_keys:
                batch[key] = item[key].unsqueeze(0)

            obs = preprocessor(batch)
            with torch.no_grad():
                action = policy.select_action(obs)
            action = postprocessor(action)

            pred_actions.append(action.squeeze(0).detach().cpu().float().numpy())
            gt_actions.append(item["action"].numpy())
            gt_states.append(item["observation.state"].numpy())

            if (i + 1) % 100 == 0 or i == n - 1:
                print(f"  ... {i + 1}/{n} frames")

        gt = np.stack(gt_actions)
        pred = np.stack(pred_actions)
        states = np.stack(gt_states)

        mae = np.abs(pred - gt).mean(axis=0)
        rmse = np.sqrt(((pred - gt) ** 2).mean(axis=0))
        # Correlation between predicted and ground-truth trajectory per dimension —
        # high correlation means the policy follows the same *shape* of trajectory
        # even if there's a constant offset/scale difference.
        corr = np.array(
            [np.corrcoef(pred[:, d], gt[:, d])[0, 1] if gt[:, d].std() > 1e-6 else float("nan") for d in range(gt.shape[1])]
        )

        ep_summary = {
            "task": task_str,
            "num_frames": int(n),
            "overall_mae": float(mae.mean()),
            "overall_rmse": float(rmse.mean()),
            "per_dim": {
                name: {"mae": float(mae[d]), "rmse": float(rmse[d]), "corr": float(corr[d])}
                for d, name in enumerate(action_names)
            },
        }
        summary[f"episode_{ep}"] = ep_summary

        print(f"  overall MAE={mae.mean():.4f}  RMSE={rmse.mean():.4f}")
        for d, name in enumerate(action_names):
            print(f"    {name:16s} MAE={mae[d]:7.4f}  RMSE={rmse[d]:7.4f}  corr={corr[d]:+.3f}")

        _plot_episode(out_dir, ep, task_str, action_names, gt, pred, states, dataset.fps, mae.mean())

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary -> {summary_path}")
    print(f"Saved trajectory plots -> {out_dir}/episode_*_trajectory.png")


def _plot_episode(out_dir, ep, task_str, action_names, gt, pred, states, fps, overall_mae):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = gt.shape[0]
    t = np.arange(n) / fps
    fig, axes = plt.subplots(len(action_names), 1, figsize=(11, 2.1 * len(action_names)), sharex=True)
    for d, name in enumerate(action_names):
        ax = axes[d]
        ax.plot(t, gt[:, d], label="ground truth (teleop action)", linewidth=1.6, color="tab:blue")
        ax.plot(t, pred[:, d], label="policy prediction (open-loop)", linewidth=1.3, linestyle="--", color="tab:orange")
        ax.plot(t, states[:, d], label="recorded EE state", linewidth=0.9, linestyle=":", color="gray", alpha=0.6)
        ax.set_ylabel(name)
        ax.grid(alpha=0.25)
        if d == 0:
            ax.legend(loc="upper right", fontsize=8, ncol=3)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"Episode {ep}  —  '{task_str}'\noverall action MAE = {overall_mae:.4f}")
    fig.tight_layout()
    plot_path = out_dir / f"episode_{ep}_trajectory.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"  saved plot -> {plot_path}")


if __name__ == "__main__":
    main()
