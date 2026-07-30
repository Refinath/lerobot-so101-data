#!/usr/bin/env python3
"""Campaign runner for SO101 language-ablation + CovGate real-robot experiments.

Runs 10 episodes for one task, each episode using a different condition from
experiments/ablations.json. After each episode the user is prompted for outcomes;
results are appended to experiments/results/trials.csv.

Usage:
    python scripts/campaign_runner.py --task cube_on_top_bbox_to_drawer --policy pi05
    python scripts/campaign_runner.py --task bowl_next_to_blue_box_to_drawer --policy smolvla
    python scripts/campaign_runner.py --task cube_on_top_bbox_to_drawer --policy pi05 \
        --start-episode 4      # resume from episode 4 (0-indexed)

Episode schedule (Option B, fixed order):
    0  C0          standard full instruction
    1  scene_graph VLM-augmented instruction
    2  C1          empty string
    3  C2          garbage
    4  C3          shuffled words (seed=42)
    5  C4          cross-task instruction
    6  C5          action verb only
    7  C6          objects only
    8  C7          wrong object
    9  covgate     standard C0 with CovGate K=4 variance reduction
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ABLATIONS_JSON = ROOT / "experiments" / "ablations.json"
RESULTS_DIR = ROOT / "experiments" / "results"
TRIALS_CSV = RESULTS_DIR / "trials.csv"
MEDIA_DIR = RESULTS_DIR / "media"

SCHEDULE = ["C0", "scene_graph", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "covgate"]

FAILURE_MODES = [
    "none",
    "approach_fail",
    "grasp_fail",
    "transport_fail",
    "placement_fail",
    "wrong_object",
    "no_motion",
]

CSV_FIELDNAMES = [
    "timestamp", "task", "task_key", "policy", "condition", "context_mode",
    "episode_idx", "env_idx",
    "instruction", "approached", "grasped", "task_success",
    "failure_mode", "ttc_s", "media_dir", "notes",
]


def load_ablations():
    with open(ABLATIONS_JSON) as f:
        return json.load(f)


def ensure_csv():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not TRIALS_CSV.exists():
        with open(TRIALS_CSV, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDNAMES).writeheader()


def append_csv(row: dict):
    with open(TRIALS_CSV, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDNAMES).writerow(row)


def ask_yn(prompt: str) -> bool:
    while True:
        ans = input(prompt + " [y/n]: ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("  Please enter y or n.")


def ask_choice(prompt: str, choices: list) -> str:
    print(prompt)
    for i, c in enumerate(choices):
        print(f"  {i}) {c}")
    while True:
        ans = input(f"  Choice [0-{len(choices)-1}]: ").strip()
        try:
            idx = int(ans)
            if 0 <= idx < len(choices):
                return choices[idx]
        except ValueError:
            pass
        print(f"  Enter a number between 0 and {len(choices)-1}.")


def build_episode_command(
    task_cfg: dict,
    condition: str,
    policy: str,
    instruction: str,
) -> list:
    """Build the subprocess command list for a single episode."""
    checkpoint_key = f"{policy}_checkpoint"
    ckpt = str(ROOT / task_cfg[checkpoint_key])

    if condition == "covgate":
        cmd = [
            sys.executable, str(ROOT / "scripts" / "deploy_ee_covgate.py"),
            "--policy-path", ckpt,
            "--task", instruction,
            "--duration", str(task_cfg["duration_s"]),
            "--k-samples", "4",
            "--gamma", "1.0",
        ]
    else:
        context_mode = "scene_graph" if condition == "scene_graph" else "standard"
        cmd = [
            sys.executable, str(ROOT / "scripts" / "run_episode_mac.py"),
            "--policy-path", ckpt,
            "--task", instruction,
            "--duration", str(task_cfg["duration_s"]),
            "--context-mode", context_mode,
        ]
    return cmd


def run_episode(
    task_key: str,
    task_cfg: dict,
    condition: str,
    policy: str,
    episode_idx: int,
    env_idx: int,
    ablation_cfg: dict,
) -> dict:
    """Interactive episode runner. Returns a filled-in result dict."""

    instructions = task_cfg["instructions"]
    full_task = task_cfg["full_task_name"]

    if condition == "covgate":
        instruction = instructions["C0"]
        context_mode = "covgate"
    elif condition == "scene_graph":
        instruction = instructions["C0"]
        context_mode = "scene_graph"
    else:
        instruction = instructions[condition]
        context_mode = "standard"

    media_subdir = (
        MEDIA_DIR / task_key / condition / f"trial_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    )

    hr = "=" * 70
    print(f"\n{hr}")
    print(f"  Episode {episode_idx + 1}/10  |  Env setup #{env_idx + 1}")
    print(f"  Task:      {full_task}")
    print(f"  Policy:    {policy}")
    print(f"  Condition: {condition}  ({ablation_cfg['conditions'].get(condition, '')})")
    print(f"  Instruction sent to policy:")
    if instruction:
        print(f"    \"{instruction}\"")
    else:
        print(f"    (empty string)")
    print(hr)

    input("\n  Set up the environment (object positions, arm pose), then press ENTER to launch.")

    cmd = build_episode_command(task_cfg, condition, policy, instruction)
    print(f"\n  Running: {' '.join(cmd[:3])} ...\n")
    start_t = datetime.now()

    try:
        proc = subprocess.run(cmd, cwd=str(ROOT))
        returncode = proc.returncode
    except KeyboardInterrupt:
        print("\n  [Interrupted]")
        returncode = -1

    elapsed = (datetime.now() - start_t).total_seconds()
    print(f"\n  Episode finished in {elapsed:.1f}s (exit code {returncode})")

    print("\n  --- Outcome ---")
    approached = ask_yn("  Did the arm approach the target object?")
    grasped = ask_yn("  Did the arm successfully grasp the object?")
    success = ask_yn("  Was the full task completed successfully?")

    if success:
        failure_mode = "none"
    else:
        failure_mode = ask_choice("  What was the failure mode?", FAILURE_MODES)

    notes = input("  Notes (optional, press ENTER to skip): ").strip()

    return {
        "timestamp": start_t.isoformat(timespec="seconds"),
        "task": full_task,
        "task_key": task_key,
        "policy": policy,
        "condition": condition,
        "context_mode": context_mode,
        "episode_idx": episode_idx,
        "env_idx": env_idx,
        "instruction": instruction,
        "approached": approached,
        "grasped": grasped,
        "task_success": success,
        "failure_mode": failure_mode,
        "ttc_s": round(elapsed, 1),
        "media_dir": str(media_subdir),
        "notes": notes,
    }


def print_summary(results: list):
    print("\n" + "=" * 70)
    print("  CAMPAIGN SUMMARY")
    print("=" * 70)
    print(f"  {'EP':>3}  {'Cond':<12}  {'App':>4}  {'Grasp':>5}  {'Success':>7}")
    print("  " + "-" * 44)
    for r in results:
        ep = r["episode_idx"] + 1
        cond = r["condition"]
        app = "Y" if r["approached"] else "N"
        gsp = "Y" if r["grasped"] else "N"
        suc = "Y" if r["task_success"] else "N"
        print(f"  {ep:>3}  {cond:<12}  {app:>4}  {gsp:>5}  {suc:>7}")

    n = len(results)
    if n:
        sr = sum(1 for r in results if r["task_success"]) / n
        print(f"\n  Success rate: {sr:.0%} ({int(sr*n)}/{n})")
    print(f"  Results saved to: {TRIALS_CSV}")
    print("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--task", required=True,
        help="Task key from ablations.json (e.g. cube_on_top_bbox_to_drawer).",
    )
    parser.add_argument(
        "--policy", required=True, choices=["pi05", "smolvla"],
        help="Which policy checkpoint to use.",
    )
    parser.add_argument(
        "--start-episode", type=int, default=0,
        help="0-indexed episode to start from (for resuming a partial campaign).",
    )
    parser.add_argument(
        "--schedule", nargs="+", default=None,
        help="Override the default 10-condition schedule (space-separated condition names).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ablations = load_ablations()

    if args.task not in ablations["tasks"]:
        valid = ", ".join(ablations["tasks"].keys())
        print(f"ERROR: unknown task '{args.task}'.\nValid tasks: {valid}")
        sys.exit(1)

    task_cfg = ablations["tasks"][args.task]
    schedule = args.schedule or SCHEDULE

    ensure_csv()

    print(textwrap.dedent(f"""
        ╔══════════════════════════════════════════════════════════════════════╗
        ║              SO101 CAMPAIGN RUNNER                                   ║
        ╠══════════════════════════════════════════════════════════════════════╣
        ║  Task:    {args.task:<57} ║
        ║  Policy:  {args.policy:<57} ║
        ║  Episodes: {len(schedule)} total, starting from #{args.start_episode + 1:<42} ║
        ╚══════════════════════════════════════════════════════════════════════╝
    """))

    results = []
    for ep_idx, condition in enumerate(schedule):
        if ep_idx < args.start_episode:
            continue
        env_idx = ep_idx
        try:
            row = run_episode(
                task_key=args.task,
                task_cfg=task_cfg,
                condition=condition,
                policy=args.policy,
                episode_idx=ep_idx,
                env_idx=env_idx,
                ablation_cfg=ablations,
            )
        except KeyboardInterrupt:
            print("\n\n  Campaign interrupted by user.")
            break

        append_csv(row)
        results.append(row)

        suc = "SUCCESS" if row["task_success"] else "FAIL"
        print(f"\n  Logged: ep{ep_idx+1}/{len(schedule)} | {condition} | {suc}")

        if ep_idx < len(schedule) - 1:
            cont = input("\n  Continue to next episode? [y/n]: ").strip().lower()
            if cont not in ("y", "yes"):
                print("  Stopping campaign.")
                break

    print_summary(results)


if __name__ == "__main__":
    main()
