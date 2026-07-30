#!/usr/bin/env python3
"""Log real-robot experiment trials and render a live results table.

One row per episode. Schema captures the task, the variant/condition
(C0-C8 for the language ablation, or `scene_graph`), the exact instruction
that was sent, grasp/task success, and the media folder.

Usage:
    # log a result
    python scripts/exp_log.py log \
        --task cube_to_drawer --condition C0 --context-mode standard \
        --env 1 --instruction "Pick up the black cube..." \
        --grasp 1 --success 0 --failure-mode grasp_fail \
        --media-dir experiments/results/media/cube_to_drawer/C0/trial_...

    # regenerate + print the live table
    python scripts/exp_log.py table

    # show last N rows
    python scripts/exp_log.py show --n 20
"""

import argparse
import csv
import os
from collections import defaultdict
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "experiments", "results")
TRIALS_CSV = os.path.join(RESULTS_DIR, "trials.csv")
LIVE_TABLE_MD = os.path.join(RESULTS_DIR, "live_table.md")

CONDITIONS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "scene_graph"]
COND_NAME = {
    "C0": "Full", "C1": "Empty", "C2": "Garbage", "C3": "Shuffled", "C4": "Cross-task",
    "C5": "Action-only", "C6": "Objects-only", "C7": "Wrong-object", "C8": "Paraphrase",
    "scene_graph": "SceneGraph",
}
FAILURE_MODES = ["approach_fail", "grasp_fail", "transport_fail", "placement_fail", "wrong_object", "no_motion", "none"]
HEADER = [
    "timestamp", "task", "condition", "context_mode", "env", "instruction",
    "grasp_success", "task_success", "ttc_s", "failure_mode", "media_dir", "notes",
]


def _ensure():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not os.path.exists(TRIALS_CSV):
        with open(TRIALS_CSV, "w", newline="") as f:
            csv.writer(f).writerow(HEADER)


def _load():
    if not os.path.exists(TRIALS_CSV):
        return []
    with open(TRIALS_CSV) as f:
        return list(csv.DictReader(f))


def log(args):
    _ensure()
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "task": args.task,
        "condition": args.condition,
        "context_mode": args.context_mode,
        "env": args.env,
        "instruction": (args.instruction or "").replace("\n", "\\n"),
        "grasp_success": int(args.grasp),
        "task_success": int(args.success),
        "ttc_s": args.ttc if args.success else "nan",
        "failure_mode": args.failure_mode if not args.success else "none",
        "media_dir": args.media_dir or "",
        "notes": (args.notes or "").replace("\n", " "),
    }
    with open(TRIALS_CSV, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=HEADER).writerow(row)
    status = "SUCCESS" if args.success else f"FAIL({row['failure_mode']})"
    print(f"[logged] {args.task} | {args.condition} ({COND_NAME.get(args.condition,'')}) | env{args.env} | "
          f"{status} | grasp={'Y' if args.grasp else 'N'}")
    render_table(print_it=True)


def _agg(rows):
    n = defaultdict(lambda: defaultdict(int))
    s = defaultdict(lambda: defaultdict(int))
    g = defaultdict(lambda: defaultdict(int))
    for r in rows:
        t, c = r["task"], r["condition"]
        n[t][c] += 1
        s[t][c] += int(r["task_success"])
        g[t][c] += int(r["grasp_success"])
    return n, s, g


def render_table(print_it=False):
    _ensure()
    rows = _load()
    n, s, g = _agg(rows)
    tasks = sorted(n)
    lines = ["# Live results table", "",
             f"_Updated {datetime.now().isoformat(timespec='seconds')} — total episodes: {len(rows)}_", ""]
    for t in tasks:
        lines += [f"## {t}", "",
                  "| Cond | Name | N | Grasp SR | Task SR |",
                  "|------|------|---|----------|---------|"]
        for c in CONDITIONS:
            cnt = n[t][c]
            if cnt == 0:
                continue
            gsr = g[t][c] / cnt
            sr = s[t][c] / cnt
            lines.append(f"| {c} | {COND_NAME.get(c,'')} | {cnt} | {gsr:.0%} | {sr:.0%} |")
        lines.append("")
    md = "\n".join(lines)
    with open(LIVE_TABLE_MD, "w") as f:
        f.write(md + "\n")
    if print_it:
        print("\n" + md)
    return md


def show(args):
    rows = _load()[-args.n:]
    print(f"\n--- last {len(rows)} trials ---")
    for r in rows:
        st = "OK " if r["task_success"] == "1" else "FAIL"
        print(f"  {r['timestamp']}  {r['task']:<16} {r['condition']:<11} env{r['env']:<3} {st}  fm={r['failure_mode']}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    lg = sub.add_parser("log")
    lg.add_argument("--task", required=True)
    lg.add_argument("--condition", required=True, choices=CONDITIONS)
    lg.add_argument("--context-mode", default="standard", choices=["standard", "scene_graph"])
    lg.add_argument("--env", type=int, required=True, help="Env-setup index (1..10)")
    lg.add_argument("--instruction", default="")
    lg.add_argument("--grasp", type=int, required=True, choices=[0, 1])
    lg.add_argument("--success", type=int, required=True, choices=[0, 1])
    lg.add_argument("--ttc", type=float, default=float("nan"))
    lg.add_argument("--failure-mode", default="none", choices=FAILURE_MODES)
    lg.add_argument("--media-dir", default="")
    lg.add_argument("--notes", default="")
    lg.set_defaults(func=log)

    tb = sub.add_parser("table")
    tb.set_defaults(func=lambda a: render_table(print_it=True))

    sh = sub.add_parser("show")
    sh.add_argument("--n", type=int, default=20)
    sh.set_defaults(func=show)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
