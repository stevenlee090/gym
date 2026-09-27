"""
Build a self-contained HTML page that replays the best evaluation game and
charts how training went.

Usage:
    python visualise.py                       # uses defaults below
    python visualise.py --replay logs/eval_replay.json --run ntuple_2048_seed42 --open
"""

import argparse
import csv
import json
import os
import webbrowser

import config
from ntuple import TUPLES

HERE = os.path.dirname(os.path.abspath(__file__))


def load_progress(path: str) -> tuple[list[list[float]], float]:
    rows, mps = [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append([
                int(r["games"]),
                round(float(r["mean_score"])),
                float(r["rate_2048"]),
                float(r["rate_8192"]),
                float(r["rate_16384"]),
            ])
            mps.append(float(r["moves_per_sec"]))
    return rows, sorted(mps)[len(mps) // 2] / 1e6


def build(args):
    with open(args.replay) as f:
        ev = json.load(f)
    progress, mps = load_progress(os.path.join(config.LOG_DIR, args.run, "progress.csv"))

    data = {
        "replay": ev["best_game"],
        "eval": {k: ev[k] for k in ("depth", "games", "scores", "rates")},
        "progress": progress,
        "total_games": progress[-1][0],
        "alpha_schedule": config.ALPHA_SCHEDULE,
        "network": f"{len(TUPLES)} tuples of {len(TUPLES[0])} cells",
        "moves_per_sec": f"{mps:.1f}",
    }
    with open(os.path.join(HERE, "replay_template.html")) as f:
        html = f.read()
    # "</" cannot appear inside a <script> block.
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = html.replace("/*__DATA__*/", payload)
    with open(args.out, "w") as f:
        f.write(html)
    print(f"Wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")
    if args.open:
        webbrowser.open("file://" + os.path.abspath(args.out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the 2048 replay page")
    parser.add_argument("--replay", default=os.path.join(config.LOG_DIR, "eval_replay.json"))
    parser.add_argument("--run", default=f"{config.MODEL_NAME}_seed{config.SEED}")
    parser.add_argument("--out", default="replay.html")
    parser.add_argument("--open", action="store_true", help="Open the page in a browser")
    args = parser.parse_args()
    build(args)
