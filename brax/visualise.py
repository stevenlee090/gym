"""
Build an HTML dashboard for a Playground PPO run: the joystick demo video synced to
commanded-vs-actual velocity, a checkpoint picker to watch the gait develop, and the
training curves.

The page references the demo videos by relative path (videos/*.mp4), so open it from
this folder. Safe to run mid-training: it reads whatever has been logged so far.

Usage:
    python visualise.py --run models/ppo_Go1JoystickFlatTerrain_seed0
    python visualise.py --run models/<run_name> --open
"""

import argparse
import glob
import json
import os
import webbrowser

import numpy as np

import config

HERE = os.path.dirname(os.path.abspath(__file__))


def load_reward_terms(run_name: str) -> dict:
    """Per-term eval reward at the first and last evaluation, from TensorBoard."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(os.path.join(config.LOG_DIR, "tensorboard", run_name),
                          size_guidance={"scalars": 0})
    ea.Reload()
    prefix = "eval/episode_reward/"
    tags = [t for t in ea.Tags()["scalars"] if t.startswith(prefix)]
    terms = {"names": [], "first": [], "last": [], "first_step": 0, "last_step": 0}
    for tag in sorted(tags):
        events = [e for e in ea.Scalars(tag) if e.step > 0]
        if not events:
            continue
        terms["names"].append(tag[len(prefix):])
        terms["first"].append(round(events[0].value, 3))
        terms["last"].append(round(events[-1].value, 3))
        terms["first_step"], terms["last_step"] = events[0].step, events[-1].step
    return terms


def load_demo(path: str) -> dict:
    with open(path) as f:
        d = json.load(f)
    tr = d["trace"]
    cmd = np.array([p["cmd"] for p in tr])
    act = np.array([[p["vx"], p["vy"], p["wz"]] for p in tr])
    fell = next((p["t"] for p in tr if p["fell"]), None)
    # RMS error over the whole demo: linear velocity (m/s) and yaw rate (rad/s).
    lin_err = float(np.sqrt(np.mean(np.sum((cmd[:, :2] - act[:, :2]) ** 2, axis=1))))
    yaw_err = float(np.sqrt(np.mean((cmd[:, 2] - act[:, 2]) ** 2)))
    video = os.path.splitext(path)[0] + ".mp4"
    return {
        "tag": os.path.basename(os.path.splitext(path)[0]).split("_", 1)[1],
        "env_steps": d["env_steps"],
        "reward_mean": d.get("reward_mean"),
        "reward_std": d.get("reward_std"),
        "survival": d.get("survival"),
        "video": os.path.relpath(video, HERE),
        "dt": d["dt"],
        "segments": d["segments"],
        "fell_at": fell,
        "lin_err": round(lin_err, 3),
        "yaw_err": round(yaw_err, 3),
        "trace": {
            "cmd": [[round(float(v), 2) for v in c] for c in cmd],
            "vx": [round(float(v), 3) for v in act[:, 0]],
            "vy": [round(float(v), 3) for v in act[:, 1]],
            "wz": [round(float(v), 3) for v in act[:, 2]],
        },
    }


def build(args):
    run_name = os.path.basename(os.path.normpath(args.run))
    with open(os.path.join(args.run, "metrics.json")) as f:
        m = json.load(f)

    demo_paths = glob.glob(os.path.join(HERE, config.VIDEO_DIR, f"{m['env']}_*.json"))
    demos = sorted((load_demo(p) for p in demo_paths
                    if os.path.exists(os.path.splitext(p)[0] + ".mp4")),
                   key=lambda d: d["env_steps"])
    if args.skip_tags:
        demos = [d for d in demos if d["tag"] not in args.skip_tags]

    try:
        import subprocess
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip().splitlines()[0]
    except Exception:
        gpu = m["device"]

    data = {
        "env": m["env"],
        "run_name": run_name,
        "gpu": gpu.replace("NVIDIA ", ""),
        "num_envs": m["num_envs"],
        "num_timesteps": m["num_timesteps"],
        "dr": m["domain_randomization"],
        "elapsed_s": m["elapsed_s"],
        "jit_time_s": m["jit_time_s"],
        "eval": [[h["step"], round(h["reward"], 3), round(h["reward_std"], 3), h["wall_time_s"]]
                 for h in m["history"]],
        "train": [[h["step"], h["sps"], round(h["reward"], 5), h["wall_time_s"]]
                  for h in m["train_history"]],
        "terms": load_reward_terms(run_name),
        "demos": demos,
    }

    with open(os.path.join(HERE, "dashboard_template.html")) as f:
        html = f.read()
    # "</" cannot appear inside a <script> block.
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = html.replace("/*__DATA__*/", payload)
    with open(args.out, "w") as f:
        f.write(html)
    print(f"Wrote {args.out} ({os.path.getsize(args.out) / 1e6:.2f} MB, "
          f"{len(demos)} demo(s): {', '.join(d['tag'] for d in demos)})")
    if args.open:
        webbrowser.open("file://" + os.path.abspath(args.out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the Go1 training dashboard")
    parser.add_argument("--run", default=os.path.join(
        config.MODEL_DIR, f"ppo_{config.ENV_NAME}_seed{config.SEED}"))
    parser.add_argument("--out", default=os.path.join(HERE, "dashboard.html"))
    parser.add_argument("--skip-tags", nargs="*", default=["smoke"],
                        help="Demo tags to leave out")
    parser.add_argument("--open", action="store_true", help="Open the page in a browser")
    args = parser.parse_args()
    build(args)
