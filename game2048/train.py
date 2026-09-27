"""
Train an n-tuple network to play 2048 with TD(0) afterstate learning.

Usage:
    python train.py
    python train.py --games 1000000 --threads 8
    python train.py --resume models/ntuple_2048_seed42/final_weights.npy --start-game 1000000

Note: training is Hogwild (threads share weights without locks), so runs are
not bit-for-bit reproducible even with a fixed seed.
"""

import argparse
import csv
import json
import os
import time

import config
import numpy as np
from game import seed as seed_numba
from ntuple import TUPLES, build_iso, learn_games, new_weights, play_games
from numba import set_num_threads
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

MILESTONES = (11, 12, 13, 14, 15)  # 2048, 4096, 8192, 16384, 32768


def alpha_at(game: int) -> float:
    alpha = config.ALPHA_SCHEDULE[0][1]
    for start, a in config.ALPHA_SCHEDULE:
        if game >= start:
            alpha = a
    return alpha


def tile_rates(tiles: np.ndarray) -> dict[int, float]:
    return {1 << m: float((tiles >= m).mean()) for m in MILESTONES}


def save(w: np.ndarray, path: str, meta: dict):
    np.save(path, w)
    with open(os.path.splitext(path)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=2)


def evaluate(w, iso, n_games: int):
    empty = np.zeros((n_games, 0), dtype=np.int64)
    scores, tiles, _ = play_games(w, iso, 0, n_games, empty, empty)
    return scores, tiles


def train(args):
    set_num_threads(args.threads)
    seed_numba(args.seed)
    np.random.seed(args.seed)

    run_name = f"{config.MODEL_NAME}_seed{args.seed}"
    run_dir = os.path.join(config.MODEL_DIR, run_name)
    log_dir = os.path.join(config.LOG_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print("  2048 N-Tuple TD Learning")
    print(f"  run:     {run_name}")
    print(f"  network: {len(TUPLES)} x {len(TUPLES[0])}-tuples x 8 symmetries")
    print(f"  games:   {args.start_game:,} -> {args.games:,}")
    print(f"  threads: {args.threads}")
    print(f"{'='*50}\n")

    iso = build_iso()
    if args.resume:
        print(f"Resuming from: {args.resume}")
        w = np.load(args.resume)
    else:
        w = new_weights()

    buf_afters = np.zeros((args.threads, config.MAX_MOVES), dtype=np.int64)
    buf_rewards = np.zeros((args.threads, config.MAX_MOVES), dtype=np.int64)

    writer = SummaryWriter(os.path.join(config.LOG_DIR, "tensorboard", run_name))
    progress_path = os.path.join(log_dir, "progress.csv")
    eval_path = os.path.join(log_dir, "eval.csv")
    append = args.resume is not None and os.path.exists(progress_path)
    progress_f = open(progress_path, "a" if append else "w", newline="")  # noqa: SIM115 (kept open for the whole run)
    eval_f = open(eval_path, "a" if append else "w", newline="")  # noqa: SIM115
    progress_csv = csv.writer(progress_f)
    eval_csv = csv.writer(eval_f)
    rate_cols = [f"rate_{1 << m}" for m in MILESTONES]
    if not append:
        progress_csv.writerow(["games", "alpha", "mean_score", "max_score", "max_tile",
                               "mean_moves", "moves_per_sec", *rate_cols])
        eval_csv.writerow(["games", "mean_score", "std_score", "max_tile", *rate_cols])

    best_eval = -1.0
    game = args.start_game
    start = time.time()
    pbar = tqdm(total=args.games, initial=game, unit="game", dynamic_ncols=True)

    while game < args.games:
        n = min(config.BATCH_GAMES, args.games - game)
        alpha = alpha_at(game)
        t0 = time.time()
        scores, tiles, moves = learn_games(w, iso, alpha, n, args.threads, buf_afters, buf_rewards)
        dt = time.time() - t0
        game += n

        rates = tile_rates(tiles)
        mps = moves.sum() / dt
        progress_csv.writerow([game, alpha, f"{scores.mean():.1f}", int(scores.max()),
                               1 << int(tiles.max()), f"{moves.mean():.1f}", f"{mps:.0f}",
                               *(f"{r:.4f}" for r in rates.values())])
        progress_f.flush()
        writer.add_scalar("train/mean_score", scores.mean(), game)
        writer.add_scalar("train/alpha", alpha, game)
        writer.add_scalar("train/moves_per_sec", mps, game)
        for tile, r in rates.items():
            writer.add_scalar(f"train/rate_{tile}", r, game)

        pbar.update(n)
        pbar.set_postfix(score=f"{scores.mean():,.0f}", r2048=f"{rates[2048]:.0%}",
                         r8192=f"{rates[8192]:.0%}", r16k=f"{rates[16384]:.1%}",
                         Mmps=f"{mps / 1e6:.1f}")

        if game % config.EVAL_FREQ == 0 or game >= args.games:
            e_scores, e_tiles = evaluate(w, iso, config.N_EVAL_GAMES)
            e_rates = tile_rates(e_tiles)
            eval_csv.writerow([game, f"{e_scores.mean():.1f}", f"{e_scores.std():.1f}",
                               1 << int(e_tiles.max()), *(f"{r:.4f}" for r in e_rates.values())])
            eval_f.flush()
            writer.add_scalar("eval/mean_score", e_scores.mean(), game)
            for tile, r in e_rates.items():
                writer.add_scalar(f"eval/rate_{tile}", r, game)
            tqdm.write(
                f"[eval @ {game:,}] mean {e_scores.mean():,.0f} | "
                + " ".join(f"{t}:{r:.1%}" for t, r in e_rates.items())
            )
            meta = {"games": game, "tuples": TUPLES, "eval_mean_score": float(e_scores.mean()),
                    "eval_rates": {str(k): v for k, v in e_rates.items()}}
            if e_scores.mean() > best_eval:
                best_eval = float(e_scores.mean())
                save(w, os.path.join(run_dir, "best_weights.npy"), meta)
                tqdm.write(f"  new best -> {run_dir}/best_weights.npy")

    pbar.close()
    final_path = os.path.join(run_dir, "final_weights.npy")
    save(w, final_path, {"games": game, "tuples": TUPLES})
    writer.close()
    progress_f.close()
    eval_f.close()
    print(f"\nTraining complete in {(time.time() - start) / 60:.1f} min. "
          f"Final weights saved to: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an n-tuple network on 2048")
    parser.add_argument("--games", type=int, default=config.TOTAL_GAMES,
                        help="Train until this many total games")
    parser.add_argument("--threads", type=int, default=config.N_THREADS)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a weights .npy to continue from")
    parser.add_argument("--start-game", type=int, default=0,
                        help="Game counter to resume from (drives the alpha schedule)")
    args = parser.parse_args()
    train(args)
