"""
Evaluate trained 2048 weights with expectimax search and save a replay of
the best game for visualise.py.

Usage:
    python evaluate.py --model models/<run_name>/best_weights.npy
    python evaluate.py --model models/<run_name>/best_weights.npy --depth 3 --games 20
    python evaluate.py --model models/<run_name>/best_weights.npy --depth 0   # greedy
"""

import argparse
import json
import os
import time

import config
import numpy as np
from game import move
from ntuple import build_iso, play_games
from numba import set_num_threads

MAX_RECORD = 60_000  # Moves recorded per game (a 32768 game is ~16k moves)


def board_hex(b: int) -> str:
    return f"{int(b) & ((1 << 64) - 1):016x}"


def build_replay(boards: np.ndarray, actions: np.ndarray, n: int) -> dict:
    rewards = [int(move(boards[i], actions[i])[1]) for i in range(n)]
    return {
        "boards": [board_hex(b) for b in boards[: n + 1]],
        "actions": "".join(str(a) for a in actions[:n]),
        "rewards": rewards,
    }


def evaluate(args):
    set_num_threads(args.threads)
    print(f"\nLoading weights: {args.model}")
    w = np.load(args.model)
    iso = build_iso()

    boards = np.zeros((args.games, MAX_RECORD), dtype=np.int64)
    actions = np.zeros((args.games, MAX_RECORD), dtype=np.int64)

    print(f"Playing {args.games} games with expectimax depth {args.depth} "
          f"on {args.threads} threads...\n")
    t0 = time.time()
    scores, tiles, moves = play_games(w, iso, args.depth, args.games, boards, actions)
    dt = time.time() - t0

    for g in range(args.games):
        print(f"  Game {g + 1:3d}: score={scores[g]:8,d}  max tile={1 << tiles[g]:6d}  moves={moves[g]:,}")

    rates = {1 << m: float((tiles >= m).mean()) for m in range(9, 17)}
    print(f"\n{'='*44}")
    print(f"  Games:          {args.games}  (depth {args.depth}, {dt:.0f}s)")
    print(f"  Mean score:     {scores.mean():,.0f} ± {scores.std():,.0f}")
    print(f"  Median / Max:   {np.median(scores):,.0f} / {scores.max():,}")
    print(f"  Speed:          {moves.sum() / dt:,.0f} moves/s")
    print("  Reached tile:")
    for tile, r in rates.items():
        if r > 0:
            print(f"    {tile:>6}: {r:6.1%}")
    print(f"{'='*44}\n")

    best = int(np.argmax(scores))
    out = {
        "model": args.model,
        "depth": args.depth,
        "games": args.games,
        "scores": scores.tolist(),
        "max_tiles": [1 << int(t) for t in tiles],
        "moves": moves.tolist(),
        "rates": {str(k): v for k, v in rates.items()},
        "best_game": {
            "index": best,
            "score": int(scores[best]),
            "max_tile": 1 << int(tiles[best]),
            **build_replay(boards[best], actions[best], int(moves[best])),
        },
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"Best game (score {scores[best]:,}, tile {1 << tiles[best]}) saved to: {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a 2048 n-tuple network")
    parser.add_argument("--model", type=str, required=True, help="Path to weights .npy")
    parser.add_argument("--games", type=int, default=config.N_FINAL_EVAL_GAMES)
    parser.add_argument("--depth", type=int, default=config.SEARCH_DEPTH,
                        help="Expectimax lookahead (0 = greedy training policy)")
    parser.add_argument("--threads", type=int, default=config.N_THREADS)
    parser.add_argument("--out", type=str, default=os.path.join(config.LOG_DIR, "eval_replay.json"))
    args = parser.parse_args()
    evaluate(args)
