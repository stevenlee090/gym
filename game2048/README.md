# 2048 — N-Tuple Network + TD Learning

A superhuman 2048 agent trained entirely by self-play on the CPU. It does not use Gymnasium or stable-baselines3: deep RL methods like PPO and DQN do poorly on 2048, and the approach that beats humans is a large **n-tuple network** (a set of lookup tables) trained with **temporal-difference learning**, plus **expectimax search** at play time.

## How it works

| Piece | Details |
|---|---|
| Game engine ([`game.py`](game.py)) | The whole board is one 64-bit integer, 4 bits per cell. Every possible row is pre-solved into lookup tables, so a move is 4 table lookups. Compiled with numba. |
| Value function ([`ntuple.py`](ntuple.py)) | 4 tuples of 6 cells, each tuple applied to all 8 rotations and reflections of the board: 32 lookups into 4 tables × 16.7M float32 weights (~270 MB). |
| Learning | TD(0) on the board right after each move, before the new tile spawns. Updates run backwards over each finished game. 8 threads share one weight table with no locks ("Hogwild"). |
| Play | Expectimax: try every move, average over every cell where a 2 (90%) or 4 (10%) could spawn, and repeat to `--depth` levels. |

Training runs at about **5 million moves per second** on an M1 Pro. 1M games take about 10 minutes.

## Usage

```bash
cd game2048

# Train (1M games, ~10 min on M1 Pro)
uv run python train.py

# Evaluate with 2-ply expectimax; saves the best game to logs/eval_replay.json
uv run python evaluate.py --model models/<run_name>/best_weights.npy

# Stronger but ~20x slower search
uv run python evaluate.py --model models/<run_name>/best_weights.npy --depth 3 --games 20

# Build and open the replay page (replay.html)
uv run python visualise.py --open
```

`best_weights.npy` is picked by the greedy (no-search) evaluation that runs every 50k games. `final_weights.npy` is saved at the end.

## Monitoring

```bash
uv run tensorboard --logdir logs/tensorboard
```

`logs/<run_name>/progress.csv` holds per-batch training stats, and `eval.csv` holds the periodic greedy evaluations.

## Ideas to push further

- **More tuples:** 8 × 6-tuples roughly doubles the memory and adds several points to the 32768 rate.
- **Multi-stage networks:** a separate weight set once a 16384 tile is on the board (Yeh et al. 2016).
- **TC learning / optimistic initialisation:** adaptive per-weight learning rates, which are the state of the art for 2048.
- **Deeper search with a transposition table:** avoids re-evaluating identical boards at depth 3+.

## References

- Szubert & Jaśkowski, *Temporal Difference Learning of N-Tuple Networks for the Game 2048*, IEEE CIG 2014
- Yeh, Wu, Kao et al., *Multi-Stage Temporal Difference Learning for 2048-like Games*, IEEE TCIAIG 2016
