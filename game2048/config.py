"""
Experiment configuration for 2048 n-tuple network TD learning.
"""

# Training
TOTAL_GAMES = 1_000_000
BATCH_GAMES = 2_000         # Games per training batch (one log line / TB point)
N_THREADS = 8               # Hogwild threads sharing one weight table
SEED = 42
MAX_MOVES = 200_000         # Per-game move buffer (a 65536 game is ~30k moves)

# Learning-rate schedule: (start_game, alpha). Each TD error is split
# across the 8 symmetric lookups of each tuple.
ALPHA_SCHEDULE = [
    (0, 0.1),
    (500_000, 0.03),
    (800_000, 0.01),
]

# Periodic evaluation (greedy policy, same as the training policy)
EVAL_FREQ = 50_000          # Evaluate every N training games
N_EVAL_GAMES = 400

# Final evaluation with expectimax search (evaluate.py defaults)
SEARCH_DEPTH = 2            # Extra (spawn, move) plies of lookahead
N_FINAL_EVAL_GAMES = 100

# Paths
LOG_DIR = "logs"
MODEL_DIR = "models"
MODEL_NAME = "ntuple_2048"
