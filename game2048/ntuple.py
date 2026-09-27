"""
N-tuple network value function, TD(0) afterstate learning, and expectimax play.

The network scores an afterstate (the board right after a move, before the
random tile spawns) as the sum of lookup-table weights: each 6-cell tuple is
read as a 24-bit index, and every tuple is applied to all 8 rotations and
reflections of the board (4 tuples x 8 symmetries = 32 lookups).

References:
    Szubert & Jaśkowski, "Temporal Difference Learning of N-Tuple Networks
        for the Game 2048" (2014)
    Yeh et al., "Multi-Stage Temporal Difference Learning for 2048-like
        Games" (2016)
"""

import numpy as np
from game import count_empty, max_tile, move, new_game, spawn
from numba import njit, prange

# Board cells are numbered row-major: 0..3 is the top row.
TUPLES = (
    (0, 1, 2, 3, 4, 5),
    (4, 5, 6, 7, 8, 9),
    (0, 1, 2, 4, 5, 6),
    (4, 5, 6, 8, 9, 10),
)
N_SYM = 8
TUPLE_LEN = 6
TABLE_SIZE = 16**TUPLE_LEN


def _symmetries() -> list[list[int]]:
    """For each of the 8 board symmetries, map cell index -> transformed cell."""
    grid = np.arange(16).reshape(4, 4)
    syms = []
    for flip in (False, True):
        g = grid.T if flip else grid
        for k in range(4):
            syms.append(np.rot90(g, k).ravel().tolist())
    return syms


def build_iso() -> np.ndarray:
    """ISO[t, s, k] = board cell read by position k of tuple t under symmetry s."""
    syms = _symmetries()
    iso = np.zeros((len(TUPLES), N_SYM, TUPLE_LEN), dtype=np.int64)
    for t, tup in enumerate(TUPLES):
        for s, sym in enumerate(syms):
            iso[t, s] = [sym[c] for c in tup]
    return iso


def new_weights() -> np.ndarray:
    return np.zeros((len(TUPLES), TABLE_SIZE), dtype=np.float32)


@njit(inline="always")
def _index(b, iso, t, s):
    idx = 0
    for k in range(iso.shape[2]):
        idx |= ((b >> (4 * iso[t, s, k])) & 0xF) << (4 * k)
    return idx


@njit
def value(w, iso, b):
    v = 0.0
    for t in range(iso.shape[0]):
        for s in range(iso.shape[1]):
            v += w[t, _index(b, iso, t, s)]
    return v


@njit
def update(w, iso, b, delta):
    for t in range(iso.shape[0]):
        for s in range(iso.shape[1]):
            w[t, _index(b, iso, t, s)] += delta


@njit
def greedy_action(w, iso, b):
    """Pick argmax_a [r + V(afterstate)]. Returns (action, afterstate, reward)."""
    best_a, best_v = -1, -1e30
    best_after, best_r = b, 0
    for a in range(4):
        after, r = move(b, a)
        if after == b:
            continue
        v = r + value(w, iso, after)
        if v > best_v:
            best_a, best_v, best_after, best_r = a, v, after, r
    return best_a, best_after, best_r


@njit
def _learn_one_game(w, iso, alpha, afters, rewards):
    """Play one greedy game, then apply TD(0) updates backwards over it."""
    b = new_game()
    n = 0
    score = 0
    while n < afters.shape[0]:
        a, after, r = greedy_action(w, iso, b)
        if a < 0:
            break
        afters[n] = after
        rewards[n] = r
        n += 1
        score += r
        b = spawn(after)

    # Backward pass: V(after_t) <- r_{t+1} + V(after_{t+1}); terminal target is 0.
    step = alpha / iso.shape[1]
    target = 0.0
    for t in range(n - 1, -1, -1):
        err = target - value(w, iso, afters[t])
        update(w, iso, afters[t], step * err)
        target = rewards[t] + value(w, iso, afters[t])
    return score, max_tile(b), n


@njit(parallel=True)
def learn_games(w, iso, alpha, n_games, n_threads, buf_afters, buf_rewards):
    """Hogwild training: threads play games concurrently and share weights."""
    scores = np.zeros(n_games, dtype=np.int64)
    tiles = np.zeros(n_games, dtype=np.int64)
    moves = np.zeros(n_games, dtype=np.int64)
    per = (n_games + n_threads - 1) // n_threads
    for th in prange(n_threads):
        for g in range(th * per, min((th + 1) * per, n_games)):
            s, m, n = _learn_one_game(w, iso, alpha, buf_afters[th], buf_rewards[th])
            scores[g] = s
            tiles[g] = m
            moves[g] = n
    return scores, tiles, moves


# ---------------------------------------------------------------------------
# Expectimax search
# ---------------------------------------------------------------------------

@njit
def _chance(w, iso, after, depth):
    """Expected value of an afterstate over the random tile spawn."""
    if depth <= 0:
        return value(w, iso, after)
    n = count_empty(after)
    if n == 0:
        return value(w, iso, after)
    total = 0.0
    for i in range(16):
        if (after >> (4 * i)) & 0xF == 0:
            total += 0.9 * _max_node(w, iso, after | (np.int64(1) << (4 * i)), depth - 1)
            total += 0.1 * _max_node(w, iso, after | (np.int64(2) << (4 * i)), depth - 1)
    return total / n


@njit
def _max_node(w, iso, b, depth):
    best = 0.0  # terminal state has no future reward
    found = False
    for a in range(4):
        after, r = move(b, a)
        if after == b:
            continue
        v = r + _chance(w, iso, after, depth)
        if not found or v > best:
            best, found = v, True
    return best


@njit
def search_action(w, iso, b, depth):
    """
    Choose a move with expectimax. depth=0 is the plain greedy policy used
    during training; each extra level looks one more (spawn, move) pair ahead.
    """
    best_a, best_v = -1, -1e30
    for a in range(4):
        after, r = move(b, a)
        if after == b:
            continue
        v = r + _chance(w, iso, after, depth)
        if v > best_v:
            best_a, best_v = a, v
    return best_a


@njit
def play_game(w, iso, depth, boards, actions):
    """
    Play one game with search. If boards is non-empty, records the board before
    each move and the action taken, plus the final board (so len(boards) must
    exceed the move count; recording stops the game when the buffer is full).
    """
    record = boards.shape[0] > 0
    b = new_game()
    n = 0
    score = 0
    while True:
        if record and n >= boards.shape[0] - 1:
            break
        a = search_action(w, iso, b, depth)
        if a < 0:
            break
        if record:
            boards[n] = b
            actions[n] = a
        after, r = move(b, a)
        score += r
        n += 1
        b = spawn(after)
    if record:
        boards[n] = b
    return score, max_tile(b), n


@njit(parallel=True)
def play_games(w, iso, depth, n_games, boards, actions):
    """
    Play n_games in parallel. boards/actions are (n_games, max_moves) record
    buffers, or (n_games, 0) to skip recording.
    """
    scores = np.zeros(n_games, dtype=np.int64)
    tiles = np.zeros(n_games, dtype=np.int64)
    moves = np.zeros(n_games, dtype=np.int64)
    for g in prange(n_games):
        s, m, n = play_game(w, iso, depth, boards[g], actions[g])
        scores[g] = s
        tiles[g] = m
        moves[g] = n
    return scores, tiles, moves
