"""
Bitboard 2048 engine compiled with numba.

The board is a single int64: cell (r, c) lives in the 4 bits at offset
4 * (4r + c) and holds log2 of the tile value (0 = empty, 1 = 2, 2 = 4, ...).
Moves are table lookups: every possible 16-bit row is pre-solved once at import.

Actions: 0 = up, 1 = right, 2 = down, 3 = left.
"""

import numpy as np
from numba import njit

ACTION_NAMES = ("up", "right", "down", "left")
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3


def _s64(x: int) -> int:
    """Reinterpret an unsigned 64-bit literal as a signed int64."""
    return x - (1 << 64) if x >= (1 << 63) else x


def _reverse_row(row: int) -> int:
    return (
        ((row & 0x000F) << 12)
        | ((row & 0x00F0) << 4)
        | ((row & 0x0F00) >> 4)
        | ((row & 0xF000) >> 12)
    )


def _unpack_col(row: int) -> int:
    """Spread a 16-bit row into column 0 of a board (nibble j -> row j)."""
    return (row | (row << 12) | (row << 24) | (row << 36)) & 0x000F000F000F000F


def _slide_left(row: int) -> tuple[int, int]:
    """Slide one row toward its low nibble. Returns (new_row, merge_score)."""
    tiles = [(row >> (4 * i)) & 0xF for i in range(4)]
    tiles = [t for t in tiles if t]
    out, score, i = [], 0, 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1] and tiles[i] < 15:
            out.append(tiles[i] + 1)
            score += 1 << (tiles[i] + 1)
            i += 2
        else:
            out.append(tiles[i])
            i += 1
    out += [0] * (4 - len(out))
    return sum(t << (4 * i) for i, t in enumerate(out)), score


def _build_tables():
    # XOR deltas: board ^= table[row] << shift applies a move to one line.
    tables = np.zeros((4, 65536), dtype=np.int64)
    scores = np.zeros((4, 65536), dtype=np.int64)
    for row in range(65536):
        left, score = _slide_left(row)
        rev_row = _reverse_row(row)
        rev_left = _reverse_row(left)
        tables[LEFT, row] = row ^ left
        tables[RIGHT, rev_row] = rev_row ^ rev_left
        tables[UP, row] = _unpack_col(row) ^ _unpack_col(left)
        tables[DOWN, rev_row] = _unpack_col(rev_row) ^ _unpack_col(rev_left)
        scores[LEFT, row] = score
        scores[UP, row] = score
        scores[RIGHT, rev_row] = score
        scores[DOWN, rev_row] = score
    return tables, scores


TABLES, SCORES = _build_tables()

_T1 = _s64(0xF0F00F0FF0F00F0F)
_T2 = 0x0000F0F00000F0F0
_T3 = 0x0F0F00000F0F0000
_T4 = _s64(0xFF00FF0000FF00FF)
_T5 = 0x00FF00FF00000000
_T6 = 0x00000000FF00FF00


@njit
def transpose(b):
    a = (b & _T1) | ((b & _T2) << 12) | ((b & _T3) >> 12)
    return (a & _T4) | ((a & _T5) >> 24) | ((a & _T6) << 24)


@njit
def move(b, action):
    """Apply a move. Returns (new_board, reward); new_board == b if illegal."""
    out = b
    reward = 0
    if action == UP or action == DOWN:
        t = transpose(b)
        for i in range(4):
            row = (t >> (16 * i)) & 0xFFFF
            out ^= TABLES[action, row] << (4 * i)
            reward += SCORES[action, row]
    else:
        for i in range(4):
            row = (b >> (16 * i)) & 0xFFFF
            out ^= TABLES[action, row] << (16 * i)
            reward += SCORES[action, row]
    return out, reward


@njit
def count_empty(b):
    n = 0
    for i in range(16):
        if (b >> (4 * i)) & 0xF == 0:
            n += 1
    return n


@njit
def spawn(b):
    """Place a 2 (90%) or 4 (10%) on a uniformly random empty cell."""
    n = count_empty(b)
    if n == 0:
        return b
    k = np.random.randint(n)
    tile = 1 if np.random.random() < 0.9 else 2
    for i in range(16):
        if (b >> (4 * i)) & 0xF == 0:
            if k == 0:
                return b | (np.int64(tile) << (4 * i))
            k -= 1
    return b


@njit
def new_game():
    return spawn(spawn(np.int64(0)))


@njit
def is_over(b):
    for a in range(4):
        nb, _ = move(b, a)
        if nb != b:
            return False
    return True


@njit
def max_tile(b):
    m = 0
    for i in range(16):
        v = (b >> (4 * i)) & 0xF
        m = max(m, v)
    return m


@njit
def seed(s):
    np.random.seed(s)


def to_grid(b: int) -> np.ndarray:
    """Board -> 4x4 array of tile values (0, 2, 4, 8, ...)."""
    b = int(b) & ((1 << 64) - 1)
    exps = np.array([(b >> (4 * i)) & 0xF for i in range(16)]).reshape(4, 4)
    return np.where(exps > 0, 1 << exps, 0)


def from_grid(grid) -> int:
    """4x4 tile values -> board."""
    b = 0
    for i, v in enumerate(np.asarray(grid).ravel()):
        if v:
            b |= int(v).bit_length() - 1 << (4 * i)
    return _s64(b)
