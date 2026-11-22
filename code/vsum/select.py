"""Redundancy-aware, budgeted highlight selection."""
from __future__ import annotations

import numpy as np

from . import config as C
from .importance import zscore


def _window_mean(x: np.ndarray, length: int) -> np.ndarray:
    """Mean over every window of `length` consecutive rows (works for 1-D and 2-D input)."""
    c = np.concatenate([np.zeros((1,) + x.shape[1:], dtype=np.float64), np.cumsum(x, axis=0, dtype=np.float64)])
    return (c[length:] - c[:-length]) / length


def merge_segments(segs: list[list[float]]) -> list[list[float]]:
    """Sort and merge segments that overlap or touch, so the result is strictly non-overlapping."""
    out: list[list[float]] = []
    for a, b in sorted(segs):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def select_highlights(imp: np.ndarray, desc: np.ndarray, duration: float, fps: float = C.FPS,
                      seg_len: float = C.SEG_LEN, budget_frac: float = C.BUDGET_FRAC) -> list[list[float]]:
    """Greedy pick of fixed-length windows: high importance, dissimilar to and away from earlier picks.

    Stops at budget_frac * duration (default 14%, under the 15% rule). Returns merged [start, end]
    pairs in seconds, rounded to 2 decimals.
    """
    n = len(imp)
    length = int(round(seg_len * fps))
    if n < length:
        return [[0.0, round(float(duration), 2)]]

    starts = np.arange(0, n - length + 1)
    score = zscore(_window_mean(imp, length))
    emb = _window_mean(desc.astype(np.float64), length)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8

    k_max = max(1, int(budget_frac * duration // seg_len))
    sigma = max(1.0, n / (2.0 * k_max))
    max_sim = np.zeros(len(starts))
    crowd = np.zeros(len(starts))
    blocked = np.zeros(len(starts), dtype=bool)
    picked: list[int] = []

    for _ in range(k_max):
        value = score - C.LAMBDA_SIM * max_sim - C.LAMBDA_TIME * crowd
        value[blocked] = -np.inf
        j = int(np.argmax(value))
        if not np.isfinite(value[j]):
            break
        picked.append(j)
        max_sim = np.maximum(max_sim, emb @ emb[j])
        crowd = np.maximum(crowd, np.exp(-np.abs(starts - starts[j]) / sigma))
        blocked |= np.abs(starts - starts[j]) <= length  # no overlap, and a 1-frame gap

    segs = [[float(starts[j]) / fps, min(float(duration), (float(starts[j]) + length) / fps)] for j in picked]
    return [[round(a, 2), round(b, 2)] for a, b in merge_segments(segs)]
