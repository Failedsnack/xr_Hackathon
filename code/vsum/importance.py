"""Per-frame importance: motion, local novelty, global uniqueness and (with CLIP) hand/action interaction."""
from __future__ import annotations

import numpy as np

from . import config as C
from .prompts import interaction_prompts


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    s = x.std()
    return (x - x.mean()) / s if s > 1e-8 else np.zeros_like(x)


def smooth(x: np.ndarray, k: int = 3) -> np.ndarray:
    if len(x) < k:
        return x
    pad = k // 2
    return np.convolve(np.pad(x, (pad, pad), mode="edge"), np.ones(k) / k, mode="valid")


def expand_clip(emb: np.ndarray, n: int, every: int = C.CLIP_EVERY) -> np.ndarray:
    """CLIP embeddings (every `every`-th frame) -> one row per sampled frame (nearest earlier)."""
    idx = np.minimum(np.arange(n) // every, len(emb) - 1)
    return emb[idx]


def descriptor(hist: np.ndarray, emb: np.ndarray | None) -> np.ndarray:
    """Unit-norm per-frame descriptor: CLIP embedding if available, else centred colour histogram."""
    n = len(hist)
    if emb is not None:
        d = expand_clip(emb, n).astype(np.float32)
    else:
        d = hist / (hist.sum(1, keepdims=True) + 1e-8)
        d = d - d.mean(0)
    return (d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-8)).astype(np.float32)


def novelty(desc: np.ndarray, long: int = C.NOVELTY_LONG, short: int = C.NOVELTY_SHORT) -> np.ndarray:
    """1 - cosine(frame, mean of the frames [t-long, t-short)): high where something new appears."""
    n, d = desc.shape
    prefix = np.vstack([np.zeros((1, d)), np.cumsum(desc.astype(np.float64), axis=0)])
    t = np.arange(n)
    a, b = np.maximum(0, t - long), np.maximum(0, t - short)
    valid = b > a
    mean = (prefix[b] - prefix[a]) / np.maximum(b - a, 1)[:, None]
    cos = (desc * mean).sum(1) / (np.linalg.norm(mean, axis=1) + 1e-8)
    out = np.where(valid, 1.0 - cos, 0.0)
    if valid.any():
        out[~valid] = np.median(out[valid])
    return out


def uniqueness(desc: np.ndarray, exclude: int = C.UNIQUE_EXCLUDE, k: int = C.UNIQUE_K) -> np.ndarray:
    """1 - mean similarity to the k most similar *distant* frames: low for repetitive footage."""
    n = len(desc)
    if n <= 2 * exclude + 1:
        return np.zeros(n)
    s = desc @ desc.T
    idx = np.arange(n)
    s[np.abs(idx[:, None] - idx[None, :]) < exclude] = -2.0
    k = min(k, n - 2 * exclude)
    top = np.partition(s, -k, axis=1)[:, -k:]
    return 1.0 - top.mean(1)


def _logsumexp(x: np.ndarray, axis: int) -> np.ndarray:
    m = x.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def interaction(emb: np.ndarray, clip, domain: str) -> np.ndarray:
    """Log-odds that a CLIP frame shows meaningful activity rather than idle footage (per CLIP frame)."""
    active, idle = interaction_prompts(domain)
    t = clip.encode_texts(active + idle)
    logits = clip.scale * (emb @ t.T)
    return _logsumexp(logits[:, :len(active)], 1) - _logsumexp(logits[:, len(active):], 1)


def importance_curve(motion, bright, hist, emb, clip, domain):
    """Return (importance [n], descriptors [n, d]); both aligned to the sampled-frame timeline."""
    n = len(motion)
    desc = descriptor(hist, emb)
    parts = {
        "motion": zscore(np.log1p(motion)),
        "novelty": zscore(novelty(desc)),
        "unique": zscore(uniqueness(desc)),
    }
    weights = C.W_NO_CLIP
    if emb is not None and clip is not None:
        parts["interact"] = zscore(expand_clip(interaction(emb, clip, domain), n))
        weights = C.W_CLIP
    imp = sum(weights[k] * parts[k] for k in weights)
    imp = np.asarray(imp, dtype=np.float64)
    imp[np.asarray(bright) < C.DARK_LEVEL] -= C.DARK_PENALTY
    return smooth(imp, 3), desc
