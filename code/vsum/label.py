"""Video-level category by zero-shot CLIP."""
from __future__ import annotations

import numpy as np

from .prompts import label_prompts


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def predict_label(emb: np.ndarray, clip, classes: dict) -> tuple[str, float]:
    """Average the per-frame class probabilities over the video; 'other' is never predicted."""
    cand = [lab for lab in classes["labels"] if lab != "other"]
    protos = []
    for lab in cand:
        t = clip.encode_texts(label_prompts(lab)).mean(0)
        protos.append(t / (np.linalg.norm(t) + 1e-8))
    probs = softmax(clip.scale * (emb @ np.stack(protos).T), axis=1).mean(0)
    i = int(np.argmax(probs))
    return cand[i], float(probs[i])
