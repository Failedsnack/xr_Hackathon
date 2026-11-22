"""Verb-noun action detection.

Kitchen videos: zero-shot CLIP scores of "verb noun" prompts over 2 s windows, re-ranked with the
EPIC training-split pair prior. Everything else: coarse 'other'/'other' actions on the highlights
(the organisers' own FPVSum sample output uses the same convention).
"""
from __future__ import annotations

import numpy as np

from . import config as C
from .classes import load_pair_prior
from .label import softmax
from .prompts import pair_prompt


def other_actions(highlights: list[list[float]], imp: np.ndarray, fps: float = C.FPS) -> list[dict]:
    out = []
    z = (imp - imp.mean()) / (imp.std() + 1e-8) if len(imp) else imp
    for s, e in highlights:
        lo, hi = int(s * fps), max(int(s * fps) + 1, int(e * fps))
        m = float(z[lo:hi].mean()) if len(z) else 0.0
        out.append({"segment": [s, e], "verb": "other", "noun": "other",
                    "score": round(0.5 + 0.4 * float(np.tanh(m / 2.0)), 3)})
    return out


def kitchen_actions(emb: np.ndarray, clip, duration: float) -> list[dict] | None:
    """Return up to ACT_MAX actions, or None when the pair prior is unavailable."""
    prior = load_pair_prior()
    if prior is None or len(emb) == 0:
        return None
    pairs, logp = prior
    text = clip.encode_texts([pair_prompt(v, n) for v, n in pairs])        # [P, d]
    step = C.CLIP_EVERY / C.FPS                                            # seconds between CLIP frames
    m, w = len(emb), min(C.ACT_WIN, len(emb))
    starts = list(range(0, max(1, m - w + 1), C.ACT_STRIDE))
    win = np.stack([emb[i:i + w].mean(0) for i in starts])
    win /= np.linalg.norm(win, axis=1, keepdims=True) + 1e-8
    probs = softmax(C.ACT_LOGIT_SCALE * (win @ text.T) + C.ACT_PRIOR_WEIGHT * logp[None, :], axis=1)

    open_runs: dict[int, list] = {}      # pair index -> [start, end, best score]
    done: list[tuple[int, float, float, float]] = []
    for wi, i in enumerate(starts):
        s, e = i * step, min(duration, (i + w) * step)
        if e <= s:
            continue
        top = np.argsort(-probs[wi])[:C.ACT_TOPK]
        for p in top:
            run = open_runs.get(int(p))
            if run is not None and s <= run[1] + C.ACT_MERGE_GAP:
                run[1] = max(run[1], e)
                run[2] = max(run[2], float(probs[wi, p]))
            else:
                if run is not None:
                    done.append((int(p), *run))
                open_runs[int(p)] = [s, e, float(probs[wi, p])]
        for p in [q for q in open_runs if q not in set(int(x) for x in top)]:
            done.append((p, *open_runs.pop(p)))
    done += [(p, *run) for p, run in open_runs.items()]

    done.sort(key=lambda r: -r[3])
    done = done[:C.ACT_MAX]
    done.sort(key=lambda r: (r[1], -r[3]))
    return [{"segment": [round(s, 2), round(e, 2)], "verb": pairs[p][0], "noun": pairs[p][1],
             "score": round(min(1.0, max(0.0, sc)), 3)} for p, s, e, sc in done]
