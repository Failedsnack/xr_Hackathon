"""Official class dictionary and the EPIC verb-noun prior."""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

from . import config as C

_FALLBACK_LABELS = ["bike", "bikepolo", "boxing", "cooking", "horse", "jumping", "longboarding", "motor",
                    "parkour", "plane", "rockclimbing", "scuba", "skate", "ski", "surfing", "other"]


@lru_cache(maxsize=1)
def load_classes() -> dict[str, list[str]]:
    """{'labels': [...], 'verbs': [...], 'nouns': [...]} read from data/supported_classes.json."""
    path = C.DATA_DIR / "supported_classes.json"
    if not path.exists():
        return {"labels": list(_FALLBACK_LABELS), "verbs": ["other"], "nouns": ["other"]}
    raw = json.loads(path.read_text())
    return {k: list(raw[k]) for k in ("labels", "verbs", "nouns")}


@lru_cache(maxsize=1)
def load_pair_prior():
    """([(verb, noun), ...], log prior [P]) from weights/epic_pair_prior.json, or None if missing."""
    path = C.PRIOR_PATH
    if not path.exists():
        return None
    doc = json.loads(path.read_text())
    cls = load_classes()
    ok_v, ok_n = set(cls["verbs"]), set(cls["nouns"])
    rows = [(v, n, c) for v, n, c in doc["pairs"] if v in ok_v and n in ok_n]
    if not rows:
        return None
    counts = np.array([c for _, _, c in rows], dtype=np.float64)
    logp = np.log((counts + 1.0) / (counts.sum() + len(rows)))
    return [(v, n) for v, n, _ in rows], logp
