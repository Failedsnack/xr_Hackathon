"""End-to-end: video file -> result dict that follows the submission schema."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

from . import config as C
from .actions import kitchen_actions, other_actions
from .classes import load_classes
from .clipmodel import get_clip
from .importance import importance_curve
from .label import predict_label
from .schema import finalize, validate
from .select import select_highlights
from .videoio import iter_frames, probe_duration

DURATION_TOLERANCE = 3.0   # seconds; when a manifest duration is supplied and agrees within this, it is used


def resolve_team_id(team_id: str | None = None) -> str:
    if team_id:
        return team_id
    if os.environ.get("VSUM_TEAM_ID"):
        return os.environ["VSUM_TEAM_ID"]
    if C.TEAM_ID_FILE.exists() and C.TEAM_ID_FILE.read_text().strip():
        return C.TEAM_ID_FILE.read_text().strip()
    return C.DEFAULT_TEAM_ID


def _features(path: str, clip):
    """One streaming pass: motion, brightness, colour histogram and (every 2nd frame) CLIP embeddings."""
    motion, bright, hists, embs = [], [], [], []
    notes: list[str] = []
    prev = None
    g = 0
    f = C.FRAME // C.LOWRES
    for chunk in iter_frames(path):
        k = len(chunk)
        low = chunk.reshape(k, C.LOWRES, f, C.LOWRES, f, 3).mean((2, 4), dtype=np.float32)   # [k, 56, 56, 3]
        grey = low.mean(3)
        ref = grey if prev is None else np.concatenate([prev[None], grey])
        d = np.abs(ref[1:] - ref[:-1]).mean((1, 2)) if len(ref) > 1 else np.zeros(0, np.float32)
        m = np.zeros(k, np.float32)
        if prev is None:
            m[1:] = d
            m[0] = d[0] if len(d) else 0.0
        else:
            m[:] = d
        motion.append(m)
        bright.append(grey.mean((1, 2)))
        prev = grey[-1]
        q = (low // 32).astype(np.int64)
        code = q[..., 0] * 64 + q[..., 1] * 8 + q[..., 2]
        hists.append(np.stack([np.bincount(c.ravel(), minlength=512) for c in code]).astype(np.float32))
        if clip is not None:
            sel = [i for i in range(k) if (g + i) % C.CLIP_EVERY == 0]
            if sel:
                try:
                    embs.append(clip.encode_images(chunk[sel]))
                except Exception as e:  # lose CLIP, keep going with colour/motion features
                    notes.append(f"clip_encode_failed:{type(e).__name__}")
                    clip, embs = None, []
        g += k
    if g == 0:
        raise RuntimeError("no frames decoded")
    emb = np.concatenate(embs) if embs and clip is not None else None
    return np.concatenate(motion), np.concatenate(bright), np.concatenate(hists), emb, clip, notes


def process_video(path: str, expected_duration: float | None = None, team_id: str | None = None,
                  info: dict | None = None) -> dict:
    """Summarise one video. `expected_duration` (from manifest.csv) only snaps duration_sec when it agrees."""
    t0 = time.time()
    info = info if info is not None else {}
    notes: list[str] = []
    classes = load_classes()

    probed = probe_duration(path)
    duration = probed
    if expected_duration is not None:
        if abs(probed - expected_duration) <= DURATION_TOLERANCE:
            duration = expected_duration
        else:
            notes.append(f"duration_mismatch:{probed:.2f}_vs_{expected_duration:.2f}")
    duration = round(float(duration), 2)

    clip = get_clip()
    if clip is None:
        notes.append("clip_unavailable")
    motion, bright, hist, emb, clip, more = _features(str(path), clip)
    notes += more
    if clip is None and "clip_unavailable" not in notes:
        notes.append("clip_lost_midway")

    if emb is not None:
        label, label_score = predict_label(emb, clip, classes)
    else:
        label, label_score = "other", 0.1
    domain = "kitchen" if label == "cooking" else "sport"

    imp, desc = importance_curve(motion, bright, hist, emb, clip, domain)
    highlights = select_highlights(imp, desc, duration)

    actions = None
    if emb is not None and label == "cooking":
        try:
            actions = kitchen_actions(emb, clip, duration)
        except Exception as e:
            notes.append(f"actions_failed:{type(e).__name__}")
    if actions is None and label != "cooking":
        actions = other_actions(highlights, imp)
    if actions is None:
        actions = []

    result = {
        "team_id": resolve_team_id(team_id),
        "video_id": Path(path).stem,
        "duration_sec": duration,
        "label": {"name": label, "score": label_score},
        "actions": actions,
        "highlights": [{"segment": s} for s in highlights],
    }
    result = finalize(result, classes)
    errors, warnings = validate(result, classes)
    if errors:
        raise RuntimeError("internal error, invalid output: " + "; ".join(errors))

    total = sum(h["segment"][1] - h["segment"][0] for h in result["highlights"])
    info.update(seconds=round(time.time() - t0, 1), frames=len(motion), notes=notes, warnings=warnings,
                label=label, n_actions=len(result["actions"]), highlight_frac=round(total / duration, 4))
    return result


def write_json(result: dict, out: str) -> None:
    import json
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text(json.dumps(result, indent=2))
    os.replace(tmp, out_path)


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)
