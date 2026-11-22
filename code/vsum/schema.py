"""Output schema: a validator that mirrors the rules in the organisers' README, and a finalizer
that repairs a result so that every file we write is valid."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from . import config as C
from .select import merge_segments

REQUIRED = ("team_id", "video_id", "duration_sec", "label", "actions", "highlights")


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate(res: dict, classes: dict | None = None, manifest_duration: float | None = None,
             filename_stem: str | None = None) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Any error makes the file invalid (scores 0 for the video)."""
    err: list[str] = []
    warn: list[str] = []
    for k in REQUIRED:
        if k not in res:
            err.append(f"missing field '{k}'")
    if err:
        return err, warn

    if not (isinstance(res["team_id"], str) and res["team_id"].strip()):
        err.append("team_id must be a non-empty string")
    if not (isinstance(res["video_id"], str) and res["video_id"].strip()):
        err.append("video_id must be a non-empty string")
    elif filename_stem is not None and res["video_id"] != filename_stem:
        err.append(f"video_id '{res['video_id']}' does not match file name '{filename_stem}'")

    d = res["duration_sec"]
    if not (_num(d) and d > 0):
        err.append("duration_sec must be a positive number")
        return err, warn
    if manifest_duration is not None and abs(d - manifest_duration) > 0.5:
        err.append(f"duration_sec {d} differs from manifest {manifest_duration} by more than 0.5 s")

    lab = res["label"]
    if not (isinstance(lab, dict) and isinstance(lab.get("name"), str) and _num(lab.get("score"))
            and 0 <= lab["score"] <= 1):
        err.append("label must be {'name': str, 'score': number in [0, 1]}")
    elif classes is not None and lab["name"] not in classes["labels"]:
        err.append(f"label '{lab['name']}' is not in supported_classes.json")

    acts = res["actions"]
    if not isinstance(acts, list) or len(acts) > C.ACT_MAX:
        err.append(f"actions must be a list of at most {C.ACT_MAX} items")
    else:
        for i, a in enumerate(acts):
            try:
                s, e = a["segment"]
                ok = (_num(s) and _num(e) and 0 <= s < e <= d + 0.5 and _num(a["score"])
                      and 0 <= a["score"] <= 1 and isinstance(a["verb"], str) and isinstance(a["noun"], str))
            except Exception:
                ok = False
            if not ok:
                err.append(f"actions[{i}] is malformed or out of range")
                break
            if classes is not None and (a["verb"] not in classes["verbs"] or a["noun"] not in classes["nouns"]):
                err.append(f"actions[{i}] uses a verb/noun that is not in supported_classes.json")
                break

    hl = res["highlights"]
    if not isinstance(hl, list) or not hl:
        err.append("highlights must contain at least one item")
        return err, warn
    segs = []
    for i, h in enumerate(hl):
        try:
            s, e = h["segment"]
            if not (_num(s) and _num(e) and 0 <= s < e <= d):
                raise ValueError
        except Exception:
            err.append(f"highlights[{i}] is malformed or outside the video")
            return err, warn
        segs.append((s, e))
    for i in range(1, len(segs)):
        if segs[i][0] <= segs[i - 1][1]:
            err.append(f"highlights[{i}] overlaps or is not chronologically sorted")
            break
    if max(e - s for s, e in segs) < C.MIN_SEG - 1e-9:
        err.append(f"no highlight is at least {C.MIN_SEG} s long")
    total = sum(e - s for s, e in segs)
    if total > C.HARD_BUDGET_FRAC * d + 1e-9:
        warn.append(f"highlights use {total / d:.1%} of the video (> 15%: soft penalty on Action Coverage)")
    return err, warn


def validate_file(path: Path, classes: dict | None = None, manifest: dict | None = None):
    try:
        res = json.loads(Path(path).read_text())
    except Exception as e:
        return [f"not valid JSON: {e}"], []
    md = (manifest or {}).get(Path(path).stem)
    if manifest is not None and md is None:
        return [f"{Path(path).stem} is not in manifest.csv"], []
    return validate(res, classes, md, Path(path).stem)


def load_manifest(path: Path) -> dict[str, float]:
    with open(path, newline="") as f:
        return {r["video_id"]: float(r["duration_sec"]) for r in csv.DictReader(f)}


def finalize(res: dict, classes: dict) -> dict:
    """Repair a result in place so that it passes validate(): clamp, sort, merge, enforce class names."""
    d = float(res["duration_sec"])

    name = res["label"]["name"]
    res["label"] = {"name": name if name in classes["labels"] else "other",
                    "score": round(min(1.0, max(0.0, float(res["label"]["score"]))), 3)}

    verbs, nouns = set(classes["verbs"]), set(classes["nouns"])
    acts = []
    for a in res.get("actions", []):
        s, e = float(a["segment"][0]), float(a["segment"][1])
        s, e = max(0.0, s), min(d + 0.5, e)
        if e - s < 0.05:
            continue
        acts.append({"segment": [round(s, 2), round(e, 2)],
                     "verb": a["verb"] if a["verb"] in verbs else "other",
                     "noun": a["noun"] if a["noun"] in nouns else "other",
                     "score": round(min(1.0, max(0.0, float(a["score"]))), 3)})
    acts = [a for a in acts if a["segment"][0] < a["segment"][1]]
    acts.sort(key=lambda a: -a["score"])
    acts = sorted(acts[:C.ACT_MAX], key=lambda a: (a["segment"][0], -a["score"]))
    res["actions"] = acts

    segs = []
    for h in res.get("highlights", []):
        s, e = max(0.0, float(h["segment"][0])), min(d, float(h["segment"][1]))
        if e - s >= 0.05:
            segs.append([round(s, 2), round(e, 2)])
    if not segs:
        segs = [[0.0, round(min(d, C.MIN_SEG), 2)]]
    segs = merge_segments(segs)
    if max(e - s for s, e in segs) < C.MIN_SEG:
        # stretch the longest segment to the minimum length (inside the video), then re-merge
        i = max(range(len(segs)), key=lambda k: segs[k][1] - segs[k][0])
        s, e = segs[i]
        e = min(d, s + C.MIN_SEG)
        s = max(0.0, e - C.MIN_SEG)
        segs[i] = [round(s, 2), round(e, 2)]
        segs = merge_segments(segs)
    res["highlights"] = [{"segment": s} for s in segs]
    return res
