#!/usr/bin/env python3
"""Validate output JSONs: python tools/validate.py outputs/ [manifest.csv]"""
import csv, json, sys
from pathlib import Path
d = Path(sys.argv[1]); man = {}
if len(sys.argv) > 2:
    man = {r["video_id"]: float(r["duration_sec"]) for r in csv.DictReader(open(sys.argv[2]))}
bad, teams = 0, set()
for p in sorted(d.glob("*.json")):
    e = []; j = json.loads(p.read_text())
    for k in ["team_id","video_id","duration_sec","label","actions","highlights"]:
        if k not in j: e.append(f"missing {k}")
    if e: print(p.name, e); bad += 1; continue
    teams.add(j["team_id"]); D = j["duration_sec"]
    if j["video_id"] != p.stem: e.append("video_id != filename")
    if man and (p.stem not in man or abs(man[p.stem]-D) > 0.5): e.append("duration/manifest mismatch")
    if not 0 <= j["label"]["score"] <= 1: e.append("label score")
    for a in j["actions"]:
        s, t = a["segment"]
        if not (0 <= s < t <= D+0.5 and 0 <= a["score"] <= 1): e.append(f"bad action {a}")
    H = [h["segment"] for h in j["highlights"]]
    if not H: e.append("no highlights")
    for i,(s,t) in enumerate(H):
        if not (0 <= s < t <= D+0.5): e.append(f"bad hl {s,t}")
        if i and s <= H[i-1][1]: e.append("hl overlap/unsorted")
    if H and max(t-s for s,t in H) < 4.0: e.append("no hl >= 4s")
    tot = sum(t-s for s,t in H)
    if tot > 0.15*D: print(p.name, f"warning: highlights {tot/D:.1%} > 15%")
    if e: print(p.name, e); bad += 1
if len(teams) > 1: print("team_id differs across files"); bad += 1
missing = set(man) - {p.stem for p in d.glob("*.json")}
if missing: print(f"{len(missing)} manifest videos missing output")
print("OK" if not bad and not missing else f"{bad} invalid files")
