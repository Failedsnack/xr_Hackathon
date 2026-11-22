#!/usr/bin/env python3
"""Build weights/epic_pair_prior.json from the public EPIC-KITCHENS-100 *training* annotations.

The prior is simply how often each (verb, noun) pair occurs in the training split. It is used
to re-rank zero-shot CLIP verb-noun scores at inference time (see code/vsum/actions.py).
No validation-split annotation is read: the script asserts that none of the 138 manifest
videos appears in the training file.

    python tools/build_pair_prior.py                    # downloads the three CSVs
    python tools/build_pair_prior.py --csv-dir ./csvs   # or reuse local copies
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/master/"
FILES = ("EPIC_100_train.csv", "EPIC_100_verb_classes.csv", "EPIC_100_noun_classes.csv")


def fetch(name: str, csv_dir: Path | None) -> Path:
    if csv_dir is not None and (csv_dir / name).exists():
        return csv_dir / name
    dest = (csv_dir or Path("/tmp")) / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {name} ...", file=sys.stderr)
    urllib.request.urlretrieve(BASE + name, dest)
    return dest


def class_keys(path: Path) -> dict[int, str]:
    with open(path, newline="") as f:
        return {int(r["id"]): r["key"] for r in csv.DictReader(f)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-dir", type=Path, default=None)
    ap.add_argument("--classes", type=Path, default=ROOT / "data" / "supported_classes.json")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest.csv")
    ap.add_argument("--top", type=int, default=2000, help="number of most frequent pairs to keep")
    ap.add_argument("--out", type=Path, default=ROOT / "weights" / "epic_pair_prior.json")
    args = ap.parse_args()

    train = fetch("EPIC_100_train.csv", args.csv_dir)
    verbs = class_keys(fetch("EPIC_100_verb_classes.csv", args.csv_dir))
    nouns = class_keys(fetch("EPIC_100_noun_classes.csv", args.csv_dir))

    classes = json.loads(args.classes.read_text())
    ok_v, ok_n = set(classes["verbs"]), set(classes["nouns"])
    manifest_ids = {r["video_id"] for r in csv.DictReader(open(args.manifest, newline=""))}

    counts: Counter = Counter()
    train_videos = set()
    with open(train, newline="") as f:
        for r in csv.DictReader(f):
            train_videos.add(r["video_id"])
            v, n = verbs[int(r["verb_class"])], nouns[int(r["noun_class"])]
            if v in ok_v and n in ok_n:
                counts[(v, n)] += 1

    overlap = train_videos & manifest_ids
    if overlap:
        print(f"ERROR: {len(overlap)} manifest videos appear in the training annotations: {sorted(overlap)[:5]}",
              file=sys.stderr)
        return 1

    top = counts.most_common(args.top)
    total = sum(c for _, c in top)
    doc = {
        "source": "EPIC-KITCHENS-100 public training-split annotations (EPIC_100_train.csv)",
        "note": "Pair frequencies only. Disjoint from the 138 validation videos in manifest.csv (asserted).",
        "n_pairs": len(top),
        "total": total,
        "pairs": [[v, n, c] for (v, n), c in top],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, separators=(",", ":")))
    print(f"wrote {args.out} ({len(top)} pairs from {sum(counts.values())} annotations, "
          f"{len(train_videos)} training videos, overlap with manifest: 0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
