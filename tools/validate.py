#!/usr/bin/env python3
"""Validate output JSON files against the Track 04 rules (TEAM_README section 5).

    python tools/validate.py --dir outputs/                         # check every file in the folder
    python tools/validate.py --dir outputs/ --manifest data/manifest.csv
                                                                    # also check ids, durations, completeness

Exit status 0 = every file valid (and, with --manifest, all 187 present), 1 otherwise.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from vsum import config as C                                        # noqa: E402
from vsum.classes import load_classes                               # noqa: E402
from vsum.schema import load_manifest, validate_file                # noqa: E402
import json                                                         # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=ROOT / "outputs", help="folder with <video_id>.json files")
    ap.add_argument("--manifest", type=Path, default=None, help="manifest.csv (default: data/manifest.csv if present)")
    ap.add_argument("--no-manifest", action="store_true", help="skip the manifest checks")
    ap.add_argument("--allow-missing", action="store_true",
                    help="do not fail when some manifest videos have no output yet (used by CI while the batch is running)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    manifest = None
    if not args.no_manifest:
        mp = args.manifest or (C.DATA_DIR / "manifest.csv")
        if mp.exists():
            manifest = load_manifest(mp)
    classes = load_classes()

    files = sorted(args.dir.glob("*.json"))
    bad, warned, teams = 0, 0, set()
    for p in files:
        errors, warnings = validate_file(p, classes, manifest)
        try:
            teams.add(json.loads(p.read_text()).get("team_id"))
        except Exception:
            pass
        if errors:
            bad += 1
            print(f"INVALID  {p.name}: " + "; ".join(errors))
        elif warnings:
            warned += 1
            if not args.quiet:
                print(f"warning  {p.name}: " + "; ".join(warnings))

    problems = bad
    if len(teams) > 1:
        print(f"INVALID  team_id is not identical across files: {sorted(map(str, teams))}")
        problems += 1
    missing = []
    if manifest is not None:
        missing = sorted(set(manifest) - {p.stem for p in files})
        if missing:
            problems += 0 if args.allow_missing else 1
            print(f"MISSING  {len(missing)} of {len(manifest)} manifest videos have no output file "
                  f"(a missing file scores 0). First few: {', '.join(missing[:8])}")

    total = len(manifest) if manifest is not None else len(files)
    print(f"{len(files)} files checked: {len(files) - bad} valid, {bad} invalid, {warned} with budget warnings"
          + (f"; {total - len(missing)}/{total} manifest videos covered" if manifest is not None else ""))
    print("OK" if problems == 0 else "FAILED")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
