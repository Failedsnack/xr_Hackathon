"""Command line: python -m vsum <video.mp4> <out.json>   (this is what run.sh calls)."""
from __future__ import annotations

import argparse
import sys
import traceback

from .pipeline import log, process_video, write_json


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vsum", description="Action-aware summary of one first-person video.")
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--expected-duration", type=float, default=None,
                    help="manifest duration; used for duration_sec when within 3 s of the probed value")
    ap.add_argument("--team-id", default=None)
    args = ap.parse_args(argv)
    info: dict = {}
    try:
        result = process_video(args.video, args.expected_duration, args.team_id, info)
    except Exception:
        traceback.print_exc()
        return 1
    write_json(result, args.out)
    log(f"[vsum] {result['video_id']}: label={info['label']} highlights={info['highlight_frac']:.1%} "
        f"actions={info['n_actions']} time={info['seconds']}s notes={info['notes'] or '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
