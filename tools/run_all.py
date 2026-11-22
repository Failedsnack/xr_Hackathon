#!/usr/bin/env python3
"""Run the whole manifest: download one video -> summarise it -> validate -> delete the video.

Designed for a laptop with limited disk and time:
  * EPIC-KITCHENS-100 is ~184 GB in total, so videos are never kept (only one to three are on disk at once);
  * downloads run in background threads while the previous video is being processed;
  * shortest videos go first and FPVSum goes before EPIC, so the number of valid output files grows as fast as
    possible if time runs out;
  * it is resumable: a valid existing output/<video_id>.json is skipped, so re-running after an interruption
    simply continues;
  * every output is validated, and every run is logged to logs/run_log.csv.

Examples
    python tools/run_all.py --team-id SRMIST_MyTeam --only fpvsum            # the 49 YouTube videos first
    python tools/run_all.py --team-id SRMIST_MyTeam                          # everything, resumable
    python tools/run_all.py --team-id SRMIST_MyTeam --video-dir ~/videos     # videos already on disk
    python tools/run_all.py --team-id SRMIST_MyTeam --commit-every 20        # also git commit + push progress
    python tools/run_all.py --dry-run                                        # show the plan only
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from vsum import config as C                                         # noqa: E402
from vsum.classes import load_classes                                # noqa: E402
from vsum.pipeline import process_video, resolve_team_id, write_json   # noqa: E402
from vsum.schema import load_manifest, validate_file                 # noqa: E402

VIDEO_EXTS = (".mp4", ".MP4", ".mkv", ".webm", ".mov")
EPIC_REPO = "https://github.com/epic-kitchens/epic-kitchens-download-scripts.git"
LOG_FIELDS = ["time", "video_id", "dataset", "status", "seconds", "label", "highlight_frac", "n_actions", "notes"]


# ----------------------------------------------------------------------------- downloading
def find_local(video_dir: Path, vid: str) -> Path | None:
    for ext in VIDEO_EXTS:
        for p in (video_dir / f"{vid}{ext}", video_dir / vid.split("_")[0] / "videos" / f"{vid}{ext}"):
            if p.exists():
                return p
    return None


def load_fpvsum_ids(path: Path) -> dict[str, str]:
    """video_name -> YouTube id, from the official FPVSum list (data/fpvsum_videolist.csv)."""
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {r["video_name"]: r["youtube_id"] for r in csv.DictReader(f)}


def youtube_id_for(vid: str, mapping: dict[str, str]) -> str:
    """The official list is authoritative: e.g. BikePolo_DHbYfmjX3WQ_1 is YouTube id hkNp3udSNV8, not DHbYfmjX3WQ.
    Fallback for ids missing from the list: <Category>_<YouTubeID>."""
    return mapping.get(vid) or vid.split("_", 1)[1]


def download_fpvsum(vid: str, work: Path, height: int, youtube_id: str) -> Path:
    """480p is plenty because frames are squashed to 224x224."""
    dest = work / "dl"
    dest.mkdir(parents=True, exist_ok=True)
    fmt = (f"bv*[height<={height}][vcodec^=avc1]/bv*[height<={height}]/b[height<={height}]/worst")
    cmd = [sys.executable, "-m", "yt_dlp", "-f", fmt, "--no-playlist", "--retries", "5", "--quiet", "--no-warnings",
           "-o", str(dest / f"{vid}.%(ext)s"), f"https://www.youtube.com/watch?v={youtube_id}"]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    hits = [p for p in dest.glob(f"{vid}.*") if p.suffix.lower() in {e.lower() for e in VIDEO_EXTS}]
    if not hits:
        raise RuntimeError("yt-dlp finished but no video file was found")
    return hits[0]


def ensure_epic_scripts(scripts: Path) -> None:
    if (scripts / "epic_downloader.py").exists():
        return
    scripts.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", EPIC_REPO, str(scripts)], check=True)


def download_epic(vid: str, work: Path, scripts: Path) -> Path:
    out = (work / "epic").resolve()
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "epic_downloader.py", "--videos", "--specific-videos", vid, "--output-path", str(out)]
    # the downloader reads data/*.csv relative to its own folder, so it must run from there
    subprocess.run(cmd, cwd=scripts, check=True, capture_output=True, text=True)
    # The official downloader writes <out>/EPIC-KITCHENS/<participant>/videos/<video>.MP4
    part = vid.split("_")[0]
    for path in (out / "EPIC-KITCHENS" / part / "videos" / f"{vid}.MP4", out / part / "videos" / f"{vid}.MP4"):
        if path.exists():
            return path
    hits = sorted(out.rglob(f"{vid}.MP4"))
    if hits:
        return hits[0]
    raise RuntimeError(f"downloader finished but no {vid}.MP4 was found under {out}")


# ----------------------------------------------------------------------------- bookkeeping
def log_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LOG_FIELDS})


def output_is_current(out: Path, vid: str, classes, manifest, team_id: str) -> bool:
    p = out / f"{vid}.json"
    if not p.exists():
        return False
    errors, _ = validate_file(p, classes, manifest)
    if errors:
        return False
    import json
    return json.loads(p.read_text()).get("team_id") == team_id


def git_commit_progress(out: Path, log: Path, done: int, total: int, added: int) -> None:
    try:
        subprocess.run(["git", "add", str(out), str(log)], cwd=ROOT, check=True)
        msg = f"Add outputs for {added} more videos ({done}/{total} of the manifest covered)"
        subprocess.run(["git", "commit", "-q", "-m", msg], cwd=ROOT, check=True)
        subprocess.run(["git", "push", "-q"], cwd=ROOT, check=True)
        print(f"  committed and pushed: {msg}", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"  (git commit/push skipped: {e})", flush=True)


def fmt_eta(seconds: float) -> str:
    h, m = int(seconds // 3600), int(seconds % 3600 // 60)
    return f"{h}h{m:02d}m" if h else f"{m}m"


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=C.DATA_DIR / "manifest.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs")
    ap.add_argument("--work", type=Path, default=ROOT / "work", help="scratch folder for downloads")
    ap.add_argument("--log", type=Path, default=ROOT / "logs" / "run_log.csv")
    ap.add_argument("--team-id", default=None, help="written into every JSON, e.g. SRMIST_MyTeam")
    ap.add_argument("--only", choices=["all", "fpvsum", "epic"], default="all")
    ap.add_argument("--order", choices=["short-first", "manifest"], default="short-first")
    ap.add_argument("--limit", type=int, default=None, help="process at most N videos this run")
    ap.add_argument("--video-dir", type=Path, default=None, help="use videos already on disk (no download)")
    ap.add_argument("--keep-videos", action="store_true", help="do not delete downloaded videos")
    ap.add_argument("--prefetch", type=int, default=2, help="videos downloaded ahead of processing")
    ap.add_argument("--yt-height", type=int, default=480)
    ap.add_argument("--epic-scripts", type=Path, default=ROOT / "third_party" / "epic-kitchens-download-scripts")
    ap.add_argument("--commit-every", type=int, default=0, help="git commit + push after every N new outputs")
    ap.add_argument("--force", action="store_true", help="re-process videos that already have a valid output")
    ap.add_argument("--allow-no-clip", action="store_true",
                    help="continue even if CLIP cannot be loaded (labels become 'other', actions empty)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    team_id = resolve_team_id(args.team_id)
    if team_id == C.DEFAULT_TEAM_ID and not args.dry_run:
        print(f"ERROR: team id is still the placeholder '{team_id}'. Pass --team-id SRMIST_<TeamName> "
              f"(or write it into team_id.txt): it is stored in every output file.", file=sys.stderr)
        return 2

    manifest = load_manifest(args.manifest)
    classes = load_classes()
    with open(args.manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    if args.only != "all":
        want = "fpvsum" if args.only == "fpvsum" else "epic_kitchens_100"
        rows = [r for r in rows if r["dataset"] == want]
    if args.order == "short-first":
        rows.sort(key=lambda r: (r["dataset"] != "fpvsum", float(r["duration_sec"])))

    args.out.mkdir(parents=True, exist_ok=True)
    todo = [r for r in rows if args.force or not output_is_current(args.out, r["video_id"], classes, manifest, team_id)]
    if args.limit:
        todo = todo[:args.limit]
    covered = len(manifest) - len([r for r in load_manifest(args.manifest) if not
                                   output_is_current(args.out, r, classes, manifest, team_id)])
    hours = sum(float(r["duration_sec"]) for r in todo) / 3600
    print(f"team_id={team_id}  videos in manifest={len(manifest)}  already done={covered}  "
          f"to do now={len(todo)} ({hours:.1f} h of footage)", flush=True)
    if args.dry_run:
        for r in todo[:20]:
            print(f"  {r['video_id']:<28} {r['dataset']:<18} {float(r['duration_sec']):8.1f}s")
        if len(todo) > 20:
            print(f"  ... and {len(todo) - 20} more")
        return 0
    if not todo:
        print("nothing to do")
        return 0

    # ---- preflight: fail now, not after hours of downloading
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            print(f"ERROR: {tool} is not installed (macOS: brew install ffmpeg)", file=sys.stderr)
            return 2
    if args.video_dir is None and any(r["dataset"] == "fpvsum" for r in todo):
        try:
            import yt_dlp  # noqa: F401
        except ImportError:
            print("ERROR: yt-dlp is missing (pip install -r requirements-tools.txt)", file=sys.stderr)
            return 2
    from vsum.clipmodel import get_clip
    if get_clip() is None and not args.allow_no_clip:
        print("ERROR: CLIP could not be loaded, so labels would be 'other' and actions empty for every video.\n"
              "Fix the install / network (python -m vsum.warmup shows the error) or pass --allow-no-clip.",
              file=sys.stderr)
        return 2
    if any(r["dataset"] != "fpvsum" for r in todo) and args.video_dir is None:
        ensure_epic_scripts(args.epic_scripts)

    fpv_ids = load_fpvsum_ids(C.DATA_DIR / "fpvsum_videolist.csv")

    def fetch(row) -> Path:
        vid = row["video_id"]
        if args.video_dir is not None:
            p = find_local(args.video_dir, vid)
            if p is None:
                raise FileNotFoundError(f"{vid} not found in {args.video_dir}")
            return p
        if row["dataset"] == "fpvsum":
            return download_fpvsum(vid, args.work, args.yt_height, youtube_id_for(vid, fpv_ids))
        return download_epic(vid, args.work, args.epic_scripts)

    done_now, failed, added_since_commit = 0, [], 0
    started = time.time()
    pool = ThreadPoolExecutor(max_workers=max(1, args.prefetch))
    queue: deque = deque()
    it = iter(todo)

    def refill():
        while len(queue) < max(1, args.prefetch):
            row = next(it, None)
            if row is None:
                return
            queue.append((row, pool.submit(fetch, row)))

    refill()
    try:
        while queue:
            row, fut = queue.popleft()
            refill()
            vid = row["video_id"]
            expected = float(row["duration_sec"])
            rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "video_id": vid, "dataset": row["dataset"]}
            path = None
            try:
                path = fut.result()
                info: dict = {}
                result = process_video(str(path), expected_duration=expected, team_id=team_id, info=info)
                write_json(result, str(args.out / f"{vid}.json"))
                errors, _ = validate_file(args.out / f"{vid}.json", classes, manifest)
                if errors:
                    (args.out / f"{vid}.json").unlink(missing_ok=True)   # never leave an invalid file behind
                    raise RuntimeError("; ".join(errors))
                rec.update(status="ok", seconds=info["seconds"], label=info["label"],
                           highlight_frac=info["highlight_frac"], n_actions=info["n_actions"],
                           notes=" ".join(info["notes"]))
                done_now += 1
                added_since_commit += 1
                eta = (time.time() - started) / done_now * (len(todo) - done_now - len(failed))
                print(f"[{done_now + len(failed)}/{len(todo)}] {vid:<24} ok  {info['seconds']:>6.1f}s  "
                      f"label={info['label']:<12} highlights={info['highlight_frac']:.1%} "
                      f"actions={info['n_actions']:<4} {('NOTES: ' + ' '.join(info['notes'])) if info['notes'] else ''}"
                      f" ETA {fmt_eta(eta)}", flush=True)
            except Exception as e:
                failed.append(vid)
                msg = (getattr(e, "stderr", "") or str(e)).strip().splitlines()
                rec.update(status="failed", notes=(msg[-1] if msg else type(e).__name__)[:200])
                print(f"[{done_now + len(failed)}/{len(todo)}] {vid:<24} FAILED: {rec['notes']}", flush=True)
            finally:
                if path is not None and not args.keep_videos and args.video_dir is None:
                    try:
                        path.unlink()
                    except OSError:
                        pass
            log_row(args.log, rec)
            if args.commit_every and added_since_commit >= args.commit_every:
                git_commit_progress(args.out, args.log, covered + done_now, len(manifest), added_since_commit)
                added_since_commit = 0
    except KeyboardInterrupt:
        print("\ninterrupted: re-run the same command to continue where you stopped", flush=True)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        if not args.keep_videos and args.video_dir is None and args.work.exists():
            shutil.rmtree(args.work, ignore_errors=True)

    if args.commit_every and added_since_commit:
        git_commit_progress(args.out, args.log, covered + done_now, len(manifest), added_since_commit)
    print(f"\nfinished: {done_now} new outputs, {len(failed)} failed")
    if failed:
        print("failed (re-run to retry): " + ", ".join(failed))
    print("next: python tools/validate.py --dir outputs/ --manifest data/manifest.csv")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
