# Action-Aware Video Summarization

**Samsung PRISM XR Hackathon 2026 · Track 04 (XR Perception Group)** · Amritanshu Aditya, SRM Institute of Science and Technology

![ci](https://github.com/Failedsnack/xr_Hackhaton/actions/workflows/ci.yml/badge.svg)

Given a first-person video (2–10 minutes), the system picks the important, non-repetitive moments and returns one JSON per video with
a **highlight reel of about 14% of the footage** (hard limit 15%), the **video category**, and **verb–noun actions** with timestamps.
Highlights always reference the original footage by timestamp; nothing is generated.

| | |
|---|---|
| Demo video (≤ 3 min) | [`Video .mp4`](Video%20.mp4) in this repository |
| Presentation | the `.pptx` in the repository root |
| Method in one line | ffmpeg → motion / novelty / uniqueness / CLIP "activity" cues → redundancy-aware greedy selection under a time budget |
| Models | OpenCLIP ViT-B/32 (LAION-2B), used zero-shot. **No training or fine-tuning was done.** |

---

## Quick start

**Requirements:** Python 3.10+ (developed on 3.13), `ffmpeg` and `ffprobe` on the PATH (`brew install ffmpeg` / `apt install ffmpeg`).

```bash
./tools/setup.sh                       # venv + dependencies + CLIP weights (downloaded once, ~600 MB)
source .venv/bin/activate
./run.sh <video.mp4> <out.json>        # the contract: one video in, one JSON out
python tools/validate.py --dir outputs/ --manifest data/manifest.csv
```

**Docker** (no local setup; weights are baked into the image, so running needs no network):

```bash
docker build -t xr-vsum .
docker run --rm -v "$PWD/videos":/in:ro -v "$PWD/out":/out xr-vsum /in/<video>.mp4 /out/<video>.json
```

### Reproducing `outputs/` (all 187 videos)

The raw videos are not distributed, so `tools/run_all.py` fetches them one at a time (EPIC-KITCHENS-100 with the official downloader,
FPVSum with `yt-dlp`), runs the same code path as `run.sh`, validates the result and deletes the video:

```bash
python tools/run_all.py --team-id SRMIST_<TeamName>          # resumable; shortest videos first
python tools/run_all.py --team-id SRMIST_<TeamName> --video-dir ~/videos   # if the videos are already on disk
```

Every run is logged in [`logs/run_log.csv`](logs) (status, runtime, label, highlight share, notes per video).

---

## Output format

One `<video_id>.json` per video, exactly the schema in [`docs/TEAM_README.md`](docs/TEAM_README.md):

```json
{ "team_id": "...", "video_id": "P01_11", "duration_sec": 561.53,
  "label": {"name": "cooking", "score": 0.93},
  "actions": [{"segment": [12.0, 14.0], "verb": "open", "noun": "fridge", "score": 0.08}],
  "highlights": [{"segment": [40.5, 44.5]}, {"segment": [101.0, 105.0]}] }
```

`tools/validate.py` implements every rule of the organisers' checklist (six fields, id and duration vs `manifest.csv` within ±0.5 s,
scores in [0, 1], class names from `supported_classes.json`, sorted strictly non-overlapping highlights, at least one ≥ 4 s,
≤ 15% budget, 187 files). In the code, `vsum.schema.finalize()` repairs every result before it is written, so the pipeline cannot emit a file that the validator rejects.

---

## How it works

```
video ──ffmpeg 2 fps, 224²──┬─ motion energy ─────────────┐
                            ├─ colour histogram           │
                            └─ CLIP image embedding (1 fps)┤
                                                           ▼
      label (zero-shot CLIP) ─────────────►  importance(t) = weighted z-scores of
                                              motion · local novelty · global uniqueness · activity cue
                                                           ▼
                        greedy pick of 4 s windows under a 14% budget
                        (penalty for looking like, or sitting next to, earlier picks)
                                                           ▼
                 sorted, merged, strictly non-overlapping highlights  +  verb–noun actions
```

| Cue | What it measures | Why |
|---|---|---|
| Motion | frame-to-frame change | hands and head are moving: something is being done |
| Local novelty | distance from the previous ~10 s | new events and scene changes |
| Global uniqueness | similarity to the most similar *distant* frames | repetitive and idle stretches score low |
| Activity cue (CLIP) | "hands using an object" vs "empty / blurry / ceiling" prompts (sport prompts for non-kitchen videos) | action awareness beyond pixel change |
| Dark-frame penalty | mean brightness | black or covered lens is never a highlight |

**Why short windows spread across the video.** *Action Coverage* (40% of the score) counts distinct actions that have ≥ 50% of their
duration inside the highlights, so many short windows spread over the whole video capture more distinct actions than a few long ones. The redundancy and
crowding penalties enforce that spread; the importance ranking targets *Highlight F1*.

**Label.** Zero-shot CLIP over the 15 non-`other` categories, class probabilities averaged over the video. Kitchen videos come out as `cooking` by content, never by file name.

**Actions.** Kitchen videos: every 2 s window is scored against ~2000 `verb noun` prompts; the scores are combined with the log-frequency of each pair in the
EPIC-KITCHENS-100 **training** split ([`weights/epic_pair_prior.json`](weights/epic_pair_prior.json)) and adjacent windows with the same pair are merged.
Other videos: coarse `other/other` actions on the highlights, which follows the organisers' FPVSum sample output.

**Graceful degradation.** If CLIP cannot be loaded, the pipeline still writes a valid file from motion/colour cues (label `other`, no actions) and says so on stderr;
`tools/run_all.py` refuses to start in that state unless `--allow-no-clip` is given.

---

## Data, models and integrity

* **Pretrained model:** OpenCLIP ViT-B/32, `laion2b_s34b_b79k` (downloaded at first run, cached in `weights/`).
* **Training-split statistics:** verb–noun pair frequencies from the public EPIC-KITCHENS-100 *training* annotations. [`tools/build_pair_prior.py`](tools/build_pair_prior.py) asserts that none of the 138 validation videos appears in that file.
* **Not used:** ground-truth annotations of the validation split or of FPVSum, hard-coded video ids, or any per-video rule. The only id-dependent code is in `tools/run_all.py`, which maps FPVSum names to YouTube ids (official list in [`data/fpvsum_videolist.csv`](data/fpvsum_videolist.csv)) in order to download videos.
* **Determinism:** no randomness; fixed sampling and parameters; CPU inference by default. Tiny floating-point differences between machines or ffmpeg builds can move a window boundary by one sample (0.5 s).

---

## Repository layout

```
run.sh                  single-video inference  ./run.sh <video.mp4> <out.json>
code/vsum/              the system (videoio, importance, select, clipmodel, label, actions, schema, pipeline)
weights/                epic_pair_prior.json (committed); CLIP checkpoint is downloaded here on first run
outputs/                one validated JSON per video
data/                   manifest.csv, supported_classes.json, fpvsum_videolist.csv
tools/                  validate.py, run_all.py, setup.sh, build_pair_prior.py
tests/                  pytest suite (schema rules, selection invariants, end-to-end run)
Dockerfile              reproducible environment
requirements.txt        dependencies (tested: Python 3.13, numpy 2.5.3, torch 2.15.0, open_clip_torch 4.2.2)
docs/                   organisers' problem statement, submission checklist
AI_DISCLOSURE.md
```

Tests: `pip install pytest && python -m pytest -q` (no weights or network needed; also run by CI on every push).

---

## Limitations (honest status)

* **No measured score.** The student package contains no ground truth, so Action mAP, Coverage, F1 and label accuracy were **not** measured, and no accuracy claims are made. The cue weights are set by hand, not tuned.
* Verb–noun recognition from single frames with zero-shot CLIP is weak. The prior helps with plausible pairs, but expect low Action mAP.
* The fixed 4 s window is a compromise; EPIC actions are often shorter. For videos under ~27 s a single 4 s window (the minimum the rules require) already exceeds 15%.
* Natural next steps: a hand–object detector as an additional cue, learning the cue weights on EPIC training videos, and a temporal action model instead of per-window CLIP scoring.

## AI disclosure

See [`AI_DISCLOSURE.md`](AI_DISCLOSURE.md).
