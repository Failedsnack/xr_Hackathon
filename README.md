# Action-Aware Video Summarization — Samsung PRISM XR Hackathon 2026, Track 04

Training-free system that selects the important, non-redundant moments of first-person
(egocentric) videos and compiles a highlight of about 14% of the original footage.

## Run
```bash
pip install -r requirements.txt      # ffmpeg/ffprobe must be on PATH
./run.sh <video.mp4> <out.json>
python3 tools/validate.py outputs/ manifest.csv
```
CLIP weights (ViT-B-32, LAION-2B) are downloaded automatically into `weights/` on the first run.
Without them, the label falls back to `"other"`.

## Pipeline
1. **Frame sampling**: ffmpeg decodes at 2 fps, downscaled to 64×64.
2. **Per-frame features**: a colour histogram (512 bins) and a coarse layout descriptor.
3. **Importance score** (z-scored, then smoothed over about 1.5 s):
   - *motion*: frame-difference energy. In egocentric video this tracks hand and head activity.
   - *local novelty*: how far a frame is from the previous ~10 s of context, i.e. new events.
   - *global uniqueness*: penalises frames similar to many other frames, i.e. repetitive or idle footage.
4. **Highlight selection**:
   - Candidate 4 s windows start every 0.5 s.
   - A greedy pick takes the highest-importance window, minus a visual-redundancy penalty and a temporal-crowding penalty.
   - It stops at 14% of the duration. Selected windows are merged so they are sorted and strictly non-overlapping.
   - Spreading short segments across the whole video maximises *Action Coverage* (40%), while importance ranking targets *Highlight F1* (25%).
5. **Video label**: CLIP zero-shot over the supported label set, averaged over 32 frames.
6. **Actions**: an empty list `[]` (allowed by the rules).

The pipeline is deterministic: no randomness, fixed sampling and fixed parameters.
No video IDs are hardcoded, and no ground-truth annotations are used.

## Structure
```
run.sh              single-video inference
code/vsum.py        complete inference source
tools/validate.py   output format checker (README section 5 rules)
weights/            CLIP checkpoint cache (downloaded on first run)
outputs/            <video_id>.json files
requirements.txt
AI_DISCLOSURE.md
```

## Limitations / future work
- Action mAP: add verb–noun zero-shot recognition with CLIP, using EPIC-100 *training*-split priors.
- Hand-object detection, as an interaction cue.
