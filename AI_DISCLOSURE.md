# AI Disclosure

## AI assistance used to build this submission
- **Claude (Anthropic)** was used as a coding and writing assistant for: the pipeline design and code (`code/vsum/`), the batch runner and
  validator (`tools/`), the tests, the Dockerfile and CI workflow, the README, the presentation draft and the demo-video script.
- The author (Amritanshu Aditya) directed the work, reviewed the code and outputs, and is responsible for the submission.

## AI models inside the submitted system
| Model | Role | Training / fine-tuning |
|---|---|---|
| OpenCLIP ViT-B/32, `laion2b_s34b_b79k` | zero-shot video label, activity cue, verb–noun scoring | none: pretrained weights used as released |

No generative model is used at inference time. Highlights are timestamps into the original footage.

## Data used
- **EPIC-KITCHENS-100 training-split annotations** (public): only verb–noun pair frequencies, stored in `weights/epic_pair_prior.json`.
  `tools/build_pair_prior.py` asserts that no validation video from `manifest.csv` appears in the training file.
- **FPVSum video list** (`data/fpvsum_videolist.csv`, from github.com/azuxmioy/fpvsum): used only to map video names to YouTube ids for downloading.
- **Not used:** ground-truth annotations of the 138 EPIC validation videos or the 49 FPVSum videos.

## Reproducibility and integrity
Outputs in `outputs/` are produced by `tools/run_all.py`, which calls the same code path as `./run.sh <video.mp4> <out.json>`.
There are no hard-coded video ids or per-video rules, and no manually edited output files.
