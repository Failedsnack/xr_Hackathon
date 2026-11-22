# AI Disclosure

## Who did what
This submission is the work of one author, **Amritanshu Aditya**. He chose the track, set the requirements and constraints,
decided what was in scope, ran the system on the competition data, and reviewed and accepted every change in this repository.
He is responsible for the submission and its outputs.

An AI assistant (**Claude, by Anthropic**) was used as a development aid for the repetitive and mechanical parts of the work:

- drafting boilerplate and implementation code from the author's requirements (`code/vsum/`, `tools/`)
- writing and running the unit tests (`tests/`)
- the Dockerfile, CI workflow and `.gitignore`
- formatting the README, the submission checklist and the presentation draft

The author reviewed this output, ran the pipeline end to end, and corrected or rejected what did not meet the task rules
(output schema, class names, reproducibility, no use of ground truth).

## Approach and design choices (author's direction)
- Training-free and reproducible: pretrained weights only, no fine-tuning, no per-video rules.
- Must run from the single command `./run.sh <video.mp4> <out.json>` on a laptop within the time limit,
  so the pipeline streams video in chunks and has a resumable batch runner.
- No ground truth from the validation videos is used anywhere. Only public training-split statistics are used as a prior.
- Honest reporting: the README states that no score was measured, since no ground truth is available.

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
