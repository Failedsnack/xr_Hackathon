# Submission checklist (deadline: 22 Nov 2026, 11:59 PM IST, no extension)

Source: organisers' slides ("Submission Guidelines") and `docs/TEAM_README.md`. Only the primary member submits, once; the form can be edited afterwards.

## Repository (public GitHub link)
- [ ] Repository opens in a private browser window (= public)
- [ ] `run.sh`, `code/`, `weights/`, `requirements.txt`, `outputs/`, `README.md` in the root (the required package layout)
- [ ] **Docker files** (`Dockerfile`) and reproducible setup steps in `README.md`
- [ ] `outputs/` holds one valid JSON per manifest video: `python tools/validate.py --dir outputs/ --manifest data/manifest.csv` prints `OK`
- [ ] `team_id` is identical in every output file (the validator checks this) and equals the team name used in the form
- [ ] `AI_DISCLOSURE.md` present
- [ ] CI is green (Actions tab)

## Presentation, video
- [ ] Presentation file named **`CollegeName_TeamName`** (`.pptx` or `.pdf`), following the "Submission_Final_forGithub" template:
      Track no. / members / team / college, Problem, Solution, Architecture, Track final output, Differentiation, Demo & local setup, Brownie slides, Thank you
- [ ] Demo video, **max 3 minutes**, in the repository (`Video .mp4`), optionally also on YouTube / Drive with the link added to the README

## Google Form
| Field | Value |
|---|---|
| Email | the SRM e-mail the form is signed in with |
| `<College>_<TeamName>` | `SRMIST_<TeamName>` (same string as `team_id`) |
| College | SRM Institute of Science and Technology |
| Theme selected | Track 04: Action-Aware Video Summarization |
| Team Member 1 (primary) | name, e-mail, mobile |
| GitHub link | the repository URL, set to public |
| Checklist for GitHub | tick: Submission package, Source code, Presentation, Video, AI disclosure, README (no APK for this track) |

Never put passwords in the form.

## What the evaluation rewards (100 pts)
Action coverage 40 · Highlight F1 25 · Action mAP 25 · Label accuracy 10. A missing or invalid file scores 0 for that video.
Only a hidden subset of the 187 videos is scored, so every video should have a valid file.

## Integrity (disqualification risks)
No ground-truth annotations, no hard-coded video ids, outputs reproducible by `./run.sh`. All three are covered in the README.
