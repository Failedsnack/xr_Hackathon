# weights/

| File | Size | In git? | What it is |
|---|---|---|---|
| `epic_pair_prior.json` | 42 KB | yes | Frequencies of the 2000 most common verb-noun pairs in the EPIC-KITCHENS-100 **training** split (`tools/build_pair_prior.py`). |
| `open_clip_*` (OpenCLIP ViT-B/32, `laion2b_s34b_b79k`) | ~600 MB | no | Pretrained CLIP checkpoint, used zero-shot (no fine-tuning). Downloaded automatically on first run, or by `python -m vsum.warmup`. Too large for a normal GitHub push (100 MB limit); the Dockerfile bakes it into the image. |
| `text_cache_*.npz` | ~2 MB | no | Cache of the text embeddings of all prompts, so they are encoded only once. |

To use a different folder for the checkpoint: `VSUM_WEIGHTS=/path/to/dir ./run.sh video.mp4 out.json`.
