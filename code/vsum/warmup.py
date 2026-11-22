"""Download the CLIP weights and build the text-embedding cache, so the first real run is not slow.

    python -m vsum.warmup            (run.sh-style: PYTHONPATH=code)
"""
from __future__ import annotations

import sys
import time

from .classes import load_classes, load_pair_prior
from .clipmodel import get_clip
from .prompts import interaction_prompts, label_prompts, pair_prompt


def main() -> int:
    t0 = time.time()
    clip = get_clip()
    if clip is None:
        print("CLIP could not be loaded (see the warning above).", file=sys.stderr)
        return 1
    classes = load_classes()
    for lab in classes["labels"]:
        clip.encode_texts(label_prompts(lab))
    for domain in ("kitchen", "sport"):
        clip.encode_texts(sum(interaction_prompts(domain), []))
    prior = load_pair_prior()
    if prior is not None:
        clip.encode_texts([pair_prompt(v, n) for v, n in prior[0]])
    print(f"CLIP ready: weights and text embeddings cached in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
