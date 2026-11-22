"""Lazy CLIP encoder with an on-disk text-embedding cache.

If CLIP cannot be loaded (no torch, no weights, offline) get_clip() returns None and the pipeline
falls back to colour/motion features, so a valid output file is still produced.
"""
from __future__ import annotations

import os
import sys

import numpy as np

from . import config as C

_STATE = {"tried": False, "clip": None}


class ClipEncoder:
    def __init__(self) -> None:
        import torch
        import open_clip

        torch.manual_seed(0)
        self.torch = torch
        self.device = C.device()
        random_init = C.env_flag("VSUM_CLIP_RANDOM_INIT")  # used by the tests only (no weights needed)
        C.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        if random_init:
            model, _, _ = open_clip.create_model_and_transforms(C.CLIP_ARCH, pretrained=None)
        else:
            model, _, _ = open_clip.create_model_and_transforms(
                C.CLIP_ARCH, pretrained=C.CLIP_PRETRAINED, cache_dir=str(C.WEIGHTS_DIR))
        self.model = model.eval().to(self.device)
        self.tokenize = open_clip.get_tokenizer(C.CLIP_ARCH)
        mean = getattr(open_clip, "OPENAI_DATASET_MEAN", (0.48145466, 0.4578275, 0.40821073))
        std = getattr(open_clip, "OPENAI_DATASET_STD", (0.26862954, 0.26130258, 0.27577711))
        self.mean = torch.tensor(mean, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=self.device).view(1, 3, 1, 1)
        self.scale = C.LOGIT_SCALE
        tag = "random" if random_init else C.CLIP_PRETRAINED
        self._cache_path = C.WEIGHTS_DIR / f"text_cache_{C.CLIP_ARCH}_{tag}.npz"
        self._cache: dict[str, np.ndarray] = {}
        self._load_cache()

    # ---- images ----
    def encode_images(self, frames: np.ndarray, batch: int = 64) -> np.ndarray:
        """uint8 [n, 224, 224, 3] -> L2-normalised float32 [n, d]."""
        torch = self.torch
        out = []
        with torch.inference_mode():
            for i in range(0, len(frames), batch):
                x = torch.from_numpy(np.ascontiguousarray(frames[i:i + batch])).to(self.device)
                x = x.permute(0, 3, 1, 2).float().div_(255.0)
                f = self.model.encode_image((x - self.mean) / self.std)
                f = f / f.norm(dim=-1, keepdim=True)
                out.append(f.float().cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    # ---- text (cached) ----
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        torch = self.torch
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            with torch.inference_mode():
                for i in range(0, len(missing), 256):
                    part = missing[i:i + 256]
                    f = self.model.encode_text(self.tokenize(part).to(self.device))
                    f = f / f.norm(dim=-1, keepdim=True)
                    for t, v in zip(part, f.float().cpu().numpy()):
                        self._cache[t] = v.astype(np.float32)
            self._save_cache()
        return np.stack([self._cache[t] for t in texts])

    def _load_cache(self) -> None:
        try:
            if self._cache_path.exists():
                z = np.load(self._cache_path, allow_pickle=False)
                self._cache = {str(k): v for k, v in zip(z["keys"], z["vecs"])}
        except Exception:
            self._cache = {}

    def _save_cache(self) -> None:
        try:
            keys = np.array(list(self._cache.keys()))
            vecs = np.stack(list(self._cache.values()))
            tmp = self._cache_path.with_name(self._cache_path.name + f".{os.getpid()}.tmp")
            with open(tmp, "wb") as f:
                np.savez_compressed(f, keys=keys, vecs=vecs)
            os.replace(tmp, self._cache_path)
        except Exception as e:  # the cache is an optimisation only
            print(f"[vsum] could not save the text cache: {e}", file=sys.stderr)


def get_clip():
    """Return the shared ClipEncoder, or None when CLIP is disabled or unavailable."""
    if C.env_flag("VSUM_DISABLE_CLIP"):
        return None
    if _STATE["tried"]:
        return _STATE["clip"]
    _STATE["tried"] = True
    try:
        _STATE["clip"] = ClipEncoder()
    except Exception as e:
        print(f"[vsum] WARNING: CLIP unavailable ({type(e).__name__}: {e}). "
              "Falling back to colour/motion features: label will be 'other' and actions empty.",
              file=sys.stderr)
        _STATE["clip"] = None
    return _STATE["clip"]


def reset() -> None:
    """Forget the loaded encoder (tests)."""
    _STATE["tried"] = False
    _STATE["clip"] = None
