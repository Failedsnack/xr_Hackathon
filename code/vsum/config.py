"""Central configuration. Every value is fixed, so a run is deterministic."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("VSUM_DATA", ROOT / "data"))
WEIGHTS_DIR = Path(os.environ.get("VSUM_WEIGHTS", ROOT / "weights"))
PRIOR_PATH = ROOT / "weights" / "epic_pair_prior.json"   # small, committed to the repo
TEAM_ID_FILE = ROOT / "team_id.txt"
DEFAULT_TEAM_ID = "SRMIST_TEAM"


def env_flag(name: str) -> bool:
    return os.environ.get(name, "") == "1"


def device() -> str:
    return os.environ.get("VSUM_DEVICE", "cpu")


# --- CLIP backbone (open_clip); weights are fetched on first use into WEIGHTS_DIR ---
CLIP_ARCH = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
LOGIT_SCALE = 100.0

# --- frame sampling ---
FPS = 2.0            # sampled frames per second
FRAME = 224          # frames are squashed to FRAME x FRAME
CHUNK = 96           # frames decoded per chunk (bounds memory)
CLIP_EVERY = 2       # CLIP sees every 2nd sampled frame -> 1 frame per second
LOWRES = 56          # side of the low-resolution frame used for motion / colour features

# --- highlight selection ---
BUDGET_FRAC = 0.14       # target share of the video (hard limit in the rules: 15%)
HARD_BUDGET_FRAC = 0.15
SEG_LEN = 4.0            # candidate window length in seconds
MIN_SEG = 4.0            # at least one highlight must be >= this long
LAMBDA_SIM = 0.6         # penalty for looking like an already chosen window
LAMBDA_TIME = 0.5        # penalty for sitting next to an already chosen window

# --- importance ---
W_NO_CLIP = {"motion": 0.40, "novelty": 0.35, "unique": 0.25}
W_CLIP = {"motion": 0.25, "novelty": 0.20, "unique": 0.20, "interact": 0.35}
NOVELTY_LONG = 20        # frames (10 s) of context for novelty
NOVELTY_SHORT = 4        # frames (2 s) skipped before the context window
UNIQUE_EXCLUDE = 10      # frames (5 s) around a frame ignored when looking for look-alikes
UNIQUE_K = 10
DARK_LEVEL = 20.0        # mean grey level below which a frame is treated as idle
DARK_PENALTY = 2.0

# --- actions ---
ACT_WIN = 2              # CLIP frames per window (2 s)
ACT_STRIDE = 1
ACT_TOPK = 3             # verb-noun pairs kept per window
ACT_MAX = 2000           # format limit
ACT_LOGIT_SCALE = 50.0
ACT_PRIOR_WEIGHT = 1.0
ACT_MERGE_GAP = 0.5      # seconds
