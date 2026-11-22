import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory):
    """A 100 s clip made of two different synthetic scenes (no real footage needed)."""
    out = tmp_path_factory.mktemp("vid") / "synthetic_clip.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25",
         "-f", "lavfi", "-i", "mandelbrot=size=320x240:rate=25",
         "-filter_complex", "[0]trim=duration=50,setpts=PTS-STARTPTS[a];[1]trim=duration=50,setpts=PTS-STARTPTS[b];[a][b]concat=n=2:v=1",
         "-pix_fmt", "yuv420p", str(out)],
        check=True)
    return out


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    """Default for every test: CLIP off, so the suite needs neither weights nor network."""
    monkeypatch.setenv("VSUM_DISABLE_CLIP", "1")
    monkeypatch.delenv("VSUM_CLIP_RANDOM_INIT", raising=False)
    monkeypatch.setenv("VSUM_WEIGHTS", str(tmp_path / "weights"))
    from vsum import clipmodel
    clipmodel.reset()
    yield
    clipmodel.reset()
