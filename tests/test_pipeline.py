import json
import subprocess
import sys
from pathlib import Path

import pytest

from vsum.classes import load_classes
from vsum.pipeline import process_video
from vsum.schema import validate

ROOT = Path(__file__).resolve().parents[1]


def test_fallback_path_produces_a_valid_file(synthetic_video):
    info = {}
    res = process_video(str(synthetic_video), info=info)
    assert validate(res, load_classes())[0] == []
    assert res["video_id"] == "synthetic_clip"
    assert abs(res["duration_sec"] - 100.0) < 0.5
    assert "clip_unavailable" in info["notes"]
    assert info["highlight_frac"] <= 0.15


def test_output_is_deterministic(synthetic_video):
    a = process_video(str(synthetic_video))
    b = process_video(str(synthetic_video))
    assert a == b


def test_expected_duration_is_only_used_when_it_agrees(synthetic_video):
    info = {}
    res = process_video(str(synthetic_video), expected_duration=100.3, info=info)
    assert res["duration_sec"] == 100.3
    res = process_video(str(synthetic_video), expected_duration=400.0, info=info)
    assert abs(res["duration_sec"] - 100.0) < 0.5 and any("duration_mismatch" in n for n in info["notes"])


def test_run_sh_contract(synthetic_video, tmp_path):
    out = tmp_path / "out" / "x.json"
    p = subprocess.run([str(ROOT / "run.sh"), str(synthetic_video), str(out)],
                       env={"PATH": __import__("os").environ["PATH"], "VSUM_DISABLE_CLIP": "1"},
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    res = json.loads(out.read_text())
    assert validate(res, load_classes())[0] == []
    bad = subprocess.run([str(ROOT / "run.sh")], capture_output=True, text=True)
    assert bad.returncode == 2


def test_validator_cli(synthetic_video, tmp_path):
    out = tmp_path / "outs"
    out.mkdir()
    res = process_video(str(synthetic_video))
    (out / "synthetic_clip.json").write_text(json.dumps(res))
    ok = subprocess.run([sys.executable, str(ROOT / "tools" / "validate.py"), "--dir", str(out), "--no-manifest"],
                        capture_output=True, text=True)
    assert ok.returncode == 0 and "OK" in ok.stdout
    (out / "broken.json").write_text("{}")
    bad = subprocess.run([sys.executable, str(ROOT / "tools" / "validate.py"), "--dir", str(out), "--no-manifest"],
                         capture_output=True, text=True)
    assert bad.returncode == 1


def test_clip_path_smoke(synthetic_video, tmp_path, monkeypatch):
    """Exercises the CLIP code path with a randomly initialised encoder (shapes and plumbing, not accuracy)."""
    pytest.importorskip("torch")
    pytest.importorskip("open_clip")
    monkeypatch.delenv("VSUM_DISABLE_CLIP")
    monkeypatch.setenv("VSUM_CLIP_RANDOM_INIT", "1")
    from vsum import clipmodel
    clipmodel.reset()
    info = {}
    res = process_video(str(synthetic_video), info=info)
    assert "clip_unavailable" not in info["notes"]
    assert validate(res, load_classes())[0] == []


def test_kitchen_actions_smoke(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("open_clip")
    import numpy as np
    monkeypatch.delenv("VSUM_DISABLE_CLIP")
    monkeypatch.setenv("VSUM_CLIP_RANDOM_INIT", "1")
    from vsum import clipmodel
    from vsum.actions import kitchen_actions
    clipmodel.reset()
    clip = clipmodel.get_clip()
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(40, 512)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    acts = kitchen_actions(emb, clip, 40.0)
    assert acts and all(0 <= a["score"] <= 1 and a["segment"][0] < a["segment"][1] <= 40.0 for a in acts)
    classes = load_classes()
    assert all(a["verb"] in classes["verbs"] and a["noun"] in classes["nouns"] for a in acts)
