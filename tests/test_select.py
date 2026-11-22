import numpy as np
import pytest

from vsum import config as C
from vsum.select import merge_segments, select_highlights


def fake(duration, seed):
    rng = np.random.default_rng(seed)
    n = int(duration * C.FPS)
    imp = rng.normal(size=n)
    desc = rng.normal(size=(n, 16)).astype(np.float32)
    desc /= np.linalg.norm(desc, axis=1, keepdims=True)
    return imp, desc


@pytest.mark.parametrize("duration", [30, 61.5, 120, 300.25, 599.9, 1968.6])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_selection_invariants(duration, seed):
    imp, desc = fake(duration, seed)
    segs = select_highlights(imp, desc, duration)
    assert segs, "at least one highlight"
    assert all(0 <= s < e <= duration for s, e in segs)
    assert all(segs[i][0] > segs[i - 1][1] for i in range(1, len(segs))), "sorted and strictly non-overlapping"
    assert max(e - s for s, e in segs) >= 4.0
    assert sum(e - s for s, e in segs) <= 0.15 * duration


def test_short_video_still_gets_one_valid_highlight():
    imp, desc = fake(11.13, 0)
    segs = select_highlights(imp, desc, 11.13)
    assert len(segs) >= 1 and max(e - s for s, e in segs) >= 4.0


def test_peaks_are_preferred_and_spread_out():
    n = int(200 * C.FPS)
    imp = np.zeros(n)
    for centre in (20, 60, 100, 140, 180):
        imp[int(centre * C.FPS) - 4: int(centre * C.FPS) + 4] = 10.0
    rng = np.random.default_rng(3)
    desc = rng.normal(size=(n, 16)).astype(np.float32)
    desc /= np.linalg.norm(desc, axis=1, keepdims=True)
    segs = select_highlights(imp, desc, 200.0)
    for centre in (20, 60, 100, 140, 180):
        assert any(s <= centre <= e for s, e in segs), f"peak at {centre}s missed"


def test_deterministic():
    imp, desc = fake(300, 5)
    assert select_highlights(imp, desc, 300) == select_highlights(imp, desc, 300)


def test_merge_segments():
    assert merge_segments([[5, 8], [1, 3], [3, 4], [7, 10]]) == [[1, 4], [5, 10]]
