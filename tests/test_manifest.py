import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_all  # noqa: E402


def rows():
    with open(ROOT / "data" / "manifest.csv", newline="") as f:
        return list(csv.DictReader(f))


def test_manifest_shape():
    r = rows()
    assert len(r) == 187
    assert sum(x["dataset"] == "epic_kitchens_100" for x in r) == 138
    assert sum(x["dataset"] == "fpvsum" for x in r) == 49
    assert len({x["video_id"] for x in r}) == 187


def test_class_dictionary_sizes():
    from vsum.classes import load_classes
    c = load_classes()
    assert (len(c["labels"]), len(c["verbs"]), len(c["nouns"])) == (16, 98, 301)
    assert "ski" in c["labels"] and "skiing" not in c["labels"]


def test_every_fpvsum_video_maps_to_a_youtube_id():
    ids = run_all.load_fpvsum_ids(ROOT / "data" / "fpvsum_videolist.csv")
    for x in rows():
        if x["dataset"] == "fpvsum":
            assert x["video_id"] in ids, x["video_id"]
            assert len(run_all.youtube_id_for(x["video_id"], ids)) == 11


def test_numbered_bikepolo_ids_use_the_official_list_not_the_name():
    ids = run_all.load_fpvsum_ids(ROOT / "data" / "fpvsum_videolist.csv")
    assert run_all.youtube_id_for("BikePolo_DHbYfmjX3WQ_1", ids) == "hkNp3udSNV8"
    assert run_all.youtube_id_for("Ski_-Z5a4UQmZLs", {}) == "-Z5a4UQmZLs"     # fallback rule


def test_epic_ids_are_downloader_compatible():
    for x in rows():
        if x["dataset"] == "epic_kitchens_100":
            p, n = x["video_id"].split("_")
            assert p.startswith("P") and len(p) == 3 and len(n) in (2, 3)
