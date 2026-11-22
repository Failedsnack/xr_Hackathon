import copy
import json

from vsum.classes import load_classes
from vsum.schema import finalize, validate

CLASSES = load_classes()


def good():
    return {
        "team_id": "T", "video_id": "v", "duration_sec": 100.0,
        "label": {"name": "ski", "score": 0.9},
        "actions": [{"segment": [1.0, 3.0], "verb": "open", "noun": "fridge", "score": 0.5}],
        "highlights": [{"segment": [10.0, 14.0]}, {"segment": [40.0, 44.0]}],
    }


def test_good_example_is_valid():
    assert validate(good(), CLASSES) == ([], [])


def test_official_samples_are_valid_except_known_label_typo():
    # The organisers' own skiing_03.json uses "skiing", which the README says is wrong (must be "ski").
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    errors, _ = validate(json.loads((root / "examples" / "P02_109.json").read_text()), CLASSES)
    assert errors == []
    errors, _ = validate(json.loads((root / "examples" / "skiing_03.json").read_text()), CLASSES)
    assert any("skiing" in e for e in errors)


def test_validator_catches_each_rule():
    cases = {
        "missing": lambda r: r.pop("label"),
        "empty team": lambda r: r.update(team_id=""),
        "bad duration": lambda r: r.update(duration_sec=0),
        "label score": lambda r: r["label"].update(score=1.5),
        "label name": lambda r: r["label"].update(name="skiing"),
        "action range": lambda r: r["actions"][0].update(segment=[5.0, 200.0]),
        "action class": lambda r: r["actions"][0].update(verb="juggle"),
        "no highlights": lambda r: r.update(highlights=[]),
        "overlap": lambda r: r.update(highlights=[{"segment": [10, 20]}, {"segment": [15, 25]}]),
        "touching": lambda r: r.update(highlights=[{"segment": [10, 20]}, {"segment": [20, 25]}]),
        "unsorted": lambda r: r.update(highlights=[{"segment": [40, 50]}, {"segment": [10, 20]}]),
        "all short": lambda r: r.update(highlights=[{"segment": [10, 13]}, {"segment": [20, 23.9]}]),
        "outside video": lambda r: r.update(highlights=[{"segment": [98, 104]}]),
    }
    for name, mutate in cases.items():
        r = good()
        mutate(r)
        errors, _ = validate(r, CLASSES)
        assert errors, f"rule not enforced: {name}"


def test_manifest_duration_tolerance():
    assert validate(good(), CLASSES, manifest_duration=100.4)[0] == []
    assert validate(good(), CLASSES, manifest_duration=101.0)[0] != []


def test_budget_overrun_is_a_warning_not_an_error():
    r = good()
    r["highlights"] = [{"segment": [0, 30]}]
    errors, warnings = validate(r, CLASSES)
    assert errors == [] and warnings


def test_finalize_repairs_messy_results():
    r = good()
    r["label"] = {"name": "skiing", "score": 7}
    r["actions"] = [{"segment": [-3, 500], "verb": "juggle", "noun": "fridge", "score": 3},
                    {"segment": [5, 5], "verb": "open", "noun": "fridge", "score": 0.4}]
    r["highlights"] = [{"segment": [50, 52]}, {"segment": [10, 11]}, {"segment": [10.5, 12]}, {"segment": [99, 400]}]
    out = finalize(copy.deepcopy(r), CLASSES)
    assert validate(out, CLASSES)[0] == []
    assert out["label"] == {"name": "other", "score": 1.0}
    assert out["actions"][0]["verb"] == "other"


def test_finalize_guarantees_a_four_second_segment():
    r = good()
    r["highlights"] = [{"segment": [20, 22]}]
    out = finalize(r, CLASSES)
    assert validate(out, CLASSES)[0] == []
    assert max(h["segment"][1] - h["segment"][0] for h in out["highlights"]) >= 4.0
