"""Tests for segment.filter_components.

Verifies that:
- the function does not mutate the input list or any of its dicts
- rejected items carry a 'rejection_reason' key
- kept items are returned without a 'rejection_reason' key
- filtering rules produce the correct classification
"""
import copy

import pytest

from segreader.segment import filter_components


# ── Shared fixture ────────────────────────────────────────────────────────────

def _make_stats():
    """Minimal stat dicts covering all four filter outcomes."""
    return [
        # area < 30 → noise
        {"label": 1, "area": 10,    "bbox": (0,  0, 5,  2),   "aspect_ratio": 2.5},
        # aspect_ratio > 2.0 AND area < 200 → colon/dot
        {"label": 2, "area": 50,    "bbox": (0,  0, 20, 5),   "aspect_ratio": 4.0},
        # area > 15 % of 100×100 image (threshold = 1500 px) → background
        {"label": 3, "area": 2000,  "bbox": (0,  0, 100, 100), "aspect_ratio": 1.0},
        # passes all rules → kept
        {"label": 4, "area": 500,   "bbox": (5, 10, 55, 30),  "aspect_ratio": 1.0},
    ]


# ── No-mutation tests ─────────────────────────────────────────────────────────

def test_input_list_not_mutated():
    """The original list object must not be modified."""
    stats = _make_stats()
    original_ids = [id(s) for s in stats]
    original_copy = copy.deepcopy(stats)

    filter_components(stats, image_height=100, image_width=100)

    assert stats == original_copy, "Input list was mutated"
    # Confirm the same dict objects are still in the list (not replaced)
    assert [id(s) for s in stats] == original_ids


def test_input_dicts_not_mutated():
    """No new keys must be injected into the original dicts."""
    stats = _make_stats()
    keys_before = [set(s.keys()) for s in stats]

    filter_components(stats, image_height=100, image_width=100)

    for s, original_keys in zip(stats, keys_before):
        assert set(s.keys()) == original_keys, (
            f"Dict for label={s['label']} was mutated: "
            f"added keys {set(s.keys()) - original_keys}"
        )


# ── Return-value tests ────────────────────────────────────────────────────────

def test_kept_and_rejected_counts():
    stats = _make_stats()
    kept, rejected = filter_components(stats, image_height=100, image_width=100)

    assert len(kept) == 1
    assert len(rejected) == 3


def test_kept_item_is_the_correct_one():
    stats = _make_stats()
    kept, _ = filter_components(stats, image_height=100, image_width=100)

    assert kept[0]["label"] == 4


def test_kept_item_has_no_rejection_reason():
    stats = _make_stats()
    kept, _ = filter_components(stats, image_height=100, image_width=100)

    for k in kept:
        assert "rejection_reason" not in k


def test_kept_item_is_independent_copy():
    """Modifying the returned kept dict must not affect the original."""
    stats = _make_stats()
    kept, _ = filter_components(stats, image_height=100, image_width=100)

    kept[0]["area"] = 99999
    assert stats[3]["area"] == 500, "Kept dict shares identity with original input dict"


def test_rejected_items_all_have_rejection_reason():
    stats = _make_stats()
    _, rejected = filter_components(stats, image_height=100, image_width=100)

    for r in rejected:
        assert "rejection_reason" in r
        assert isinstance(r["rejection_reason"], str)
        assert len(r["rejection_reason"]) > 0


# ── Rule-correctness tests ────────────────────────────────────────────────────

def test_noise_rejection_reason():
    stats = [{"label": 1, "area": 10, "bbox": (0, 0, 5, 2), "aspect_ratio": 2.5}]
    _, rejected = filter_components(stats, image_height=100, image_width=100)

    assert len(rejected) == 1
    assert rejected[0]["rejection_reason"].startswith("noise")


def test_colon_rejection_reason():
    stats = [{"label": 2, "area": 50, "bbox": (0, 0, 20, 5), "aspect_ratio": 4.0}]
    _, rejected = filter_components(stats, image_height=100, image_width=100)

    assert len(rejected) == 1
    assert rejected[0]["rejection_reason"].startswith("colon")


def test_background_rejection_reason():
    # 15 % of 100×100 = 1500; area=2000 > 1500 → background
    stats = [{"label": 3, "area": 2000, "bbox": (0, 0, 100, 100), "aspect_ratio": 1.0}]
    _, rejected = filter_components(stats, image_height=100, image_width=100)

    assert len(rejected) == 1
    assert rejected[0]["rejection_reason"].startswith("background")


def test_no_background_filter_when_image_width_zero():
    """image_width=0 disables the background-area rule (max_area = inf)."""
    stats = [{"label": 1, "area": 999999, "bbox": (0, 0, 1000, 1000), "aspect_ratio": 1.0}]
    kept, rejected = filter_components(stats, image_height=1000, image_width=0)

    assert len(kept) == 1
    assert len(rejected) == 0


def test_empty_input():
    kept, rejected = filter_components([], image_height=100, image_width=100)

    assert kept == []
    assert rejected == []


def test_noise_takes_priority_over_colon_rule():
    """A blob with area < 30 AND aspect_ratio > 2 is classified as noise, not colon."""
    stats = [{"label": 1, "area": 5, "bbox": (0, 0, 10, 2), "aspect_ratio": 5.0}]
    _, rejected = filter_components(stats, image_height=100, image_width=100)

    assert rejected[0]["rejection_reason"].startswith("noise")
