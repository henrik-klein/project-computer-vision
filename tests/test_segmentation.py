"""Tests for segment.filter_components, smooth_projection,
compute_projection_threshold, split_run_at_valleys, find_digit_boxes,
and vertical_projection.

Verifies that:
- the function does not mutate the input list or any of its dicts
- rejected items carry a 'rejection_reason' key
- kept items are returned without a 'rejection_reason' key
- filtering rules produce the correct classification
- projection helpers are pure and return correct shapes/values
"""
import copy

import numpy as np
import pytest

from segreader.segment import (
    filter_components,
    filter_components_with_fallback,
    component_summary_from_stats,
    smooth_projection,
    compute_projection_threshold,
    split_run_at_valleys,
    find_digit_boxes,
    vertical_projection,
)


# ── component_summary_from_stats ─────────────────────────────────────────────

def test_csfas_basic():
    stats = [{"label": 1, "area": 500}, {"label": 2, "area": 200}]
    result = component_summary_from_stats(stats, 10000)
    assert result["count"] == 2
    assert result["largest_area"] == 500
    assert result["largest_ratio"] == 0.05


def test_csfas_single_component():
    stats = [{"label": 1, "area": 1500}]
    result = component_summary_from_stats(stats, 10000)
    assert result["count"] == 1
    assert result["largest_area"] == 1500
    assert result["largest_ratio"] == 0.15


def test_csfas_empty_stats():
    result = component_summary_from_stats([], 10000)
    assert result["count"] == 0
    assert result["largest_area"] == 0
    assert result["largest_ratio"] == 0.0


def test_csfas_zero_image_area():
    stats = [{"label": 1, "area": 500}]
    result = component_summary_from_stats(stats, 0)
    assert result["count"] == 1
    assert result["largest_area"] == 0
    assert result["largest_ratio"] == 0.0


def test_csfas_largest_is_max():
    stats = [{"label": 1, "area": 100}, {"label": 2, "area": 800}, {"label": 3, "area": 50}]
    result = component_summary_from_stats(stats, 10000)
    assert result["largest_area"] == 800


def test_csfas_does_not_mutate_stats():
    stats = [{"label": 1, "area": 500}]
    import copy
    original = copy.deepcopy(stats)
    component_summary_from_stats(stats, 10000)
    assert stats == original


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


# ── filter_components_with_fallback ──────────────────────────────────────────

def _kept_stat(**kwargs) -> dict:
    base = {"label": 1, "area": 500, "bbox": (0, 0, 50, 20), "aspect_ratio": 1.0}
    base.update(kwargs)
    return base


def test_fallback_not_triggered_when_some_kept():
    """Normal filter keeps a component → fallback is not used."""
    stats = [_kept_stat(area=500)]
    kept, rejected, fallback_used = filter_components_with_fallback(stats, 100, 100)
    assert len(kept) == 1
    assert not fallback_used


def test_fallback_triggered_when_background_rejects_all():
    """All components rejected as background → fallback without background rule."""
    # 100×100 image, normal max_area=0.15*10000=1500; component area=2000 → background
    stats = [{"label": 1, "area": 2000, "bbox": (0, 0, 100, 50), "aspect_ratio": 1.0}]
    kept, rejected, fallback_used = filter_components_with_fallback(stats, 100, 100)
    assert fallback_used
    assert len(kept) == 1
    assert "rejection_reason" not in kept[0]


def test_fallback_still_rejects_noise():
    """Fallback activates but noise components are still rejected."""
    stats = [{"label": 1, "area": 5, "bbox": (0, 0, 5, 2), "aspect_ratio": 2.5}]
    kept, rejected, fallback_used = filter_components_with_fallback(stats, 100, 100)
    assert len(kept) == 0
    assert not fallback_used  # fallback tried but found nothing → not considered "used"


def test_fallback_not_triggered_on_empty_stats():
    kept, rejected, fallback_used = filter_components_with_fallback([], 100, 100)
    assert len(kept) == 0
    assert not fallback_used


def test_fallback_returns_false_when_first_pass_keeps_some():
    """Mixed stats: one kept, one background-rejected → normal path, fallback=False."""
    stats = [
        {"label": 1, "area": 500,  "bbox": (0, 0, 50, 20), "aspect_ratio": 1.0},  # kept
        {"label": 2, "area": 2000, "bbox": (0, 0, 100, 50), "aspect_ratio": 1.0},  # background
    ]
    kept, rejected, fallback_used = filter_components_with_fallback(stats, 100, 100)
    assert len(kept) == 1
    assert not fallback_used


def test_fallback_does_not_mutate_stats():
    stats = [{"label": 1, "area": 2000, "bbox": (0, 0, 100, 50), "aspect_ratio": 1.0}]
    import copy
    original = copy.deepcopy(stats)
    filter_components_with_fallback(stats, 100, 100)
    assert stats == original




# ── vertical_projection ───────────────────────────────────────────────────────

def test_vertical_projection_sums_columns():
    binary = np.array([[1, 0, 1], [1, 1, 0]], dtype=np.uint8)
    proj = vertical_projection(binary)
    np.testing.assert_array_equal(proj, [2.0, 1.0, 1.0])


def test_vertical_projection_dtype():
    binary = np.ones((3, 4), dtype=np.uint8)
    assert vertical_projection(binary).dtype == np.float64


def test_vertical_projection_all_zeros():
    binary = np.zeros((5, 5), dtype=np.uint8)
    proj = vertical_projection(binary)
    assert proj.max() == 0.0


# ── smooth_projection ─────────────────────────────────────────────────────────

def test_smooth_projection_window_one_is_identity():
    proj = np.array([1.0, 5.0, 2.0, 8.0])
    result = smooth_projection(proj, window=1)
    np.testing.assert_array_equal(result, proj)


def test_smooth_projection_window_one_returns_same_object():
    proj = np.array([1.0, 5.0, 2.0, 8.0])
    assert smooth_projection(proj, window=1) is proj


def test_smooth_projection_window_three_reduces_spike():
    """A single spike is spread by a window-3 average: peak value decreases."""
    proj = np.array([0.0, 0.0, 10.0, 0.0, 0.0])
    result = smooth_projection(proj, window=3)
    assert result[2] < 10.0


def test_smooth_projection_same_length():
    proj = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert len(smooth_projection(proj, window=3)) == len(proj)


def test_smooth_projection_empty_returns_empty():
    proj = np.array([], dtype=np.float64)
    result = smooth_projection(proj, window=3)
    assert len(result) == 0


def test_smooth_projection_does_not_mutate_input():
    proj = np.array([1.0, 5.0, 1.0])
    original = proj.copy()
    smooth_projection(proj, window=3)
    np.testing.assert_array_equal(proj, original)


# ── compute_projection_threshold ─────────────────────────────────────────────

def test_threshold_returns_fixed_when_reachable():
    """When some column meets threshold_factor * height, return fixed threshold."""
    proj = np.array([0.0, 8.0, 0.0])  # max=8, threshold=5 (0.05*100)
    t = compute_projection_threshold(proj, height=100, threshold_factor=0.05)
    assert t == pytest.approx(5.0)


def test_threshold_fallback_when_max_below_fixed():
    """All projection values below fixed threshold → fallback to percentile."""
    # height=100, fixed=5; max projection=3 → fallback to percentile of [3,3]
    proj = np.array([0.0, 3.0, 3.0, 0.0])
    t = compute_projection_threshold(proj, height=100, threshold_factor=0.05)
    assert t < 5.0
    assert t > 0.0


def test_threshold_all_zero_returns_fixed():
    """All-zero projection: no fallback data → returns fixed threshold."""
    proj = np.zeros(10)
    t = compute_projection_threshold(proj, height=100, threshold_factor=0.05)
    assert t == pytest.approx(5.0)


def test_threshold_single_nonzero_below_fixed():
    """Single nonzero value below fixed: fallback uses that value's percentile."""
    proj = np.array([0.0, 2.0, 0.0, 0.0])  # max=2, fixed=5
    t = compute_projection_threshold(proj, height=100, threshold_factor=0.05)
    # fallback_percentile=25 of [2.0] = 2.0
    assert t == pytest.approx(2.0)


# ── split_run_at_valleys ──────────────────────────────────────────────────────

def _proj_for(values: list) -> np.ndarray:
    return np.array(values, dtype=np.float64)


def test_split_no_valley_returns_unsplit():
    """All columns above valley_threshold → no valley → single range returned."""
    proj = _proj_for([10.0] * 20)
    result = split_run_at_valleys(proj, 0, 20, valley_threshold=5.0, min_segment_width=5)
    assert result == [(0, 20)]


def test_split_clear_valley_at_center():
    """Below-threshold gap in the middle splits one run into two."""
    vals = [10.0] * 8 + [0.0] * 4 + [10.0] * 8  # 20 cols; threshold=5
    proj = _proj_for(vals)
    result = split_run_at_valleys(proj, 0, 20, valley_threshold=5.0, min_segment_width=5)
    assert len(result) == 2
    left, right = result
    assert left == (0, 8)
    assert right == (12, 20)


def test_split_range_too_small_returns_unsplit():
    """Range < 2 * min_segment_width cannot be split → returned as-is."""
    proj = _proj_for([10.0, 0.0, 10.0])  # width=3, min_segment_width=5
    result = split_run_at_valleys(proj, 0, 3, valley_threshold=5.0, min_segment_width=5)
    assert result == [(0, 3)]


def test_split_all_zero_below_threshold_returns_unsplit():
    """Entire range is below threshold → no sub-range qualifies → unsplit."""
    proj = _proj_for([0.0] * 20)
    result = split_run_at_valleys(proj, 0, 20, valley_threshold=5.0, min_segment_width=5)
    assert result == [(0, 20)]


def test_split_at_threshold_not_a_valley():
    """A column with projection exactly at threshold is active, not a valley."""
    vals = [10.0] * 7 + [5.0] * 6 + [10.0] * 7  # middle cols at exactly threshold
    proj = _proj_for(vals)
    result = split_run_at_valleys(proj, 0, 20, valley_threshold=5.0, min_segment_width=5)
    assert result == [(0, 20)]


def test_split_subrange_too_narrow_discarded():
    """Left sub-range < min_segment_width → falls back to unsplit."""
    # col 3 is below threshold; left segment [0,3) width=3 < min_segment_width=5
    vals = [10.0] * 3 + [0.0] * 2 + [10.0] * 15
    proj = _proj_for(vals)
    result = split_run_at_valleys(proj, 0, 20, valley_threshold=5.0, min_segment_width=5)
    assert result == [(0, 20)]


def test_split_does_not_mutate_projection():
    proj = _proj_for([10.0] * 8 + [0.0] * 4 + [10.0] * 8)
    original = proj.copy()
    split_run_at_valleys(proj, 0, 20, valley_threshold=5.0)
    np.testing.assert_array_equal(proj, original)


# ── find_digit_boxes ──────────────────────────────────────────────────────────

def _make_binary_with_blobs(height: int, col_ranges: list) -> np.ndarray:
    """Create a binary image with foreground rectangles spanning full height."""
    binary = np.zeros((height, max(r[1] for r in col_ranges) + 1), dtype=np.uint8)
    for x1, x2 in col_ranges:
        binary[:, x1:x2] = 1
    return binary


def test_fdb_empty_binary_returns_no_boxes():
    binary = np.zeros((20, 30), dtype=np.uint8)
    proj = vertical_projection(binary)
    assert find_digit_boxes(binary, proj) == []


def test_fdb_single_blob_one_box():
    binary = _make_binary_with_blobs(20, [(5, 15)])
    proj = vertical_projection(binary)
    boxes = find_digit_boxes(binary, proj, threshold_factor=0.05, min_gap=1)
    assert len(boxes) == 1
    _, x1, _, x2 = boxes[0]
    assert x1 == 5
    assert x2 == 15


def test_fdb_two_separated_blobs_two_boxes():
    binary = _make_binary_with_blobs(20, [(2, 8), (14, 20)])
    proj = vertical_projection(binary)
    boxes = find_digit_boxes(binary, proj, threshold_factor=0.05, min_gap=1)
    assert len(boxes) == 2


def test_fdb_shallow_gap_stays_merged():
    """Two blobs connected by a moderate-projection bridge stay as one box.

    height=20, threshold_factor=0.05 → threshold=1.0.
    Bridge cols have projection=6 (>= threshold), so they are NOT valley columns.
    Gap-merging and valley-split both agree: one box.
    """
    binary = np.zeros((20, 22), dtype=np.uint8)
    binary[:, 0:8] = 1           # left blob,  proj = 20
    binary[0:6, 8:10] = 1        # bridge cols, proj = 6  (>= threshold=1.0)
    binary[:, 10:22] = 1         # right blob,  proj = 20
    proj = vertical_projection(binary)
    assert proj[8] == pytest.approx(6.0)
    boxes = find_digit_boxes(binary, proj, threshold_factor=0.05, min_gap=1,
                              min_segment_width=5)
    assert len(boxes) == 1


def test_fdb_large_gap_not_merged():
    """A 10-column gap is NOT bridged with min_gap=3 → two boxes."""
    binary = _make_binary_with_blobs(20, [(0, 8), (18, 26)])
    proj = vertical_projection(binary)
    boxes = find_digit_boxes(binary, proj, threshold_factor=0.05, min_gap=3)
    assert len(boxes) == 2


def test_fdb_adaptive_threshold_fallback():
    """When projection max < threshold_factor * height, fallback kicks in and finds boxes."""
    # height=100, threshold_factor=0.05 → fixed threshold=5
    # Place only 3 foreground rows → max projection = 3 < 5
    binary = np.zeros((100, 20), dtype=np.uint8)
    binary[0:3, 5:15] = 1
    proj = vertical_projection(binary)
    assert proj.max() < 0.05 * 100  # confirm fallback is needed
    boxes = find_digit_boxes(binary, proj, threshold_factor=0.05)
    assert len(boxes) == 1


def test_fdb_valley_split_separates_adjacent_groups():
    """Two pixel groups bridged by a large min_gap are re-split at the zero-projection valley."""
    # Two blobs far apart; min_gap=30 bridges the gap; valley split re-separates
    binary = np.zeros((20, 40), dtype=np.uint8)
    binary[:, 0:8] = 1
    binary[:, 32:40] = 1
    proj = vertical_projection(binary)
    boxes = find_digit_boxes(
        binary, proj,
        threshold_factor=0.05,
        min_gap=30,          # artificially large → one merged run
        min_segment_width=5,
    )
    assert len(boxes) == 2


def test_fdb_uniform_blob_not_split():
    """A single solid rectangle never gets false-split: all columns are above threshold."""
    binary = np.zeros((20, 20), dtype=np.uint8)
    binary[:, :] = 1
    proj = vertical_projection(binary)
    boxes = find_digit_boxes(binary, proj, threshold_factor=0.05, min_gap=1,
                              min_segment_width=5)
    assert len(boxes) == 1


def test_fdb_does_not_mutate_binary():
    binary = _make_binary_with_blobs(20, [(2, 10)])
    proj = vertical_projection(binary)
    original = binary.copy()
    find_digit_boxes(binary, proj)
    np.testing.assert_array_equal(binary, original)


def test_fdb_does_not_mutate_projection():
    binary = _make_binary_with_blobs(20, [(2, 10)])
    proj = vertical_projection(binary)
    original = proj.copy()
    find_digit_boxes(binary, proj)
    np.testing.assert_array_equal(proj, original)
