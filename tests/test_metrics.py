"""Tests for segreader.metrics: all public functions."""
import json

import pytest

from segreader.metrics import (
    build_per_image_diagnostics,
    classify_error,
    compare_prediction,
    compute_diagnostic_summary,
    compute_summary,
    compute_sweep_row,
    extract_pipeline_info,
    load_ground_truth,
    parse_thresholds,
)


# ── load_ground_truth ─────────────────────────────────────────────────────────

def test_load_ground_truth_returns_dict(tmp_path):
    gt = {"pic1": "142530", "pic2": "2359"}
    p = tmp_path / "samples.json"
    p.write_text(json.dumps(gt), encoding="utf-8")
    assert load_ground_truth(str(p)) == gt


def test_load_ground_truth_values_are_strings(tmp_path):
    p = tmp_path / "samples.json"
    p.write_text(json.dumps({"pic1": "42"}), encoding="utf-8")
    result = load_ground_truth(str(p))
    assert isinstance(result["pic1"], str)


def test_load_ground_truth_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_ground_truth(str(tmp_path / "nonexistent.json"))


# ── compare_prediction ────────────────────────────────────────────────────────

def test_compare_exact_match():
    r = compare_prediction("142530", "142530")
    assert r["correct"] is True
    assert r["char_errors"] == 0
    assert r["expected"] == "142530"


def test_compare_single_char_wrong():
    r = compare_prediction("142531", "142530")
    assert r["correct"] is False
    assert r["char_errors"] == 1


def test_compare_all_chars_wrong():
    r = compare_prediction("000", "111")
    assert r["correct"] is False
    assert r["char_errors"] == 3


def test_compare_predicted_shorter():
    """Two matching chars + 2 missing chars in predicted → char_errors = 2."""
    r = compare_prediction("12", "1234")
    assert r["correct"] is False
    assert r["char_errors"] == 2


def test_compare_predicted_longer():
    """Extra chars in predicted also count as errors."""
    r = compare_prediction("12345", "123")
    assert r["correct"] is False
    assert r["char_errors"] == 2


def test_compare_predicted_none():
    """None prediction (pipeline error) → not correct, char_errors = len(expected)."""
    r = compare_prediction(None, "5555")
    assert r["correct"] is False
    assert r["char_errors"] == 4


def test_compare_empty_strings_match():
    r = compare_prediction("", "")
    assert r["correct"] is True
    assert r["char_errors"] == 0


def test_compare_returns_expected_field():
    r = compare_prediction("abc", "xyz")
    assert r["expected"] == "xyz"


def test_compare_colon_display_exact():
    """Values with colons (like clock displays) compare correctly."""
    r = compare_prediction("9:41", "9:41")
    assert r["correct"] is True
    assert r["char_errors"] == 0


# ── compute_summary ───────────────────────────────────────────────────────────

def _entry(predicted, error=None):
    """Build a minimal result entry dict."""
    e = {"predicted": predicted, "overlay_b64": None}
    if error:
        e["error"] = error
    return e


def test_summary_all_correct():
    results = {"p1": _entry("100"), "p2": _entry("200")}
    gt = {"p1": "100", "p2": "200"}
    s = compute_summary(results, gt)
    assert s["total"] == 2
    assert s["evaluated"] == 2
    assert s["correct"] == 2
    assert s["incorrect"] == 0
    assert s["accuracy_pct"] == 100.0
    assert s["errors"] == 0


def test_summary_partial_correct():
    results = {"p1": _entry("100"), "p2": _entry("999")}
    gt = {"p1": "100", "p2": "200"}
    s = compute_summary(results, gt)
    assert s["correct"] == 1
    assert s["incorrect"] == 1
    assert s["accuracy_pct"] == 50.0


def test_summary_missing_gt_entry_not_evaluated():
    """A result stem not in ground_truth is processed but not counted in evaluated."""
    results = {"p1": _entry("100"), "p2": _entry("200")}
    gt = {"p1": "100"}
    s = compute_summary(results, gt)
    assert s["total"] == 2
    assert s["evaluated"] == 1
    assert s["correct"] == 1
    assert s["accuracy_pct"] == 100.0


def test_summary_extra_gt_entry_ignored():
    """A gt stem with no matching result entry does not affect totals."""
    results = {"p1": _entry("100")}
    gt = {"p1": "100", "p2": "999"}
    s = compute_summary(results, gt)
    assert s["total"] == 1
    assert s["evaluated"] == 1
    assert s["correct"] == 1


def test_summary_error_entry_is_not_correct():
    """A pipeline-error entry (predicted=None) counts as evaluated but not correct."""
    results = {"p1": _entry(None, error="pipeline crashed")}
    gt = {"p1": "100"}
    s = compute_summary(results, gt)
    assert s["evaluated"] == 1
    assert s["correct"] == 0
    assert s["incorrect"] == 1
    assert s["errors"] == 1


def test_summary_errors_counted_without_gt():
    """Pipeline errors are counted even if the stem has no gt entry."""
    results = {
        "p1": _entry(None, error="fail"),
        "p2": _entry("200"),
    }
    gt = {"p2": "200"}
    s = compute_summary(results, gt)
    assert s["errors"] == 1
    assert s["evaluated"] == 1
    assert s["correct"] == 1


def test_summary_empty_results():
    s = compute_summary({}, {"p1": "100"})
    assert s["total"] == 0
    assert s["evaluated"] == 0
    assert s["accuracy_pct"] is None


def test_summary_empty_ground_truth():
    results = {"p1": _entry("100")}
    s = compute_summary(results, {})
    assert s["evaluated"] == 0
    assert s["correct"] == 0
    assert s["accuracy_pct"] is None


def test_summary_accuracy_rounding():
    """1 correct out of 3 → 33.3%."""
    results = {"p1": _entry("a"), "p2": _entry("b"), "p3": _entry("c")}
    gt = {"p1": "a", "p2": "x", "p3": "x"}
    s = compute_summary(results, gt)
    assert s["accuracy_pct"] == pytest.approx(33.3, abs=0.1)


# ── extract_pipeline_info ─────────────────────────────────────────────────────

def _make_steps():
    """Minimal steps list that mirrors what run_pipeline produces."""
    return [
        {"id": "localize",   "details": {"method_used": "color", "found": True, "coverage": 0.42}},
        {"id": "components", "details": {"total_components": 12}},
        {"id": "filtering",  "details": {"kept": 10, "rejected_count": 2}},
        {"id": "projection", "details": {"num_digit_boxes": 4}},
        {"id": "decoding",   "details": {"segment_threshold": 0.40, "num_digits": 4}},
    ]


def test_extract_pipeline_info_localize_fields():
    info = extract_pipeline_info(_make_steps())
    assert info["localize_method"] == "color"
    assert info["localize_found"] is True
    assert info["localize_coverage"] == pytest.approx(0.42)


def test_extract_pipeline_info_component_counts():
    info = extract_pipeline_info(_make_steps())
    assert info["total_components"] == 12
    assert info["kept_components"] == 10
    assert info["rejected_components"] == 2


def test_extract_pipeline_info_digit_boxes():
    info = extract_pipeline_info(_make_steps())
    assert info["digit_boxes"] == 4


def test_extract_pipeline_info_segment_threshold():
    info = extract_pipeline_info(_make_steps())
    assert info["segment_threshold"] == pytest.approx(0.40)


def test_extract_pipeline_info_empty_steps():
    info = extract_pipeline_info([])
    assert info == {}


def test_extract_pipeline_info_ignores_unknown_step_ids():
    steps = [{"id": "unknown_step", "details": {"foo": "bar"}}]
    info = extract_pipeline_info(steps)
    assert "foo" not in info


def test_extract_pipeline_info_missing_details_key():
    steps = [{"id": "localize"}]  # no "details" key
    info = extract_pipeline_info(steps)
    assert info.get("localize_method") is None


def _make_full_steps():
    """Richer steps that include all new step ids (opening, clean, localize-disabled)."""
    return [
        {
            "id": "localize",
            "details": {"method_used": "disabled", "found": False, "coverage": 1.0},
        },
        {
            "id": "opening",
            "details": {"pixels_before": 500, "pixels_after": 420, "pixels_removed": 80},
        },
        {
            "id": "components",
            "details": {
                "total_components": 3,
                "image_height": 50,
                "image_width": 100,
                "components": [
                    {"label": 1, "area": 800, "bbox": [0, 0, 50, 60], "aspect_ratio": 1.2, "kept": False},
                    {"label": 2, "area": 200, "bbox": [0, 70, 50, 90], "aspect_ratio": 1.0, "kept": True},
                    {"label": 3, "area": 50,  "bbox": [0, 95, 50, 100], "aspect_ratio": 1.5, "kept": True},
                ],
            },
        },
        {
            "id": "filtering",
            "details": {
                "total": 3, "kept": 2, "rejected_count": 1,
                "component_fallback_used": False,
                "rejected": [
                    {"label": 1, "area": 800, "aspect_ratio": 1.2,
                     "reason": "background: area 800 > 750px (15% of image)"},
                ],
            },
        },
        {
            "id": "clean",
            "details": {"foreground_pixels": 250, "foreground_ratio": 0.05},
        },
        {
            "id": "projection",
            "details": {"num_digit_boxes": 2},
        },
        {
            "id": "decoding",
            "details": {"segment_threshold": 0.35, "num_digits": 2},
        },
    ]


def test_extract_localize_fallback_used_true():
    """method_used='disabled' → localize_fallback_used is True."""
    info = extract_pipeline_info(_make_full_steps())
    assert info["localize_fallback_used"] is True


def test_extract_localize_fallback_used_false():
    """method_used='color' → localize_fallback_used is False."""
    info = extract_pipeline_info(_make_steps())
    assert info["localize_fallback_used"] is False


def test_extract_fg_pixels_before_filter():
    info = extract_pipeline_info(_make_full_steps())
    assert info["fg_pixels_before_filter"] == 420


def test_extract_fg_pixels_after_filter():
    info = extract_pipeline_info(_make_full_steps())
    assert info["fg_pixels_after_filter"] == 250


def test_extract_clean_binary_empty_false():
    info = extract_pipeline_info(_make_full_steps())
    assert info["clean_binary_empty"] is False


def test_extract_clean_binary_empty_true():
    steps = [{"id": "clean", "details": {"foreground_pixels": 0, "foreground_ratio": 0.0}}]
    info = extract_pipeline_info(steps)
    assert info["clean_binary_empty"] is True


def test_extract_largest_component_area():
    info = extract_pipeline_info(_make_full_steps())
    assert info["largest_component_area"] == 800


def test_extract_largest_component_area_ratio():
    # image_height=50, image_width=100, largest_area=800 → ratio = 800/5000 = 0.16
    info = extract_pipeline_info(_make_full_steps())
    assert info["largest_component_area_ratio"] == pytest.approx(0.16)


def test_extract_rejection_reason_counts():
    info = extract_pipeline_info(_make_full_steps())
    assert info["rejection_reason_counts"] == {"background": 1}


def test_extract_rejection_reason_counts_multiple():
    steps = [
        {"id": "filtering", "details": {
            "total": 4, "kept": 1, "rejected_count": 3,
            "component_fallback_used": False,
            "rejected": [
                {"label": 1, "reason": "noise: area 5 < 30"},
                {"label": 2, "reason": "noise: area 8 < 30"},
                {"label": 3, "reason": "background: area 2000 > 1500px (15%)"},
            ],
        }},
    ]
    info = extract_pipeline_info(steps)
    assert info["rejection_reason_counts"]["noise"] == 2
    assert info["rejection_reason_counts"]["background"] == 1


def test_extract_component_fallback_used_false():
    info = extract_pipeline_info(_make_full_steps())
    assert info["component_fallback_used"] is False


def test_extract_component_fallback_used_true():
    steps = [
        {"id": "filtering", "details": {
            "total": 1, "kept": 1, "rejected_count": 0,
            "component_fallback_used": True, "rejected": [],
        }},
    ]
    info = extract_pipeline_info(steps)
    assert info["component_fallback_used"] is True


def test_extract_no_rejection_counts_when_none_rejected():
    """When rejected_count=0, rejection_reason_counts is not added to info."""
    steps = [
        {"id": "filtering", "details": {
            "total": 2, "kept": 2, "rejected_count": 0,
            "component_fallback_used": False, "rejected": [],
        }},
    ]
    info = extract_pipeline_info(steps)
    assert "rejection_reason_counts" not in info


# ── classify_error ────────────────────────────────────────────────────────────

def test_classify_error_correct():
    assert classify_error("123", "123") == "correct"


def test_classify_error_pipeline_error():
    assert classify_error(None, "123") == "pipeline_error"


def test_classify_error_missing_ground_truth():
    assert classify_error("123", None) == "missing_ground_truth"


def test_classify_error_contains_unknown():
    assert classify_error("1?3", "123") == "contains_unknown"


def test_classify_error_length_mismatch():
    assert classify_error("12", "1234") == "length_mismatch"


def test_classify_error_wrong_digits():
    assert classify_error("129", "123") == "wrong_digits"


def test_classify_error_missing_gt_takes_priority_over_none():
    """missing_ground_truth fires before pipeline_error."""
    assert classify_error(None, None) == "missing_ground_truth"


def test_classify_error_contains_unknown_takes_priority_over_length():
    """'?' check fires before length check so we see the dominant problem."""
    # "??" vs "1234" — has unknowns AND wrong length
    assert classify_error("??", "1234") == "contains_unknown"


# ── build_per_image_diagnostics ───────────────────────────────────────────────

def test_diag_correct_prediction():
    entry = {"predicted": "123"}
    d = build_per_image_diagnostics(entry, "123")
    assert d["predicted_length"] == 3
    assert d["expected_length"] == 3
    assert d["length_matches"] is True
    assert d["unknown_count"] == 0
    assert d["has_unknown"] is False
    assert d["error_category"] == "correct"


def test_diag_contains_unknown():
    entry = {"predicted": "1?3"}
    d = build_per_image_diagnostics(entry, "123")
    assert d["unknown_count"] == 1
    assert d["has_unknown"] is True
    assert d["error_category"] == "contains_unknown"


def test_diag_length_mismatch():
    entry = {"predicted": "12"}
    d = build_per_image_diagnostics(entry, "1234")
    assert d["length_matches"] is False
    assert d["error_category"] == "length_mismatch"


def test_diag_wrong_digits():
    entry = {"predicted": "129"}
    d = build_per_image_diagnostics(entry, "123")
    assert d["length_matches"] is True
    assert d["error_category"] == "wrong_digits"


def test_diag_pipeline_error():
    entry = {"predicted": None}
    d = build_per_image_diagnostics(entry, "123")
    assert d["predicted_length"] is None
    assert d["unknown_count"] is None
    assert d["has_unknown"] is None
    assert d["error_category"] == "pipeline_error"


def test_diag_missing_ground_truth():
    entry = {"predicted": "123"}
    d = build_per_image_diagnostics(entry, None)
    assert "expected_length" not in d
    assert "length_matches" not in d
    assert d["error_category"] == "missing_ground_truth"


def test_diag_pipeline_info_merged():
    """pipeline_info dict is flattened into the returned diag dict."""
    entry = {
        "predicted": "42",
        "pipeline_info": {"localize_method": "color", "digit_boxes": 2},
    }
    d = build_per_image_diagnostics(entry, "42")
    assert d["localize_method"] == "color"
    assert d["digit_boxes"] == 2


def test_diag_does_not_modify_entry():
    entry = {"predicted": "123"}
    original = dict(entry)
    build_per_image_diagnostics(entry, "123")
    assert entry == original


# ── compute_diagnostic_summary ────────────────────────────────────────────────

def test_diag_summary_category_counts():
    results = {
        "a": _entry("100"),        # correct
        "b": _entry("?00"),        # contains_unknown
        "c": _entry("99"),         # length_mismatch (expected "100")
        "d": _entry("101"),        # wrong_digits
        "e": _entry(None, "fail"), # pipeline_error
        "f": _entry("555"),        # missing_ground_truth (not in gt)
    }
    gt = {"a": "100", "b": "100", "c": "100", "d": "100", "e": "100"}
    s = compute_diagnostic_summary(results, gt)
    assert s["correct"] == 1
    assert s["contains_unknown_count"] == 1
    assert s["length_mismatch_count"] == 1
    assert s["wrong_digits_count"] == 1
    assert s["pipeline_errors"] == 1
    assert s["missing_ground_truth_count"] == 1


def test_diag_summary_average_char_errors():
    results = {
        "a": _entry("100"),  # 0 errors
        "b": _entry("199"),  # 2 errors (positions 1,2)
    }
    gt = {"a": "100", "b": "100"}
    s = compute_diagnostic_summary(results, gt)
    assert s["average_char_errors"] == pytest.approx(1.0)


def test_diag_summary_average_unknown_count():
    results = {
        "a": _entry("?00"),   # 1 unknown
        "b": _entry("???"),   # 3 unknowns
    }
    gt = {"a": "100", "b": "100"}
    s = compute_diagnostic_summary(results, gt)
    assert s["average_unknown_count"] == pytest.approx(2.0)


def test_diag_summary_empty():
    s = compute_diagnostic_summary({}, {})
    assert s["accuracy_pct"] is None
    assert s["average_char_errors"] is None
    assert s["average_unknown_count"] is None


# ── parse_thresholds ──────────────────────────────────────────────────────────

def test_parse_thresholds_multiple():
    assert parse_thresholds("0.25,0.30,0.40") == pytest.approx([0.25, 0.30, 0.40])


def test_parse_thresholds_single():
    assert parse_thresholds("0.35") == pytest.approx([0.35])


def test_parse_thresholds_with_spaces():
    assert parse_thresholds(" 0.25 , 0.50 ") == pytest.approx([0.25, 0.50])


def test_parse_thresholds_empty_raises():
    with pytest.raises(ValueError):
        parse_thresholds("")


def test_parse_thresholds_blank_raises():
    with pytest.raises(ValueError):
        parse_thresholds("   ")


def test_parse_thresholds_invalid_raises():
    with pytest.raises(ValueError):
        parse_thresholds("0.25,abc")


# ── compute_sweep_row ─────────────────────────────────────────────────────────

def test_sweep_row_all_correct():
    results = {"p1": _entry("100"), "p2": _entry("200")}
    gt = {"p1": "100", "p2": "200"}
    r = compute_sweep_row(results, gt, 0.30)
    assert r["threshold"] == pytest.approx(0.30)
    assert r["correct"] == 2
    assert r["evaluated"] == 2
    assert r["accuracy_pct"] == 100.0


def test_sweep_row_partial():
    results = {"p1": _entry("100"), "p2": _entry("?00")}
    gt = {"p1": "100", "p2": "100"}
    r = compute_sweep_row(results, gt, 0.40)
    assert r["correct"] == 1
    assert r["contains_unknown_count"] == 1
    assert r["accuracy_pct"] == 50.0


def test_sweep_row_unknown_threshold_field():
    r = compute_sweep_row({}, {}, 0.99)
    assert r["threshold"] == pytest.approx(0.99)
    assert r["accuracy_pct"] is None


def test_sweep_row_length_mismatch_counted():
    results = {"p1": _entry("12")}
    gt = {"p1": "1234"}
    r = compute_sweep_row(results, gt, 0.35)
    assert r["length_mismatch_count"] == 1
    assert r["correct"] == 0
