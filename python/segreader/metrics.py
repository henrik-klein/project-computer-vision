"""Accuracy metrics and diagnostics for batch evaluation against a ground-truth file.

All functions are pure (no side-effects beyond file I/O in load_ground_truth)
and operate on plain dicts so they are easy to unit-test with in-memory data.
"""
from __future__ import annotations

import json


# ── Ground-truth I/O ─────────────────────────────────────────────────────────

def load_ground_truth(path: str) -> dict[str, str]:
    """Load a ground-truth JSON file that maps stem -> expected string.

    The expected format is the project-standard samples.json:
      { "pic1": "142530", "display_spi": "005358", ... }

    Keys must match image file stems (filename without extension) exactly.
    """
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ── Per-image comparison ──────────────────────────────────────────────────────

def compare_prediction(predicted: str | None, expected: str) -> dict:
    """Compare one predicted value against its expected value.

    Returns a dict with three keys:
      expected    — the expected value (passed through unchanged)
      correct     — True iff predicted == expected (exact string match)
      char_errors — number of differing character positions (Hamming distance
                    up to the shorter string) plus the absolute length difference.
                    Zero when the strings are equal; len(expected) when
                    predicted is None (pipeline error).
    """
    if predicted is None:
        return {"expected": expected, "correct": False, "char_errors": len(expected)}
    correct = predicted == expected
    char_errors = (
        sum(1 for a, b in zip(predicted, expected) if a != b)
        + abs(len(predicted) - len(expected))
    )
    return {"expected": expected, "correct": correct, "char_errors": char_errors}


# ── Aggregate accuracy summary ────────────────────────────────────────────────

def compute_summary(results: dict, ground_truth: dict) -> dict:
    """Compute accuracy summary across all results.

    Args:
        results:      dict[stem, entry_dict] — the ``results`` section from the
                      evaluation output.  An entry with an ``"error"`` key
                      (and ``predicted=None``) represents a pipeline failure.
        ground_truth: dict[stem, expected_str] — from samples.json.

    An entry is *evaluated* when its stem exists in ground_truth.
    An entry is *correct* when predicted == expected (None is never correct).
    ``errors`` counts entries that carry an ``"error"`` key regardless of
    whether ground truth is available for them.

    Returns:
        total        — number of images processed
        evaluated    — images whose stem appears in ground_truth
        correct      — evaluated images with an exact prediction match
        incorrect    — evaluated - correct
        accuracy_pct — round(correct/evaluated*100, 1), or None if evaluated==0
        errors       — number of pipeline failures (entry has "error" key)
    """
    total = len(results)
    evaluated = 0
    correct = 0
    errors = sum(1 for entry in results.values() if entry.get("error"))

    for stem, entry in results.items():
        if stem not in ground_truth:
            continue
        evaluated += 1
        predicted = entry.get("predicted")
        if predicted is not None and predicted == ground_truth[stem]:
            correct += 1

    incorrect = evaluated - correct
    accuracy_pct = round(correct / evaluated * 100, 1) if evaluated > 0 else None

    return {
        "total": total,
        "evaluated": evaluated,
        "correct": correct,
        "incorrect": incorrect,
        "accuracy_pct": accuracy_pct,
        "errors": errors,
    }


# ── Diagnostics ───────────────────────────────────────────────────────────────

def extract_pipeline_info(steps: list) -> dict:
    """Extract compact diagnostic values from the steps list returned by run_pipeline.

    Only reads ``step["details"]`` dicts — does not touch numpy arrays.
    Returns an empty dict when steps is empty or expected step ids are absent.

    New fields (all optional; absent when the step is not present):
      fg_pixels_before_filter  — foreground pixels after morphological opening
      fg_pixels_after_filter   — foreground pixels in clean_binary
      clean_binary_empty       — True when fg_pixels_after_filter == 0
      largest_component_area   — area of the largest connected component
      largest_component_area_ratio — largest_area / (image_height * image_width)
      rejection_reason_counts  — dict mapping reason prefix to count
      component_fallback_used  — True when background-rule relaxation was applied
      localize_fallback_used   — True when localize method is "disabled" (retry path)
    """
    info: dict = {}
    for step in steps:
        d = step.get("details", {})
        sid = step.get("id")
        if sid == "localize":
            info["localize_method"] = d.get("method_used")
            info["localize_found"] = d.get("found")
            info["localize_coverage"] = d.get("coverage")
            info["localize_fallback_used"] = (d.get("method_used") == "disabled")
            info["localize_crop_w"] = d.get("crop_width")
            info["localize_crop_h"] = d.get("crop_height")
            info["localize_crop_x"] = d.get("crop_x")
            info["localize_crop_y"] = d.get("crop_y")
            info["localize_crop_aspect_ratio"] = d.get("crop_aspect_ratio")
            info["localize_shrink_applied"] = d.get("localize_shrink_applied", False)
        elif sid == "binary":
            info["fg_pixels_binary"] = d.get("foreground_pixels")
            info["polarity_inverted"] = d.get("polarity_inverted")
            info["binarization_mode"] = d.get("binarization_mode", "otsu_bright")
            info["surface_dominant"] = d.get("surface_dominant")
            cs = d.get("component_summary")
            if cs:
                info["components_after_binary"] = cs.get("count")
                info["largest_ratio_after_binary"] = cs.get("largest_ratio")
        elif sid == "opening":
            info["fg_pixels_binary"] = info.get("fg_pixels_binary") or d.get("pixels_before")
            info["fg_pixels_before_filter"] = d.get("pixels_after")
            info["morph_type_used"] = d.get("morph_type", "opening")
            cs = d.get("component_summary")
            if cs:
                info["components_after_opening"] = cs.get("count")
                info["largest_ratio_after_opening"] = cs.get("largest_ratio")
        elif sid == "components":
            info["total_components"] = d.get("total_components")
            img_h = d.get("image_height")
            img_w = d.get("image_width")
            components_list = d.get("components", [])
            if components_list:
                largest_area = max(c.get("area", 0) for c in components_list)
                info["largest_component_area"] = largest_area
                if img_h and img_w:
                    info["largest_component_area_ratio"] = round(
                        largest_area / (img_h * img_w), 4
                    )
        elif sid == "filtering":
            info["kept_components"] = d.get("kept")
            info["rejected_components"] = d.get("rejected_count")
            info["component_fallback_used"] = d.get("component_fallback_used", False)
            rejected_list = d.get("rejected", [])
            counts: dict[str, int] = {}
            for r in rejected_list:
                reason = r.get("reason", "")
                key = reason.split(":")[0].strip() if reason else "unknown"
                counts[key] = counts.get(key, 0) + 1
            if d.get("rejected_count", 0) > 0:
                info["rejection_reason_counts"] = counts
        elif sid == "clean":
            clean_px = d.get("foreground_pixels")
            info["fg_pixels_after_filter"] = clean_px
            info["clean_binary_empty"] = (clean_px == 0) if clean_px is not None else None
        elif sid == "projection":
            info["digit_boxes"] = d.get("num_digit_boxes")
        elif sid == "decoding":
            info["segment_threshold"] = d.get("segment_threshold")
    return info


def classify_error(predicted: str | None, expected: str | None) -> str:
    """Return the failure-mode category for one prediction.

    Priority order (first matching rule wins):
      "missing_ground_truth" — expected is None (no label for this image)
      "pipeline_error"       — predicted is None (exception during processing)
      "correct"              — exact string match
      "contains_unknown"     — prediction has one or more '?' characters
      "length_mismatch"      — predicted and expected have different lengths
      "wrong_digits"         — same length but at least one digit differs
    """
    if expected is None:
        return "missing_ground_truth"
    if predicted is None:
        return "pipeline_error"
    if predicted == expected:
        return "correct"
    if "?" in predicted:
        return "contains_unknown"
    if len(predicted) != len(expected):
        return "length_mismatch"
    return "wrong_digits"


def build_per_image_diagnostics(entry: dict, expected: str | None) -> dict:
    """Return new diagnostic fields to merge into a result entry.

    Does not modify ``entry``.  The returned dict is intended to be applied
    with ``entry.update(diag)``.

    Fields ``expected``, ``correct``, ``char_errors`` are NOT included here
    — they are produced by ``compare_prediction`` and already present in the
    entry when ground truth is available.

    If ``entry["pipeline_info"]`` exists (populated by extract_pipeline_info),
    its keys are included in the returned dict so the caller can pop the
    temporary key after calling ``entry.update(diag)``.
    """
    predicted = entry.get("predicted")
    diag: dict = {}

    diag["predicted_length"] = len(predicted) if predicted is not None else None

    if expected is not None:
        diag["expected_length"] = len(expected)
        diag["length_matches"] = (
            len(predicted) == len(expected) if predicted is not None else False
        )

    diag["unknown_count"] = predicted.count("?") if predicted is not None else None
    diag["has_unknown"] = bool(diag["unknown_count"]) if predicted is not None else None
    diag["error_category"] = classify_error(predicted, expected)

    # Absorb pipeline info if the entry carries it.
    diag.update(entry.get("pipeline_info", {}))

    return diag


def compute_diagnostic_summary(results: dict, ground_truth: dict) -> dict:
    """Extended summary with per-error-category breakdown and per-image averages.

    Extends ``compute_summary`` with:
      pipeline_errors, contains_unknown_count, length_mismatch_count,
      wrong_digits_count, missing_ground_truth_count,
      average_char_errors, average_unknown_count.
    """
    base = compute_summary(results, ground_truth)

    counts: dict[str, int] = {
        "correct": 0, "pipeline_error": 0, "contains_unknown": 0,
        "length_mismatch": 0, "wrong_digits": 0, "missing_ground_truth": 0,
    }
    total_char_errors = 0
    n_for_char_errors = 0
    total_unknown = 0
    n_for_unknown = 0

    for stem, entry in results.items():
        expected = ground_truth.get(stem)
        predicted = entry.get("predicted")

        cat = classify_error(predicted, expected)
        counts[cat] = counts.get(cat, 0) + 1

        if predicted is not None:
            total_unknown += predicted.count("?")
            n_for_unknown += 1
            if expected is not None:
                cmp = compare_prediction(predicted, expected)
                total_char_errors += cmp["char_errors"]
                n_for_char_errors += 1

    return {
        **base,
        "pipeline_errors": counts["pipeline_error"],
        "contains_unknown_count": counts["contains_unknown"],
        "length_mismatch_count": counts["length_mismatch"],
        "wrong_digits_count": counts["wrong_digits"],
        "missing_ground_truth_count": counts["missing_ground_truth"],
        "average_char_errors": (
            round(total_char_errors / n_for_char_errors, 2)
            if n_for_char_errors > 0 else None
        ),
        "average_unknown_count": (
            round(total_unknown / n_for_unknown, 2)
            if n_for_unknown > 0 else None
        ),
    }


# ── Threshold sweep ───────────────────────────────────────────────────────────

def parse_thresholds(s: str) -> list[float]:
    """Parse a comma-separated list of threshold floats.

    parse_thresholds("0.25,0.30,0.40") -> [0.25, 0.30, 0.40]
    Raises ValueError for non-numeric tokens or an empty/blank string.
    """
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError(f"No thresholds found in {s!r}")
    return [float(p) for p in parts]


def compute_sweep_row(results: dict, ground_truth: dict, threshold: float) -> dict:
    """Compute one summary row for a threshold-sweep table.

    Args:
        results:      dict[stem, {predicted, ...}] from a lightweight run.
        ground_truth: dict[stem, expected_str].
        threshold:    the segment threshold used for this batch.
    """
    evaluated = 0
    correct = 0
    counts: dict[str, int] = {}

    for stem, entry in results.items():
        expected = ground_truth.get(stem)
        predicted = entry.get("predicted")
        cat = classify_error(predicted, expected)
        counts[cat] = counts.get(cat, 0) + 1
        if expected is not None:
            evaluated += 1
            if predicted is not None and predicted == expected:
                correct += 1

    accuracy_pct = round(correct / evaluated * 100, 1) if evaluated > 0 else None

    return {
        "threshold": threshold,
        "correct": correct,
        "evaluated": evaluated,
        "total": len(results),
        "accuracy_pct": accuracy_pct,
        "contains_unknown_count": counts.get("contains_unknown", 0),
        "length_mismatch_count": counts.get("length_mismatch", 0),
        "wrong_digits_count": counts.get("wrong_digits", 0),
    }
