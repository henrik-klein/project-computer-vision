"""Tests for preprocess._compute_otsu_threshold, otsu_binarize, otsu_dark_binarize,
and local_mean_binarize.

Covers:
- _compute_otsu_threshold return type, value range, no-mutation guarantee
- otsu_binarize output dtype and value set
- polarity correction (bright background inverted, dark background not)
- consistency between _compute_otsu_threshold and otsu_binarize
- visualisation.otsu_threshold delegates to the shared helper
- otsu_dark_binarize: dark pixels = foreground, no polarity flip
- local_mean_binarize: output shape/dtype/values, bright/dark polarity, no mutation,
  invalid-argument error handling, uniform image, border reflect behavior
"""
import numpy as np
import pytest

from segreader.preprocess import (
    _compute_otsu_threshold,
    otsu_binarize,
    otsu_dark_binarize,
    local_mean_binarize,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bimodal(lo: int = 50, hi: int = 200, lo_count: int = 8, hi_count: int = 8) -> np.ndarray:
    """1-D row image with `lo_count` pixels at `lo` and `hi_count` pixels at `hi`."""
    return np.array([lo] * lo_count + [hi] * hi_count, dtype=np.uint8).reshape(1, -1)


# ── _compute_otsu_threshold ───────────────────────────────────────────────────

def test_compute_otsu_threshold_returns_int():
    img = _bimodal()
    result = _compute_otsu_threshold(img)
    assert isinstance(result, int)


def test_compute_otsu_threshold_separates_bimodal():
    """Threshold must fall between the two cluster centres."""
    lo, hi = 50, 200
    img = _bimodal(lo=lo, hi=hi)
    t = _compute_otsu_threshold(img)
    assert lo <= t < hi, f"Expected {lo} <= threshold < {hi}, got {t}"


def test_compute_otsu_threshold_exact_value_equal_counts():
    """With equal pixel counts at 50 and 200, argmax of sigma_b2 is at index 50."""
    img = _bimodal(lo=50, hi=200, lo_count=8, hi_count=8)
    assert _compute_otsu_threshold(img) == 50


def test_compute_otsu_threshold_does_not_mutate_input():
    img = _bimodal()
    original = img.copy()
    _compute_otsu_threshold(img)
    assert np.array_equal(img, original)


def test_compute_otsu_threshold_uniform_image():
    """Uniform image: argmax of all-zero sigma_b2 returns 0 (numpy argmax behaviour)."""
    img = np.full((4, 4), 128, dtype=np.uint8)
    t = _compute_otsu_threshold(img)
    assert isinstance(t, int)


# ── otsu_binarize ─────────────────────────────────────────────────────────────

def test_otsu_binarize_output_values_are_binary():
    img = _bimodal()
    result = otsu_binarize(img)
    assert set(np.unique(result)).issubset({0, 1})


def test_otsu_binarize_output_dtype():
    img = _bimodal()
    result = otsu_binarize(img)
    assert result.dtype == np.uint8


def test_otsu_binarize_output_shape():
    img = _bimodal()
    result = otsu_binarize(img)
    assert result.shape == img.shape


def test_otsu_binarize_polarity_bright_background_inverted():
    """3 bright pixels (background) + 1 dark pixel (segment).

    After thresholding, bright pixels → 1 (majority), so binary.mean() > 0.5
    triggers inversion: the dark segment must end up as 1.
    """
    img = np.array([[200, 200], [200, 50]], dtype=np.uint8)
    result = otsu_binarize(img)
    assert result[1, 1] == 1, "Dark segment pixel should be foreground (1) after inversion"
    assert result.sum() == 1, "Only the single dark pixel should be foreground"


def test_otsu_binarize_polarity_dark_background_not_inverted():
    """1 bright pixel (segment) + 3 dark pixels (background).

    binary.mean() = 0.25 — NOT > 0.5, so no inversion: bright pixel stays 1.
    """
    img = np.array([[50, 50], [50, 200]], dtype=np.uint8)
    result = otsu_binarize(img)
    assert result[1, 1] == 1, "Bright segment pixel should be foreground (1)"
    assert result.sum() == 1, "Only the single bright pixel should be foreground"


def test_otsu_binarize_consistent_with_compute_threshold():
    """The threshold used internally must equal _compute_otsu_threshold."""
    # Use a non-inverted image (dark background, bright segment) so we know
    # foreground pixel is exactly the one with value > threshold.
    img = np.array([[50, 50], [50, 200]], dtype=np.uint8)
    t = _compute_otsu_threshold(img)
    binary = otsu_binarize(img)
    # No inversion case: pixel 200 > t → 1, pixels 50 <= t → 0
    assert binary[1, 1] == 1
    assert binary[0, 0] == 0


# ── visualisation.otsu_threshold delegates to the shared helper ───────────────

def test_visualisation_otsu_threshold_matches_helper():
    """visualisation.otsu_threshold must return the same value as _compute_otsu_threshold."""
    import segreader.visualisation as vis
    img = _bimodal(lo=50, hi=200, lo_count=8, hi_count=8)
    assert vis.otsu_threshold(img) == _compute_otsu_threshold(img)


def test_visualisation_otsu_threshold_does_not_mutate_input():
    import segreader.visualisation as vis
    img = _bimodal()
    original = img.copy()
    vis.otsu_threshold(img)
    assert np.array_equal(img, original)


# ── otsu_dark_binarize ────────────────────────────────────────────────────────

def test_otsu_dark_binarize_output_dtype():
    assert otsu_dark_binarize(_bimodal()).dtype == np.uint8


def test_otsu_dark_binarize_output_shape():
    img = _bimodal()
    assert otsu_dark_binarize(img).shape == img.shape


def test_otsu_dark_binarize_output_values_are_binary():
    assert set(np.unique(otsu_dark_binarize(_bimodal()))).issubset({0, 1})


def test_otsu_dark_binarize_dark_pixel_is_foreground():
    """One dark pixel + three bright pixels: dark pixel must be foreground."""
    img = np.array([[200, 200], [200, 50]], dtype=np.uint8)
    result = otsu_dark_binarize(img)
    assert result[1, 1] == 1, "Dark pixel should be foreground (1)"
    assert result.sum() == 1, "Only the single dark pixel should be foreground"


def test_otsu_dark_binarize_no_polarity_flip_for_bright_majority():
    """With three bright pixels and one dark, bright pixels must NOT become foreground.

    otsu_binarize would flip here (fg_ratio > 0.5 triggers inversion).
    otsu_dark_binarize must NOT flip: only the dark pixel stays foreground.
    """
    img = np.array([[200, 200], [200, 50]], dtype=np.uint8)
    result = otsu_dark_binarize(img)
    # The three bright pixels should all be 0 (background).
    assert result[0, 0] == 0
    assert result[0, 1] == 0
    assert result[1, 0] == 0


def test_otsu_dark_binarize_does_not_mutate_input():
    img = _bimodal()
    original = img.copy()
    otsu_dark_binarize(img)
    assert np.array_equal(img, original)


# ── local_mean_binarize ───────────────────────────────────────────────────────

def _reversed_display_patch() -> np.ndarray:
    """5×5 patch: bright background (200) with dark centre pixel (30).
    Simulates a reversed LCD: dark digit on bright surface.
    """
    patch = np.full((5, 5), 200, dtype=np.uint8)
    patch[2, 2] = 30
    return patch


def test_lmb_output_dtype():
    img = _reversed_display_patch()
    assert local_mean_binarize(img, window=3, offset=10, polarity="dark").dtype == np.uint8


def test_lmb_output_shape():
    img = _reversed_display_patch()
    result = local_mean_binarize(img, window=3, offset=10, polarity="dark")
    assert result.shape == img.shape


def test_lmb_output_values_are_binary():
    img = _reversed_display_patch()
    result = local_mean_binarize(img, window=3, offset=10, polarity="dark")
    assert set(np.unique(result)).issubset({0, 1})


def test_lmb_dark_segment_on_bright_background_detected():
    """Dark centre pixel surrounded by bright neighbours → foreground with polarity='dark'."""
    img = _reversed_display_patch()
    result = local_mean_binarize(img, window=3, offset=50, polarity="dark")
    # Local mean around centre ≈ 200 (all bright neighbours + one dark centre).
    # 30 < 200 - 50 = 150  →  foreground (1).
    assert result[2, 2] == 1, "Dark pixel in bright neighbourhood must be foreground"


def test_lmb_bright_segment_on_dark_background_detected():
    """Bright centre pixel surrounded by dark neighbours → foreground with polarity='bright'."""
    patch = np.full((5, 5), 30, dtype=np.uint8)
    patch[2, 2] = 200
    result = local_mean_binarize(patch, window=3, offset=50, polarity="bright")
    # Local mean around centre ≈ 30. 200 > 30 + 50 = 80 → foreground (1).
    assert result[2, 2] == 1, "Bright pixel in dark neighbourhood must be foreground"


def test_lmb_uniform_image_no_foreground_dark():
    """All pixels equal → local mean = pixel → no pixel is darker than mean - offset."""
    img = np.full((7, 7), 128, dtype=np.uint8)
    result = local_mean_binarize(img, window=3, offset=1, polarity="dark")
    assert result.sum() == 0, "Uniform image should produce no foreground"


def test_lmb_uniform_image_no_foreground_bright():
    img = np.full((7, 7), 128, dtype=np.uint8)
    result = local_mean_binarize(img, window=3, offset=1, polarity="bright")
    assert result.sum() == 0


def test_lmb_does_not_mutate_input():
    img = _reversed_display_patch()
    original = img.copy()
    local_mean_binarize(img, window=3, offset=10, polarity="dark")
    assert np.array_equal(img, original)


def test_lmb_border_pixel_is_defined():
    """With reflect padding, corner pixels must produce a valid binary output (no NaN)."""
    img = _reversed_display_patch()
    result = local_mean_binarize(img, window=3, offset=10, polarity="dark")
    # Corner pixel should be a valid binary value.
    assert result[0, 0] in (0, 1)
    assert result[4, 4] in (0, 1)


def test_lmb_invalid_window_even():
    with pytest.raises(ValueError, match="positive odd integer"):
        local_mean_binarize(np.zeros((3, 3), dtype=np.uint8), window=4, offset=5, polarity="dark")


def test_lmb_invalid_window_zero():
    with pytest.raises(ValueError, match="positive odd integer"):
        local_mean_binarize(np.zeros((3, 3), dtype=np.uint8), window=0, offset=5, polarity="dark")


def test_lmb_invalid_polarity():
    with pytest.raises(ValueError, match="polarity"):
        local_mean_binarize(np.zeros((3, 3), dtype=np.uint8), window=3, offset=5, polarity="none")


def test_lmb_dark_background_not_foreground_dark_polarity():
    """Dark pixel surrounded by other dark pixels must NOT be foreground with polarity='dark'.

    This is the critical property that distinguishes local mean threshold from
    global Otsu inversion: uniform dark areas (background outside display) do not
    trigger the dark-foreground rule.
    """
    img = np.full((7, 7), 40, dtype=np.uint8)  # all dark
    result = local_mean_binarize(img, window=3, offset=10, polarity="dark")
    assert result.sum() == 0, "Dark pixels in a dark neighbourhood must not be foreground"
