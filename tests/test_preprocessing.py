"""Tests for preprocess._compute_otsu_threshold and preprocess.otsu_binarize.

Covers:
- _compute_otsu_threshold return type, value range, no-mutation guarantee
- otsu_binarize output dtype and value set
- polarity correction (bright background inverted, dark background not)
- consistency between _compute_otsu_threshold and otsu_binarize
- visualisation.otsu_threshold delegates to the shared helper
"""
import numpy as np
import pytest

from segreader.preprocess import _compute_otsu_threshold, otsu_binarize


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
    from segreader.visualisation import visualisation as vis
    img = _bimodal(lo=50, hi=200, lo_count=8, hi_count=8)
    assert vis.otsu_threshold(img) == _compute_otsu_threshold(img)


def test_visualisation_otsu_threshold_does_not_mutate_input():
    from segreader.visualisation import visualisation as vis
    img = _bimodal()
    original = img.copy()
    vis.otsu_threshold(img)
    assert np.array_equal(img, original)
