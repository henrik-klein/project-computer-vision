"""Equivalence tests: Rust label_components vs. frozen Python reference.

Labels are defined only up to permutation — pixel (0,0) might get label 1 in
Python and label 3 in Rust.  Two arrays are *equivalent* iff there is a
bijection between nonzero labels such that the induced partition of foreground
pixels is identical.

If segreader_native is not installed, all tests are skipped gracefully.
"""
from __future__ import annotations

import numpy as np
import pytest

segreader_native = pytest.importorskip("segreader_native")

from segreader.labeling import label_components as py_label_components  # noqa: E402


# ── Helper ────────────────────────────────────────────────────────────────────

def assert_same_partition(a: np.ndarray, b: np.ndarray) -> None:
    """Assert that two label arrays encode the same pixel partition.

    Rules:
      * Background (label == 0) must agree exactly.
      * Any two pixels that share a label in ``a`` must share a label in ``b``,
        and vice-versa.
      * There must be a bijection between the nonzero label sets.

    Raises AssertionError with a diagnostic message on failure.
    """
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
    assert a.dtype == np.int32, f"Expected int32, got {a.dtype}"
    assert b.dtype == np.int32, f"Expected int32, got {b.dtype}"

    # Background pixels must match.
    bg_a = a == 0
    bg_b = b == 0
    if not np.array_equal(bg_a, bg_b):
        diff = int((bg_a != bg_b).sum())
        raise AssertionError(
            f"Background mismatch: {diff} pixel(s) differ between the two arrays."
        )

    # For foreground pixels, build label-to-label mapping in both directions.
    a_to_b: dict[int, int] = {}
    b_to_a: dict[int, int] = {}

    if a.size == 0:
        return  # empty arrays trivially equivalent

    it = np.nditer([a, b])
    for la, lb in it:
        la_i, lb_i = int(la), int(lb)
        if la_i == 0:
            continue  # background already verified above
        # Check a→b consistency.
        if la_i in a_to_b:
            if a_to_b[la_i] != lb_i:
                raise AssertionError(
                    f"Label {la_i} in array-a maps to both {a_to_b[la_i]} and "
                    f"{lb_i} in array-b — not a valid bijection."
                )
        else:
            a_to_b[la_i] = lb_i
        # Check b→a consistency.
        if lb_i in b_to_a:
            if b_to_a[lb_i] != la_i:
                raise AssertionError(
                    f"Label {lb_i} in array-b maps to both {b_to_a[lb_i]} and "
                    f"{la_i} in array-a — not a valid bijection."
                )
        else:
            b_to_a[lb_i] = la_i


def _rust(binary: np.ndarray) -> np.ndarray:
    return segreader_native.label_components(binary)


def _py(binary: np.ndarray) -> np.ndarray:
    return py_label_components(binary)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_empty_image():
    binary = np.zeros((0, 0), dtype=np.uint8)
    r = _rust(binary)
    p = _py(binary)
    assert_same_partition(r, p)


def test_all_background():
    binary = np.zeros((5, 7), dtype=np.uint8)
    assert_same_partition(_rust(binary), _py(binary))


def test_single_foreground_pixel():
    binary = np.zeros((4, 4), dtype=np.uint8)
    binary[2, 2] = 1
    assert_same_partition(_rust(binary), _py(binary))


def test_full_image_one_component():
    binary = np.ones((6, 8), dtype=np.uint8)
    r = _rust(binary)
    p = _py(binary)
    assert_same_partition(r, p)
    # All foreground pixels must share the same label.
    assert len(np.unique(r[r != 0])) == 1
    assert len(np.unique(p[p != 0])) == 1


def test_two_separated_components():
    binary = np.zeros((1, 5), dtype=np.uint8)
    binary[0, 0] = 1
    binary[0, 4] = 1
    r = _rust(binary)
    p = _py(binary)
    assert_same_partition(r, p)
    # Exactly two distinct nonzero labels in each result.
    assert len(np.unique(r[r != 0])) == 2
    assert len(np.unique(p[p != 0])) == 2


def test_four_isolated_corners():
    binary = np.zeros((3, 3), dtype=np.uint8)
    binary[0, 0] = 1
    binary[0, 2] = 1
    binary[2, 0] = 1
    binary[2, 2] = 1
    r = _rust(binary)
    p = _py(binary)
    assert_same_partition(r, p)
    assert len(np.unique(r[r != 0])) == 4


def test_diagonal_not_connected():
    """Pixels that touch only diagonally must be separate components."""
    binary = np.zeros((2, 2), dtype=np.uint8)
    binary[0, 0] = 1
    binary[1, 1] = 1
    r = _rust(binary)
    p = _py(binary)
    assert_same_partition(r, p)
    assert r[0, 0] != r[1, 1]
    assert r[0, 1] == 0
    assert r[1, 0] == 0


def test_l_shape_union():
    """L-shape exercises the union-merge path (west and north neighbors)."""
    binary = np.array([
        [1, 0],
        [1, 0],
        [1, 1],
    ], dtype=np.uint8)
    r = _rust(binary)
    p = _py(binary)
    assert_same_partition(r, p)
    lbl = r[0, 0]
    assert r[1, 0] == lbl
    assert r[2, 0] == lbl
    assert r[2, 1] == lbl
    assert r[0, 1] == 0


def test_u_shape_union():
    """U-shape — all three arms must merge into one component."""
    binary = np.array([
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
    ], dtype=np.uint8)
    r = _rust(binary)
    p = _py(binary)
    assert_same_partition(r, p)
    assert len(np.unique(r[r != 0])) == 1


def test_horizontal_strip():
    binary = np.zeros((5, 10), dtype=np.uint8)
    binary[2, :] = 1
    assert_same_partition(_rust(binary), _py(binary))


def test_vertical_strip():
    binary = np.zeros((10, 5), dtype=np.uint8)
    binary[:, 2] = 1
    assert_same_partition(_rust(binary), _py(binary))


def test_multiple_rows_with_gaps():
    """Three horizontal strips separated by empty rows."""
    binary = np.zeros((7, 5), dtype=np.uint8)
    binary[0, :] = 1
    binary[3, :] = 1
    binary[6, :] = 1
    r = _rust(binary)
    p = _py(binary)
    assert_same_partition(r, p)
    assert len(np.unique(r[r != 0])) == 3


def test_output_dtype_is_int32():
    binary = np.ones((3, 3), dtype=np.uint8)
    r = _rust(binary)
    assert r.dtype == np.int32, f"Expected int32, got {r.dtype}"


def test_output_shape_matches_input():
    binary = np.zeros((7, 11), dtype=np.uint8)
    binary[3, 5] = 1
    r = _rust(binary)
    assert r.shape == binary.shape


def test_real_pipeline_binary():
    """Run the actual Python pipeline to get a real binary image, then compare."""
    import sys
    import os
    # Find a real test image
    img_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "testing", "images",
    )
    # Use first available image
    candidates = [
        f for f in os.listdir(img_dir)
        if f.endswith((".png", ".jpg", ".jpeg"))
    ] if os.path.isdir(img_dir) else []

    if not candidates:
        pytest.skip("No test images found")

    import cv2
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
    from segreader.preprocess import to_grayscale, adjust_contrast, otsu_binarize
    from segreader.morphology import make_rect_kernel, apply_variant

    img_path = os.path.join(img_dir, sorted(candidates)[0])
    img = cv2.imread(img_path)
    if img is None:
        pytest.skip(f"Could not load {img_path}")
    if img.shape[1] > 800:
        scale = 800 / img.shape[1]
        img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)),
                         interpolation=cv2.INTER_AREA)

    gray = to_grayscale(img)
    gray = adjust_contrast(gray)
    binary = otsu_binarize(gray)
    kernel = make_rect_kernel(3, 3)
    binary = apply_variant(binary, "opening", kernel)

    r = _rust(binary)
    p = _py(binary)
    assert_same_partition(r, p)
