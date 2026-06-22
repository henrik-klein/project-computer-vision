"""Tests for morphology.erode / dilate / opening / closing / make_rect_kernel.

All expected outputs are hand-computed from the algorithm's definition:
  erode  = AND of all (cy-r, cx-c)-shifted copies of binary
  dilate = OR  of all (cy-r, cx-c)-shifted copies of binary
  opening  = dilate(erode(B, K), K)
  closing  = erode(dilate(B, K), K)

No cv2, no scipy — NumPy assertions only.
"""
import numpy as np
import pytest

from segreader.morphology import make_rect_kernel, erode, dilate, opening, closing, apply_variant


# ── make_rect_kernel ──────────────────────────────────────────────────────────

def test_make_rect_kernel_shape():
    k = make_rect_kernel(3, 5)
    assert k.shape == (3, 5)


def test_make_rect_kernel_dtype():
    k = make_rect_kernel(3, 3)
    assert k.dtype == np.uint8


def test_make_rect_kernel_all_ones():
    k = make_rect_kernel(4, 4)
    assert k.min() == 1 and k.max() == 1


def test_make_rect_kernel_1x1():
    k = make_rect_kernel(1, 1)
    np.testing.assert_array_equal(k, np.array([[1]], dtype=np.uint8))


# ── erode ─────────────────────────────────────────────────────────────────────

def test_erode_shrinks_block_by_one():
    """5×5 all-ones eroded by 3×3: border pixels removed, 3×3 interior survives."""
    binary = np.ones((5, 5), dtype=np.uint8)
    result = erode(binary, make_rect_kernel(3, 3))
    expected = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.uint8)
    np.testing.assert_array_equal(result, expected)


def test_erode_isolated_pixel_disappears():
    """A lone pixel has no 3×3 block of 1s around it, so erosion removes it."""
    binary = np.zeros((5, 5), dtype=np.uint8)
    binary[2, 2] = 1
    result = erode(binary, make_rect_kernel(3, 3))
    np.testing.assert_array_equal(result, np.zeros((5, 5), dtype=np.uint8))


def test_erode_identity_kernel():
    """1×1 kernel is the identity: erode must return an equal array."""
    binary = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=np.uint8)
    result = erode(binary, make_rect_kernel(1, 1))
    np.testing.assert_array_equal(result, binary)


def test_erode_two_iterations():
    """Eroding a 7×7 block twice with a 3×3 kernel shrinks by one pixel on each
    side per iteration: after two passes the inner 3×3 survives."""
    binary = np.ones((7, 7), dtype=np.uint8)
    k = make_rect_kernel(3, 3)
    result = erode(erode(binary, k), k)
    expected = np.zeros((7, 7), dtype=np.uint8)
    expected[2:5, 2:5] = 1
    np.testing.assert_array_equal(result, expected)


def test_erode_does_not_mutate_input():
    binary = np.ones((5, 5), dtype=np.uint8)
    original = binary.copy()
    erode(binary, make_rect_kernel(3, 3))
    np.testing.assert_array_equal(binary, original)


# ── dilate ────────────────────────────────────────────────────────────────────

def test_dilate_single_center_pixel_expands():
    """A single pixel at the centre of a 5×5 grid expands to a 3×3 block."""
    binary = np.zeros((5, 5), dtype=np.uint8)
    binary[2, 2] = 1
    result = dilate(binary, make_rect_kernel(3, 3))
    expected = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.uint8)
    np.testing.assert_array_equal(result, expected)


def test_dilate_corner_pixel_clipped():
    """A pixel at (0,0) can only expand into the image — clipped at the border."""
    binary = np.zeros((5, 5), dtype=np.uint8)
    binary[0, 0] = 1
    result = dilate(binary, make_rect_kernel(3, 3))
    expected = np.zeros((5, 5), dtype=np.uint8)
    expected[0:2, 0:2] = 1
    np.testing.assert_array_equal(result, expected)


def test_dilate_all_ones_stays_all_ones():
    """Dilating an already-full array changes nothing."""
    binary = np.ones((5, 5), dtype=np.uint8)
    result = dilate(binary, make_rect_kernel(3, 3))
    np.testing.assert_array_equal(result, binary)


def test_dilate_identity_kernel():
    """1×1 kernel is the identity: dilate must return an equal array."""
    binary = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=np.uint8)
    result = dilate(binary, make_rect_kernel(1, 1))
    np.testing.assert_array_equal(result, binary)


def test_dilate_does_not_mutate_input():
    binary = np.zeros((5, 5), dtype=np.uint8)
    binary[2, 2] = 1
    original = binary.copy()
    dilate(binary, make_rect_kernel(3, 3))
    np.testing.assert_array_equal(binary, original)


# ── opening ───────────────────────────────────────────────────────────────────

def test_opening_removes_isolated_noise():
    """An isolated single pixel is removed; a 3×3 solid block survives intact."""
    binary = np.zeros((7, 7), dtype=np.uint8)
    binary[0, 0] = 1          # isolated pixel — smaller than 3×3 kernel
    binary[2:5, 2:5] = 1     # 3×3 block — exactly kernel-sized, preserved
    result = opening(binary, make_rect_kernel(3, 3))
    expected = np.zeros((7, 7), dtype=np.uint8)
    expected[2:5, 2:5] = 1
    np.testing.assert_array_equal(result, expected)


def test_opening_all_zeros_stays_zero():
    result = opening(np.zeros((5, 5), dtype=np.uint8), make_rect_kernel(3, 3))
    np.testing.assert_array_equal(result, np.zeros((5, 5), dtype=np.uint8))


def test_opening_does_not_mutate_input():
    binary = np.ones((5, 5), dtype=np.uint8)
    original = binary.copy()
    opening(binary, make_rect_kernel(3, 3))
    np.testing.assert_array_equal(binary, original)


# ── closing ───────────────────────────────────────────────────────────────────

def test_closing_fills_single_hole():
    """A 3×3 ring (hole in the centre) → after closing with 3×3 kernel the
    centre pixel is filled and the shape becomes a solid 3×3 block."""
    binary = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 0, 1, 0],   # hole at (2, 2)
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.uint8)
    result = closing(binary, make_rect_kernel(3, 3))
    expected = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],   # filled
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.uint8)
    np.testing.assert_array_equal(result, expected)


def test_closing_all_zeros_stays_zero():
    result = closing(np.zeros((5, 5), dtype=np.uint8), make_rect_kernel(3, 3))
    np.testing.assert_array_equal(result, np.zeros((5, 5), dtype=np.uint8))


def test_closing_does_not_mutate_input():
    binary = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.uint8)
    original = binary.copy()
    closing(binary, make_rect_kernel(3, 3))
    np.testing.assert_array_equal(binary, original)


# ── apply_variant ─────────────────────────────────────────────────────────────

def test_apply_variant_none_returns_copy_not_same_object():
    binary = np.ones((5, 5), dtype=np.uint8)
    result = apply_variant(binary, "none", make_rect_kernel(3, 3))
    np.testing.assert_array_equal(result, binary)
    assert result is not binary


def test_apply_variant_opening_matches_opening():
    binary = np.zeros((7, 7), dtype=np.uint8)
    binary[2:5, 2:5] = 1
    binary[0, 0] = 1  # noise
    k = make_rect_kernel(3, 3)
    np.testing.assert_array_equal(apply_variant(binary, "opening", k), opening(binary, k))


def test_apply_variant_closing_matches_closing():
    binary = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.uint8)
    k = make_rect_kernel(3, 3)
    np.testing.assert_array_equal(apply_variant(binary, "closing", k), closing(binary, k))


def test_apply_variant_close_then_open():
    """close_then_open fills holes then removes noise."""
    binary = np.zeros((9, 9), dtype=np.uint8)
    binary[2:7, 2:7] = 1
    binary[4, 4] = 0   # hole — closing should fill
    binary[0, 0] = 1   # noise — opening should remove
    k = make_rect_kernel(3, 3)
    result = apply_variant(binary, "close_then_open", k)
    expected = opening(closing(binary, k), k)
    np.testing.assert_array_equal(result, expected)


def test_apply_variant_open_then_close():
    """open_then_close removes noise then fills holes."""
    binary = np.zeros((9, 9), dtype=np.uint8)
    binary[2:7, 2:7] = 1
    binary[4, 4] = 0
    binary[0, 0] = 1
    k = make_rect_kernel(3, 3)
    result = apply_variant(binary, "open_then_close", k)
    expected = closing(opening(binary, k), k)
    np.testing.assert_array_equal(result, expected)


def test_apply_variant_unknown_raises():
    binary = np.zeros((3, 3), dtype=np.uint8)
    k = make_rect_kernel(3, 3)
    with pytest.raises(ValueError, match="Unknown morphology variant"):
        apply_variant(binary, "badvariant", k)


def test_apply_variant_none_all_zeros():
    binary = np.zeros((5, 5), dtype=np.uint8)
    result = apply_variant(binary, "none", make_rect_kernel(3, 3))
    np.testing.assert_array_equal(result, binary)


def test_apply_variant_does_not_mutate_input():
    binary = np.ones((5, 5), dtype=np.uint8)
    original = binary.copy()
    apply_variant(binary, "opening", make_rect_kernel(3, 3))
    np.testing.assert_array_equal(binary, original)
