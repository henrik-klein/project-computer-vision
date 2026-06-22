"""Tests for localize.shrink_bbox.

Covers:
- Identity at shrink=0.0
- Correct inward movement of origin (ox, oy only increase)
- Width and height reduction proportional to shrink
- min_w / min_h clamp when shrink is large
- No negative origin (origin is always >= input ox/oy)
- Symmetric shrink: total width reduction = 2*dx, total height reduction = 2*dy
- Custom min_w and min_h respected
"""
import pytest

from segreader.localize import shrink_bbox


# ── helpers ──────────────────────────────────────────────────────────────────

def _bbox(ox=0, oy=0, w=200, h=100):
    return ox, oy, w, h


# ── identity at shrink=0 ─────────────────────────────────────────────────────

def test_shrink_bbox_zero_shrink_is_identity():
    ox, oy, w, h = _bbox(ox=10, oy=20, w=200, h=100)
    new_ox, new_oy, new_w, new_h = shrink_bbox(ox, oy, w, h, shrink=0.0)
    assert new_ox == ox
    assert new_oy == oy
    assert new_w == w
    assert new_h == h


# ── origin moves inward (only increases) ─────────────────────────────────────

def test_shrink_bbox_origin_moves_inward():
    ox, oy, w, h = _bbox(ox=5, oy=10, w=100, h=80)
    new_ox, new_oy, new_w, new_h = shrink_bbox(ox, oy, w, h, shrink=0.1)
    assert new_ox >= ox
    assert new_oy >= oy


def test_shrink_bbox_origin_offset_equals_dx():
    """new_ox - ox == int(w * shrink / 2)."""
    ox, oy, w, h = _bbox(ox=0, oy=0, w=200, h=100)
    shrink = 0.1
    new_ox, new_oy, new_w, new_h = shrink_bbox(ox, oy, w, h, shrink)
    expected_dx = int(w * shrink / 2)  # = 10
    assert new_ox == ox + expected_dx
    assert new_oy == oy + int(h * shrink / 2)


# ── dimensions reduce proportionally ─────────────────────────────────────────

def test_shrink_bbox_width_reduced_by_two_dx():
    ox, oy, w, h = _bbox(w=200, h=100)
    shrink = 0.10
    new_ox, new_oy, new_w, new_h = shrink_bbox(ox, oy, w, h, shrink)
    dx = int(w * shrink / 2)  # 10
    assert new_w == w - 2 * dx   # 180


def test_shrink_bbox_height_reduced_by_two_dy():
    ox, oy, w, h = _bbox(w=200, h=100)
    shrink = 0.10
    new_ox, new_oy, new_w, new_h = shrink_bbox(ox, oy, w, h, shrink)
    dy = int(h * shrink / 2)  # 5
    assert new_h == h - 2 * dy   # 90


# ── min_w / min_h clamp ───────────────────────────────────────────────────────

def test_shrink_bbox_clamp_to_min_w():
    """Extreme shrink: w - 2*dx would undercut min_w → clamp to min_w."""
    ox, oy, w, h = _bbox(w=30, h=30)
    new_ox, new_oy, new_w, new_h = shrink_bbox(ox, oy, w, h, shrink=0.9, min_w=20)
    assert new_w >= 20


def test_shrink_bbox_clamp_to_min_h():
    ox, oy, w, h = _bbox(w=30, h=30)
    new_ox, new_oy, new_w, new_h = shrink_bbox(ox, oy, w, h, shrink=0.9, min_h=10)
    assert new_h >= 10


def test_shrink_bbox_default_min_w_is_20():
    ox, oy, w, h = _bbox(w=10, h=30)
    new_ox, new_oy, new_w, new_h = shrink_bbox(ox, oy, w, h, shrink=0.99)
    assert new_w >= 20


def test_shrink_bbox_default_min_h_is_10():
    ox, oy, w, h = _bbox(w=30, h=5)
    new_ox, new_oy, new_w, new_h = shrink_bbox(ox, oy, w, h, shrink=0.99)
    assert new_h >= 10


# ── custom min_w / min_h ─────────────────────────────────────────────────────

def test_shrink_bbox_custom_min_w_respected():
    ox, oy, w, h = _bbox(w=100, h=100)
    new_ox, new_oy, new_w, new_h = shrink_bbox(ox, oy, w, h, shrink=0.99, min_w=50)
    assert new_w >= 50


def test_shrink_bbox_custom_min_h_respected():
    ox, oy, w, h = _bbox(w=100, h=100)
    new_ox, new_oy, new_w, new_h = shrink_bbox(ox, oy, w, h, shrink=0.99, min_h=30)
    assert new_h >= 30


# ── shrunk box lies within original ──────────────────────────────────────────

def test_shrink_bbox_right_edge_within_original():
    """new_ox + new_w <= ox + w when no min-clamp is active."""
    ox, oy, w, h = _bbox(ox=10, oy=10, w=200, h=100)
    new_ox, new_oy, new_w, new_h = shrink_bbox(ox, oy, w, h, shrink=0.10)
    assert new_ox + new_w <= ox + w


def test_shrink_bbox_bottom_edge_within_original():
    ox, oy, w, h = _bbox(ox=10, oy=10, w=200, h=100)
    new_ox, new_oy, new_w, new_h = shrink_bbox(ox, oy, w, h, shrink=0.10)
    assert new_oy + new_h <= oy + h


# ── return type ──────────────────────────────────────────────────────────────

def test_shrink_bbox_returns_four_ints():
    result = shrink_bbox(0, 0, 100, 50, shrink=0.1)
    assert len(result) == 4
    assert all(isinstance(v, int) for v in result)


# ── no mutation (pure function) ───────────────────────────────────────────────

def test_shrink_bbox_does_not_mutate_inputs():
    original = (5, 10, 200, 100)
    shrink_bbox(*original, shrink=0.1)
    assert original == (5, 10, 200, 100)


# ── non-zero origin preserved ─────────────────────────────────────────────────

def test_shrink_bbox_nonzero_origin_preserved():
    ox, oy, w, h = 50, 30, 300, 150
    new_ox, new_oy, new_w, new_h = shrink_bbox(ox, oy, w, h, shrink=0.0)
    assert new_ox == 50
    assert new_oy == 30
