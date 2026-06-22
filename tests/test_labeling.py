"""Tests for labeling.label_components and labeling.get_component_stats.

Uses 4-connectivity (north / south / east / west — no diagonals).
All inputs are small synthetic NumPy arrays; no image files are loaded.
"""
import numpy as np
import pytest

from segreader.labeling import label_components, get_component_stats


# ── label_components ──────────────────────────────────────────────────────────

def test_label_empty_image_all_zero():
    binary = np.zeros((5, 5), dtype=np.uint8)
    labels = label_components(binary)
    np.testing.assert_array_equal(labels, np.zeros((5, 5), dtype=np.int32))


def test_label_single_pixel_gets_one_label():
    binary = np.zeros((5, 5), dtype=np.uint8)
    binary[2, 3] = 1
    labels = label_components(binary)
    unique = [v for v in np.unique(labels) if v != 0]
    assert len(unique) == 1
    assert labels[2, 3] == unique[0]


def test_label_background_stays_zero():
    binary = np.zeros((3, 3), dtype=np.uint8)
    binary[1, 1] = 1
    labels = label_components(binary)
    assert labels[0, 0] == 0
    assert labels[0, 2] == 0
    assert labels[2, 0] == 0


def test_label_two_separated_pixels_get_different_labels():
    """Two pixels separated by a gap must receive distinct labels."""
    binary = np.zeros((1, 5), dtype=np.uint8)
    binary[0, 0] = 1
    binary[0, 4] = 1
    labels = label_components(binary)
    assert labels[0, 0] != 0
    assert labels[0, 4] != 0
    assert labels[0, 0] != labels[0, 4]


def test_label_count_equals_component_count():
    """A binary with three separated blobs must produce exactly three unique labels."""
    binary = np.zeros((3, 9), dtype=np.uint8)
    binary[1, 0] = 1   # blob A
    binary[1, 4] = 1   # blob B
    binary[1, 8] = 1   # blob C
    labels = label_components(binary)
    unique = [v for v in np.unique(labels) if v != 0]
    assert len(unique) == 3


def test_label_diagonal_not_connected():
    """4-connectivity: diagonally adjacent pixels are NOT in the same component."""
    binary = np.array([
        [1, 0],
        [0, 1],
    ], dtype=np.uint8)
    labels = label_components(binary)
    assert labels[0, 0] != 0
    assert labels[1, 1] != 0
    assert labels[0, 0] != labels[1, 1]


def test_label_horizontal_neighbours_connected():
    binary = np.array([[1, 1, 1]], dtype=np.uint8)
    labels = label_components(binary)
    assert labels[0, 0] == labels[0, 1] == labels[0, 2]
    assert labels[0, 0] != 0


def test_label_vertical_neighbours_connected():
    binary = np.array([[1], [1], [1]], dtype=np.uint8)
    labels = label_components(binary)
    assert labels[0, 0] == labels[1, 0] == labels[2, 0]
    assert labels[0, 0] != 0


def test_label_l_shape_is_one_component():
    """L-shaped foreground connected by shared edges → single component."""
    binary = np.array([
        [1, 1],
        [1, 0],
    ], dtype=np.uint8)
    labels = label_components(binary)
    unique = [v for v in np.unique(labels) if v != 0]
    assert len(unique) == 1
    assert labels[0, 0] == labels[0, 1] == labels[1, 0]


def test_label_u_shape_bottom_connects_arms():
    """U-shape: two arms are separate in pass-1 but joined through the base."""
    binary = np.array([
        [1, 0, 1],
        [1, 1, 1],
    ], dtype=np.uint8)
    labels = label_components(binary)
    unique = [v for v in np.unique(labels) if v != 0]
    assert len(unique) == 1


def test_label_output_shape_matches_input():
    binary = np.zeros((6, 8), dtype=np.uint8)
    binary[2:4, 3:5] = 1
    labels = label_components(binary)
    assert labels.shape == binary.shape


def test_label_does_not_mutate_input():
    binary = np.zeros((5, 5), dtype=np.uint8)
    binary[1, 1] = 1
    original = binary.copy()
    label_components(binary)
    np.testing.assert_array_equal(binary, original)


# ── get_component_stats ───────────────────────────────────────────────────────

def test_stats_empty_image_returns_empty_list():
    labels = np.zeros((5, 5), dtype=np.int32)
    assert get_component_stats(labels) == []


def test_stats_single_component_one_entry():
    binary = np.zeros((5, 5), dtype=np.uint8)
    binary[1:4, 1:3] = 1   # 3 rows × 2 cols
    labels = label_components(binary)
    stats = get_component_stats(labels)
    assert len(stats) == 1


def test_stats_label_field_matches_array():
    """The 'label' field must equal the value found in the labels array."""
    binary = np.zeros((5, 5), dtype=np.uint8)
    binary[2, 2] = 1
    labels = label_components(binary)
    stats = get_component_stats(labels)
    assert stats[0]["label"] == int(labels[2, 2])


def test_stats_area_is_pixel_count():
    """area must equal the exact number of foreground pixels."""
    binary = np.zeros((7, 7), dtype=np.uint8)
    binary[1:4, 1:3] = 1   # 3 rows × 2 cols = 6 pixels
    labels = label_components(binary)
    stats = get_component_stats(labels)
    assert stats[0]["area"] == 6


def test_stats_bbox_coordinates():
    """bbox = (y1, x1, y2, x2) with y2/x2 exclusive upper bounds."""
    binary = np.zeros((7, 7), dtype=np.uint8)
    binary[1:4, 2:5] = 1   # rows 1,2,3 — cols 2,3,4
    labels = label_components(binary)
    stats = get_component_stats(labels)
    y1, x1, y2, x2 = stats[0]["bbox"]
    assert (y1, x1, y2, x2) == (1, 2, 4, 5)


def test_stats_bbox_width_and_height():
    """Width and height derived from bbox must match the actual block dimensions."""
    binary = np.zeros((8, 8), dtype=np.uint8)
    binary[2:5, 1:4] = 1   # h=3, w=3
    labels = label_components(binary)
    stats = get_component_stats(labels)
    y1, x1, y2, x2 = stats[0]["bbox"]
    assert y2 - y1 == 3   # height
    assert x2 - x1 == 3   # width


def test_stats_aspect_ratio_tall():
    """A 4-pixel-tall × 2-pixel-wide block has aspect_ratio = 4/2 = 2.0."""
    binary = np.zeros((8, 8), dtype=np.uint8)
    binary[1:5, 3:5] = 1   # h=4, w=2
    labels = label_components(binary)
    stats = get_component_stats(labels)
    assert stats[0]["aspect_ratio"] == pytest.approx(2.0)


def test_stats_aspect_ratio_square():
    """A 3×3 square has aspect_ratio = 1.0."""
    binary = np.zeros((7, 7), dtype=np.uint8)
    binary[2:5, 2:5] = 1
    labels = label_components(binary)
    stats = get_component_stats(labels)
    assert stats[0]["aspect_ratio"] == pytest.approx(1.0)


def test_stats_aspect_ratio_single_pixel():
    """A 1×1 pixel has h=1, w=1, aspect_ratio = 1.0."""
    binary = np.zeros((5, 5), dtype=np.uint8)
    binary[3, 3] = 1
    labels = label_components(binary)
    stats = get_component_stats(labels)
    assert stats[0]["aspect_ratio"] == pytest.approx(1.0)


def test_stats_sorted_left_to_right():
    """Stats must be sorted by x1 (leftmost column) in ascending order."""
    binary = np.zeros((5, 10), dtype=np.uint8)
    binary[2, 8] = 1   # right component (x1=8)
    binary[2, 1] = 1   # left  component (x1=1)
    labels = label_components(binary)
    stats = get_component_stats(labels)
    assert len(stats) == 2
    assert stats[0]["bbox"][1] == 1   # x1 of left component
    assert stats[1]["bbox"][1] == 8   # x1 of right component


def test_stats_two_components_correct_areas():
    """Two distinct blobs: individual areas are correct."""
    binary = np.zeros((5, 10), dtype=np.uint8)
    binary[1:3, 0:2] = 1   # 2×2 = 4 pixels
    binary[1:4, 6:9] = 1   # 3×3 = 9 pixels
    labels = label_components(binary)
    stats = get_component_stats(labels)
    assert len(stats) == 2
    areas = {s["bbox"][1]: s["area"] for s in stats}   # keyed by x1
    assert areas[0] == 4
    assert areas[6] == 9


def test_stats_does_not_mutate_labels():
    binary = np.zeros((5, 5), dtype=np.uint8)
    binary[1:4, 1:4] = 1
    labels = label_components(binary)
    original = labels.copy()
    get_component_stats(labels)
    np.testing.assert_array_equal(labels, original)
