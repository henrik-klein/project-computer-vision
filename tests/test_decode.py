"""Tests der 7-Segment-Lookup-Tabelle für alle Ziffern 0-9.

Verwendet synthetische Binär-Patches, in denen die Segmentzonen direkt
gemalt werden — unabhängig von I/O oder Vorverarbeitung.
"""
import numpy as np
import pytest

from segreader.decode import (
    SEGMENT_ZONES,
    LOOKUP_TABLE,
    sample_segments,
    decode_digit,
    decode_number,
)

PATCH_H, PATCH_W = 80, 50


def make_digit_patch(active_segments: set) -> np.ndarray:
    """Erstellt einen Binär-Patch mit nur den angegebenen Segmenten."""
    patch = np.zeros((PATCH_H, PATCH_W), dtype=np.uint8)
    for seg, (ys, ye, xs, xe) in SEGMENT_ZONES.items():
        if seg in active_segments:
            r1 = int(ys * PATCH_H)
            r2 = max(int(ye * PATCH_H), r1 + 1)
            c1 = int(xs * PATCH_W)
            c2 = max(int(xe * PATCH_W), c1 + 1)
            patch[r1:r2, c1:c2] = 1
    return patch


@pytest.mark.parametrize("digit,segments", [
    ("0", set("abcdef")),
    ("1", set("bc")),
    ("2", set("abdeg")),
    ("3", set("abcdg")),
    ("4", set("bcfg")),
    ("5", set("acdfg")),
    ("6", set("acdefg")),
    ("7", set("abc")),
    ("8", set("abcdefg")),
    ("9", set("abcdfg")),
])
def test_decode_digit_synthetic(digit: str, segments: set):
    patch = make_digit_patch(segments)
    result = decode_digit(patch, threshold=0.3)
    assert result == digit, f"Erwartet {digit!r}, erhalten {result!r}"


def test_decode_unknown_returns_question_mark():
    patch = np.zeros((PATCH_H, PATCH_W), dtype=np.uint8)
    assert decode_digit(patch) == "?"


def test_sample_segments_all_active():
    patch = np.ones((PATCH_H, PATCH_W), dtype=np.uint8)
    assert sample_segments(patch, threshold=0.3) == frozenset("abcdefg")


def test_sample_segments_all_inactive():
    patch = np.zeros((PATCH_H, PATCH_W), dtype=np.uint8)
    assert sample_segments(patch, threshold=0.3) == frozenset()


def test_decode_number_sequence():
    """Dekodiert drei nebeneinander gelegte Patches als Zahl '123'."""
    specs = [("1", set("bc")), ("2", set("abdeg")), ("3", set("abcdg"))]
    patches = [make_digit_patch(segs) for _, segs in specs]
    gap = np.zeros((PATCH_H, 5), dtype=np.uint8)
    combined = np.hstack([patches[0], gap, patches[1], gap, patches[2]])

    boxes = [
        (0, 0,               PATCH_H, PATCH_W),
        (0, PATCH_W + 5,     PATCH_H, 2 * PATCH_W + 5),
        (0, 2 * PATCH_W + 10, PATCH_H, 3 * PATCH_W + 10),
    ]
    number_str, per_digit = decode_number(combined, boxes, threshold=0.3)
    assert number_str == "123"
    assert per_digit == ["1", "2", "3"]
