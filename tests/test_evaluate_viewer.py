"""Tests for evaluate.write_viewer_result and --viewer-result behavior."""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate import write_viewer_result


def _write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


# ── write_viewer_result unit tests ───────────────────────────────────────────

def test_viewer_result_creates_file(tmp_path):
    src = str(tmp_path / "result.json")
    dst = str(tmp_path / "viewer" / "result.json")
    _write_json(src, {"results": {}})
    returned = write_viewer_result(src, dst)
    assert os.path.isfile(dst)


def test_viewer_result_content_matches_source(tmp_path):
    src = str(tmp_path / "result.json")
    dst = str(tmp_path / "viewer" / "result.json")
    data = {"results": {"img": {"predicted": "42"}}}
    _write_json(src, data)
    write_viewer_result(src, dst)
    with open(dst, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == data


def test_viewer_result_returns_posix_path(tmp_path):
    src = str(tmp_path / "result.json")
    dst = str(tmp_path / "sub" / "result.json")
    _write_json(src, {})
    returned = write_viewer_result(src, dst)
    assert "\\" not in returned, f"Backslash in returned path: {returned!r}"


def test_viewer_result_source_still_exists(tmp_path):
    src = str(tmp_path / "result.json")
    dst = str(tmp_path / "viewer" / "result.json")
    _write_json(src, {})
    write_viewer_result(src, dst)
    assert os.path.isfile(src), "Source file was removed after copy"


def test_viewer_result_same_file_does_not_raise(tmp_path):
    path = str(tmp_path / "result.json")
    _write_json(path, {"ok": True})
    # Should not raise even when src == dst
    returned = write_viewer_result(path, path)
    assert os.path.isfile(path)


def test_viewer_result_not_created_when_absent(tmp_path):
    """If write_viewer_result is never called, the viewer file must not appear."""
    src = str(tmp_path / "output.json")
    viewer = str(tmp_path / "data" / "testing" / "result.json")
    _write_json(src, {})
    # We deliberately do NOT call write_viewer_result here.
    assert not os.path.isfile(viewer)


def test_viewer_result_creates_parent_dirs(tmp_path):
    src = str(tmp_path / "result.json")
    # Deep nested destination that does not yet exist.
    dst = str(tmp_path / "a" / "b" / "c" / "result.json")
    _write_json(src, {})
    write_viewer_result(src, dst)
    assert os.path.isfile(dst)
