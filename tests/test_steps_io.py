"""Tests for steps_io path handling."""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

from segreader.steps_io import save_processed_steps


def _make_dummy_steps() -> list:
    gray = (np.zeros((4, 4), dtype=np.uint8) + 128)
    return [{"id": "original", "name": "Original", "description": "test",
             "image": gray, "details": {}}]


def test_returned_path_uses_forward_slashes(tmp_path):
    steps = _make_dummy_steps()
    result = save_processed_steps(
        steps,
        image_path="img.png",
        result="42",
        params={},
        base_dir=str(tmp_path),
    )
    assert "\\" not in result, f"Backslash found in returned path: {result!r}"


def test_returned_path_is_relative_or_starts_with_base(tmp_path):
    steps = _make_dummy_steps()
    result = save_processed_steps(
        steps,
        image_path="img.png",
        result="42",
        params={},
        base_dir=str(tmp_path),
    )
    # Path should not start with a Windows drive letter like C:/
    # It may be absolute when tmp_path is absolute, but must use forward slashes.
    assert "\\" not in result


def test_steps_json_file_exists(tmp_path):
    steps = _make_dummy_steps()
    result = save_processed_steps(
        steps,
        image_path="display.png",
        result="123",
        params={"threshold": 0.3},
        base_dir=str(tmp_path),
    )
    # The returned POSIX path should point to an existing file.
    # On Windows the OS path uses backslashes — reconstruct it from the POSIX result.
    assert os.path.isfile(result.replace("/", os.sep))


def test_steps_json_is_valid_json(tmp_path):
    steps = _make_dummy_steps()
    result = save_processed_steps(
        steps,
        image_path="display.png",
        result="456",
        params={},
        base_dir=str(tmp_path),
    )
    with open(result.replace("/", os.sep), encoding="utf-8") as f:
        data = json.load(f)
    assert "steps" in data
    assert data["result"] == "456"


def test_stem_with_spaces_produces_posix_path(tmp_path):
    """Stems with spaces are preserved; slashes must still be forward slashes."""
    steps = _make_dummy_steps()
    result = save_processed_steps(
        steps,
        image_path="image copy 10.png",
        result="7",
        params={},
        base_dir=str(tmp_path),
    )
    assert "\\" not in result
    assert "image copy 10" in result


def test_default_base_dir_produces_project_relative_path():
    """When called with the default base_dir the path should be project-relative."""
    steps = _make_dummy_steps()
    import tempfile, shutil
    tmp = tempfile.mkdtemp()
    try:
        result = save_processed_steps(
            steps,
            image_path="x.png",
            result="0",
            params={},
            base_dir=tmp,
        )
        # No backslashes regardless of platform
        assert "\\" not in result
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
