"""Smoke tests for backend selection and pipeline backend equivalence."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

from segreader.backend import get_backend

# ── Unit: get_backend ─────────────────────────────────────────────────────────

def test_get_backend_python_accepted():
    assert get_backend("python") == "python"


def test_get_backend_invalid_raises_valueerror():
    with pytest.raises(ValueError, match="Unbekanntes Backend"):
        get_backend("invalid")


def test_get_backend_rust_accepted_when_installed():
    pytest.importorskip("segreader_native")
    assert get_backend("rust") == "rust"


def test_get_backend_rust_importerror_when_missing(monkeypatch):
    """If segreader_native is absent, get_backend('rust') must raise ImportError."""
    import importlib
    import builtins

    original_import = builtins.__import__

    def _block_native(name, *args, **kwargs):
        if name == "segreader_native":
            raise ImportError("segreader_native not available (mocked)")
        return original_import(name, *args, **kwargs)

    # Remove the cached module so our fake import is reached.
    monkeypatch.setitem(sys.modules, "segreader_native", None)  # type: ignore[arg-type]
    monkeypatch.setattr(builtins, "__import__", _block_native)

    with pytest.raises(ImportError, match="segreader_native"):
        get_backend("rust")


# ── Integration: full pipeline with both backends ────────────────────────────

SAMPLE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "samples", "pic1.png",
)


def _run(backend: str) -> str:
    from segreader import run_pipeline
    number_str, _ = run_pipeline(SAMPLE, backend=backend)
    return number_str


@pytest.mark.skipif(not os.path.isfile(SAMPLE), reason="Sample image not found")
def test_pipeline_python_backend_returns_nonempty():
    result = _run("python")
    assert result and result != "?"


@pytest.mark.skipif(not os.path.isfile(SAMPLE), reason="Sample image not found")
def test_pipeline_rust_backend_matches_python():
    pytest.importorskip("segreader_native")
    assert _run("rust") == _run("python")
