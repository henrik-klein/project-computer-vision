"""
Backend-Schalter: wählt zwischen 'python' (default) und 'rust' für
labeling-Funktionen.

Default backend is always 'python' so the project works without building Rust.
Select 'rust' only when segreader_native has been compiled and installed.
"""
from __future__ import annotations

SUPPORTED_BACKENDS = ("python", "rust")


def get_backend(name: str) -> str:
    if name not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unbekanntes Backend {name!r}. Verfügbar: {SUPPORTED_BACKENDS}"
        )
    if name == "rust":
        try:
            import segreader_native  # noqa: F401
        except ImportError:
            raise ImportError(
                "Backend 'rust' selected but segreader_native is not installed.\n"
                "Build and install it with:\n"
                "  cd rust/segreader-native\n"
                "  RUSTUP_TOOLCHAIN=stable-x86_64-pc-windows-gnu "
                "uv run maturin develop --target x86_64-pc-windows-gnu\n"
                "See rust/README.md for full build instructions."
            )
    return name
