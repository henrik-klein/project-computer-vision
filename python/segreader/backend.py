"""
Backend-Schalter: wählt zwischen 'python' (heute) und 'rust' (später) für
morphology- und labeling-Funktionen.
"""

SUPPORTED_BACKENDS = ("python",)


def get_backend(name: str) -> str:
    if name not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unbekanntes Backend {name!r}. Verfügbar: {SUPPORTED_BACKENDS}"
        )
    return name
