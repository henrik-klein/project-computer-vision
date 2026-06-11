"""
Morphologische Filter — eigene numpy-Referenzimplementierung (Baseline für Rust-Vergleich).
Kein cv2 / scipy im Implementierungscode; nur in Tests zum Gegenchecken erlaubt.
"""
import numpy as np


def make_rect_kernel(height: int, width: int) -> np.ndarray:
    return np.ones((height, width), dtype=np.uint8)


def _shift_and_clip(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Verschiebt arr um (dy, dx) und setzt eingerollte Ränder auf 0."""
    shifted = np.roll(arr, shift=dy, axis=0)
    if dy > 0:
        shifted[:dy, :] = 0
    elif dy < 0:
        shifted[dy:, :] = 0

    shifted = np.roll(shifted, shift=dx, axis=1)
    if dx > 0:
        shifted[:, :dx] = 0
    elif dx < 0:
        shifted[:, dx:] = 0

    return shifted


def erode(binary: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    # Morphologische Erosion: AND aller verschobenen Kopien gemäß Strukturelement
    kh, kw = kernel.shape
    cy, cx = kh // 2, kw // 2
    result = np.ones(binary.shape, dtype=np.uint8)
    for r in range(kh):
        for c in range(kw):
            if kernel[r, c]:
                result &= _shift_and_clip(binary, cy - r, cx - c)
    return result


def dilate(binary: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    # Morphologische Dilatation: OR aller verschobenen Kopien gemäß Strukturelement
    kh, kw = kernel.shape
    cy, cx = kh // 2, kw // 2
    result = np.zeros(binary.shape, dtype=np.uint8)
    for r in range(kh):
        for c in range(kw):
            if kernel[r, c]:
                result |= _shift_and_clip(binary, cy - r, cx - c)
    return result


def opening(binary: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    # Opening = Erosion dann Dilatation: entfernt kleine Vordergrund-Objekte
    return dilate(erode(binary, kernel), kernel)


def closing(binary: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    # Closing = Dilatation dann Erosion: schließt kleine Löcher im Vordergrund
    return erode(dilate(binary, kernel), kernel)
