import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Konvertiert ein BGR-Bild in Graustufen (gewichtete Luma-Summe)."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def adjust_contrast(gray: np.ndarray, alpha: float = 1.5, beta: int = 0) -> np.ndarray:
    """Lineare Kontraststreckung: output = clip(alpha * x + beta, 0, 255)."""
    stretched = alpha * gray.astype(np.float32) + beta
    return np.clip(stretched, 0, 255).astype(np.uint8)


def _compute_otsu_threshold(gray: np.ndarray) -> int:
    """Berechnet den Otsu-Schwellwert durch Maximierung der Zwischen-Klassen-Varianz.

    Gibt nur den Schwellwert als int zurück — keine Maske, keine Polaritätskorrektur.
    Eingabe wird nicht verändert.
    """
    # Histogramm / normierte Wahrscheinlichkeiten
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    prob = hist / total

    # kumulierte Summen für effiziente Berechnung der Klassenmittelwerte
    cum_prob = np.cumsum(prob)
    cum_mean = np.cumsum(prob * np.arange(256, dtype=np.float64))
    global_mean = cum_mean[-1]

    w0 = cum_prob
    w1 = 1.0 - cum_prob
    # Division durch null vermeiden
    with np.errstate(divide="ignore", invalid="ignore"):
        mu0 = np.where(w0 > 0, cum_mean / w0, 0.0)
        mu1 = np.where(w1 > 0, (global_mean - cum_mean) / w1, 0.0)
    sigma_b2 = w0 * w1 * (mu0 - mu1) ** 2

    return int(np.argmax(sigma_b2))


def otsu_binarize(gray: np.ndarray) -> np.ndarray:
    # Otsu-Schwellwert berechnen, dann binarisieren
    threshold = _compute_otsu_threshold(gray)
    binary = (gray > threshold).astype(np.uint8)

    # Polaritätskorrektur (Punktoperation): heller Hintergrund -> invertieren
    if binary.mean() > 0.5:
        binary = 1 - binary

    return binary


def otsu_dark_binarize(gray: np.ndarray) -> np.ndarray:
    """Binarize using Otsu threshold with dark pixels as foreground (no polarity flip).

    Unlike otsu_binarize, this treats pixels *below* the Otsu threshold as
    foreground (=1).  No polarity correction is applied — it directly targets
    dark LCD digit segments on a bright display surface.

    Returns uint8 ndarray with values in {0, 1}, same shape as input.
    Input is not modified.
    """
    threshold = _compute_otsu_threshold(gray)
    return (gray <= threshold).astype(np.uint8)


def local_mean_binarize(
    gray: np.ndarray,
    window: int = 31,
    offset: int = 10,
    polarity: str = "bright",
) -> np.ndarray:
    """Local mean adaptive binarization (pure NumPy, O(H*W) integral-image box filter).

    Each pixel is compared to the mean of its window×window neighbourhood:
      polarity='bright' -> foreground if pixel > local_mean + offset
      polarity='dark'   -> foreground if pixel < local_mean - offset

    This correctly handles reversed LCD displays: a dark digit segment surrounded
    by a bright display surface will have a high local mean, so the dark pixel
    falls well below local_mean - offset.  A dark background pixel surrounded by
    other dark pixels has a low local mean and does NOT trigger the 'dark' rule.

    Args:
        gray:     uint8 grayscale image.
        window:   positive odd integer — side length of the neighbourhood square.
        offset:   non-negative integer — shifts the decision boundary away from
                  the local mean (larger = more selective).
        polarity: 'bright' or 'dark'.

    Returns:
        uint8 ndarray with values in {0, 1}, same shape as input.

    Raises:
        ValueError: if window is not a positive odd integer, or polarity is unknown.
    """
    if window < 1 or window % 2 == 0:
        raise ValueError(f"window must be a positive odd integer, got {window}")
    if polarity not in ("bright", "dark"):
        raise ValueError(f"polarity must be 'bright' or 'dark', got {polarity!r}")

    r = window // 2
    # Reflect-pad so every pixel has a full window neighbourhood at the borders.
    padded = np.pad(gray.astype(np.float64), r, mode="reflect")
    # Integral image (one zero row/col prefix for clean box-sum formula).
    integral = np.pad(
        padded.cumsum(axis=0).cumsum(axis=1),
        ((1, 0), (1, 0)),
        mode="constant",
        constant_values=0,
    )

    h, w = gray.shape
    rows = np.arange(h)[:, np.newaxis]   # (h, 1)
    cols = np.arange(w)[np.newaxis, :]   # (1, w)

    # Box sum over the window×window region around each (original) pixel.
    box_sum = (
        integral[rows + window, cols + window]
        - integral[rows,        cols + window]
        - integral[rows + window, cols       ]
        + integral[rows,          cols       ]
    )
    local_mean = box_sum / (window * window)

    g = gray.astype(np.float64)
    if polarity == "bright":
        return (g > local_mean + offset).astype(np.uint8)
    return (g < local_mean - offset).astype(np.uint8)
