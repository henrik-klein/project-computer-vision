import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    # Punktoperation: gewichtete Summe der RGB-Kanäle -> Graustufenbild
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def adjust_contrast(gray: np.ndarray, alpha: float = 1.5, beta: int = 0) -> np.ndarray:
    # Punktoperation: lineare Kontraststreckung out = clip(alpha * x + beta, 0, 255)
    stretched = alpha * gray.astype(np.float32) + beta
    return np.clip(stretched, 0, 255).astype(np.uint8)


def otsu_binarize(gray: np.ndarray) -> np.ndarray:
    # Histogramm / Otsu-Schwellwert: maximiert die Zwischen-Klassen-Varianz
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

    threshold = int(np.argmax(sigma_b2))
    binary = (gray > threshold).astype(np.uint8)

    # Polaritätskorrektur (Punktoperation): heller Hintergrund -> invertieren
    if binary.mean() > 0.5:
        binary = 1 - binary

    return binary
