"""Display localization — extracts the segment display region before decoding.

Three strategies selectable via the `method` parameter:

  "brightness" — thresholds bright, saturated pixels in HSV (Kurs 10); works
                 for any colored LED/7-segment display on a dark background
                 regardless of segment color. Morphological closing (Kurs 07)
                 bridges gaps between individual digit strokes.

  "color"      — same as brightness but uses explicit per-hue ranges (Kurs 10),
                 covering the full LED spectrum (red, orange, yellow, green,
                 cyan, blue). More selective than brightness alone.

  "contour"    — Canny edge detection (Kurs 05), morphological dilation (Kurs 07),
                 contour analysis + approxPolyDP quad detection (Kurs 05/06),
                 perspective warp when 4 corners found (Kurs 06). Best for
                 displays with a clear rectangular border on any background.

  "auto"       — tries color → brightness → contour in order; returns the first
                 that finds a region large enough. Order reflects empirical data:
                 color+B channel outperforms brightness+gray across the test set.
"""
from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _order_quad(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(diff)],
        pts[np.argmax(s)],
        pts[np.argmax(diff)],
    ], dtype=np.float32)


def _draw_box(img: np.ndarray, x: int, y: int, w: int, h: int,
              color, label: str) -> None:
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(img, label, (x, max(y - 8, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def _adaptive_close_ksize(img_bgr: np.ndarray) -> int:
    """Closing kernel that scales with image size to bridge digit gaps."""
    return max(15, max(img_bgr.shape[:2]) // 20)


def _crop_from_mask(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    close_ksize: int,
    min_area_ratio: float,
    pad_px: int,
    label: str,
) -> tuple[np.ndarray | None, np.ndarray, bool]:
    """Shared logic: close mask → bounding box → crop."""
    h, w = img_bgr.shape[:2]
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (close_ksize, close_ksize))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    pts = cv2.findNonZero(closed)
    debug = img_bgr.copy()

    if pts is None or cv2.countNonZero(closed) < min_area_ratio * h * w:
        return None, debug, False

    bx, by, bw, bh = cv2.boundingRect(pts)
    bx = max(0, bx - pad_px)
    by = max(0, by - pad_px)
    bw = min(w - bx, bw + 2 * pad_px)
    bh = min(h - by, bh + 2 * pad_px)

    crop = img_bgr[by:by + bh, bx:bx + bw].copy()
    _draw_box(debug, bx, by, bw, bh, (0, 200, 255), label)
    return crop, debug, True


# ---------------------------------------------------------------------------
# Brightness-based localization (Kurs 10 + 07)
# Full-spectrum: any bright, saturated pixel — catches all LED colors at once.
# ---------------------------------------------------------------------------

def _find_by_brightness(
    img_bgr: np.ndarray,
    val_thresh: int = 150,
    sat_min: int = 50,
    close_ksize: int | None = None,
    min_area_ratio: float = 0.005,
    pad_px: int = 10,
) -> tuple[np.ndarray | None, np.ndarray, bool]:
    if close_ksize is None:
        close_ksize = _adaptive_close_ksize(img_bgr)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lo = np.array([0,       sat_min,  val_thresh], dtype=np.uint8)
    hi = np.array([180,     255,      255],         dtype=np.uint8)
    mask = cv2.inRange(hsv, lo, hi)

    return _crop_from_mask(img_bgr, mask, close_ksize, min_area_ratio, pad_px,
                           "Display (brightness)")


# ---------------------------------------------------------------------------
# Color-based localization (Kurs 10 + 07)
# Explicit hue ranges cover the full visible LED spectrum.
# ---------------------------------------------------------------------------

# (hue_lo, hue_hi, sat_min, val_min) — OpenCV HSV: H in [0,180]
_FULL_SPECTRUM_RANGES: list[tuple[int, int, int, int]] = [
    (0,   30,  80, 100),  # red → orange
    (30,  80,  80, 100),  # yellow → green
    (80, 130,  80, 100),  # cyan → blue
    (130, 155, 80, 100),  # blue → violet
    (155, 180, 80, 100),  # red wrap-around
]


def _find_by_color(
    img_bgr: np.ndarray,
    color_ranges: list[tuple[int, int, int, int]] | None = None,
    close_ksize: int | None = None,
    min_area_ratio: float = 0.005,
    pad_px: int = 10,
) -> tuple[np.ndarray | None, np.ndarray, bool]:
    if color_ranges is None:
        color_ranges = _FULL_SPECTRUM_RANGES
    if close_ksize is None:
        close_ksize = _adaptive_close_ksize(img_bgr)

    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    mask = np.zeros((h, w), dtype=np.uint8)
    for h_lo, h_hi, s_min, v_min in color_ranges:
        lo = np.array([h_lo, s_min, v_min], dtype=np.uint8)
        hi = np.array([h_hi, 255,   255],   dtype=np.uint8)
        mask |= cv2.inRange(hsv, lo, hi)

    return _crop_from_mask(img_bgr, mask, close_ksize, min_area_ratio, pad_px,
                           "Display (color)")


# ---------------------------------------------------------------------------
# Contour-based localization (Kurs 05 + 06 + 07)
# ---------------------------------------------------------------------------

def _find_by_contour(
    img_bgr: np.ndarray,
    min_area_ratio: float = 0.01,
    canny_low: int = 50,
    canny_high: int = 150,
    dilate_iters: int = 2,
) -> tuple[np.ndarray | None, np.ndarray, bool]:
    h, w = img_bgr.shape[:2]
    min_area = min_area_ratio * h * w

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=dilate_iters)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = sorted(contours, key=cv2.contourArea, reverse=True)

    debug = img_bgr.copy()

    for cnt in candidates:
        if cv2.contourArea(cnt) < min_area:
            break
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            pts = _order_quad(approx)
            bx, by, bw, bh = cv2.boundingRect(approx)
            dst = np.array([[0, 0], [bw, 0], [bw, bh], [0, bh]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(pts, dst)
            crop = cv2.warpPerspective(img_bgr, M, (bw, bh))
            cv2.drawContours(debug, [approx], -1, (0, 255, 0), 2)
            for pt in approx.reshape(-1, 2):
                cv2.circle(debug, tuple(pt.astype(int)), 6, (0, 0, 255), -1)
            cv2.putText(debug, "Display (quad)", (bx, max(by - 8, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            return crop, debug, True

    if candidates and cv2.contourArea(candidates[0]) >= min_area:
        cnt = candidates[0]
        bx, by, bw, bh = cv2.boundingRect(cnt)
        crop = img_bgr[by:by + bh, bx:bx + bw].copy()
        _draw_box(debug, bx, by, bw, bh, (0, 200, 255), "Display (bbox)")
        return crop, debug, True

    return None, debug, False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_display(
    img_bgr: np.ndarray,
    method: str = "auto",
    color_ranges: list[tuple[int, int, int, int]] | None = None,
    close_ksize: int | None = None,
    min_area_ratio: float = 0.005,
    canny_low: int = 50,
    canny_high: int = 150,
    dilate_iters: int = 2,
    pad_px: int = 10,
) -> tuple[np.ndarray, np.ndarray, bool, str]:
    """Locate a segment display region in a scene image.

    method: "brightness" | "color" | "contour" | "auto"

    Returns (crop, debug_img, found, method_used).
    """
    shared = dict(close_ksize=close_ksize, min_area_ratio=min_area_ratio, pad_px=pad_px)
    contour_args = dict(min_area_ratio=min_area_ratio, canny_low=canny_low,
                        canny_high=canny_high, dilate_iters=dilate_iters)

    if method == "brightness":
        crop, debug, found = _find_by_brightness(img_bgr, **shared)
        method_used = "brightness" if found else "none"
    elif method == "color":
        crop, debug, found = _find_by_color(img_bgr, color_ranges=color_ranges, **shared)
        method_used = "color" if found else "none"
    elif method == "contour":
        crop, debug, found = _find_by_contour(img_bgr, **contour_args)
        method_used = "contour" if found else "none"
    elif method == "auto":
        # color first: data shows color+B outperforms brightness+gray across the dataset.
        crop, debug, found = _find_by_color(img_bgr, color_ranges=color_ranges, **shared)
        method_used = "color"
        if not found:
            crop, debug, found = _find_by_brightness(img_bgr, **shared)
            method_used = "brightness"
        if not found:
            crop, debug, found = _find_by_contour(img_bgr, **contour_args)
            method_used = "contour"
        if not found:
            method_used = "none"
    else:
        raise ValueError(f"Unknown method {method!r}. Use brightness/color/contour/auto.")

    if not found:
        crop = img_bgr.copy()
    return crop, debug, found, method_used


def compose_debug(localize_debug: np.ndarray, annotated: np.ndarray) -> np.ndarray:
    """Side-by-side: localization result left, annotated crop right."""
    target_h = max(localize_debug.shape[0], annotated.shape[0])

    def pad(img: np.ndarray) -> np.ndarray:
        ph = target_h - img.shape[0]
        return img if ph <= 0 else np.vstack(
            [img, np.zeros((ph, img.shape[1], 3), dtype=np.uint8)])

    divider = np.full((target_h, 2, 3), 60, dtype=np.uint8)
    return np.hstack([pad(localize_debug), divider, pad(annotated)])
