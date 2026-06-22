"""Display localization — extracts the segment display region before decoding.

Three strategies selectable via the `method` parameter:

  "brightness"      — thresholds bright, saturated pixels in HSV (Kurs 10); works
                      for any colored LED/7-segment display on a dark background
                      regardless of segment color. Morphological closing (Kurs 07)
                      bridges gaps between individual digit strokes.

  "color"           — same as brightness but uses explicit per-hue ranges (Kurs 10),
                      covering the full LED spectrum (red, orange, yellow, green,
                      cyan, blue). More selective than brightness alone.

  "top_brightness"  — scene-adaptive: thresholds only the brightest pixels by
                      taking the 93rd percentile of the V channel (minimum V=180).
                      Finds glowing displays even when the background is medium-bright.

  "contour"         — Canny edge detection (Kurs 05), morphological dilation (Kurs 07),
                      contour analysis + approxPolyDP quad detection (Kurs 05/06),
                      perspective warp when 4 corners found (Kurs 06). Best for
                      displays with a clear rectangular border on any background.

  "auto"            — tries color → brightness → top_brightness → contour in order;
                      returns the first that finds a region large enough. Order
                      reflects empirical data: color+B channel outperforms
                      brightness+gray across the test set.
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
) -> tuple[np.ndarray | None, np.ndarray, bool, tuple[int, int]]:
    """Shared logic: close mask → bounding box → crop. Returns (crop, debug, found, (ox, oy))."""
    h, w = img_bgr.shape[:2]
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (close_ksize, close_ksize))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    pts = cv2.findNonZero(closed)
    debug = img_bgr.copy()

    if pts is None or cv2.countNonZero(closed) < min_area_ratio * h * w:
        return None, debug, False, (0, 0)

    bx, by, bw, bh = cv2.boundingRect(pts)
    bx = max(0, bx - pad_px)
    by = max(0, by - pad_px)
    bw = min(w - bx, bw + 2 * pad_px)
    bh = min(h - by, bh + 2 * pad_px)

    crop = img_bgr[by:by + bh, bx:bx + bw].copy()
    _draw_box(debug, bx, by, bw, bh, (0, 200, 255), label)
    return crop, debug, True, (bx, by)


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
) -> tuple[np.ndarray | None, np.ndarray, bool, tuple[int, int]]:
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
) -> tuple[np.ndarray | None, np.ndarray, bool, tuple[int, int]]:
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
# Top-brightness localization — scene-adaptive (Kurs 10 + 07)
# Takes only the top-N% brightest pixels; robust to medium-bright backgrounds.
# ---------------------------------------------------------------------------

def _find_by_top_brightness(
    img_bgr: np.ndarray,
    percentile: float = 93.0,
    abs_min: int = 180,
    close_ksize: int | None = None,
    min_area_ratio: float = 0.005,
    pad_px: int = 10,
) -> tuple[np.ndarray | None, np.ndarray, bool, tuple[int, int]]:
    if close_ksize is None:
        close_ksize = _adaptive_close_ksize(img_bgr)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    threshold = max(abs_min, np.percentile(v, percentile))
    mask = (v >= threshold).astype(np.uint8) * 255

    return _crop_from_mask(img_bgr, mask, close_ksize, min_area_ratio, pad_px,
                           "Display (top_brightness)")


# ---------------------------------------------------------------------------
# Contour-based localization (Kurs 05 + 06 + 07)
# ---------------------------------------------------------------------------

def _find_by_contour(
    img_bgr: np.ndarray,
    min_area_ratio: float = 0.01,
    canny_low: int = 50,
    canny_high: int = 150,
    dilate_iters: int = 2,
    pad_px: int = 20,
) -> tuple[np.ndarray | None, np.ndarray, bool, tuple[int, int]]:
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
            rx = max(0, bx - pad_px)
            ry = max(0, by - pad_px)
            rw = min(w - rx, bw + 2 * pad_px)
            rh = min(h - ry, bh + 2 * pad_px)
            dx, dy = bx - rx, by - ry
            dst = np.array(
                [[dx, dy], [dx + bw, dy], [dx + bw, dy + bh], [dx, dy + bh]],
                dtype=np.float32,
            )
            M = cv2.getPerspectiveTransform(pts, dst)
            crop = cv2.warpPerspective(img_bgr, M, (rw, rh))
            cv2.drawContours(debug, [approx], -1, (0, 255, 0), 2)
            for pt in approx.reshape(-1, 2):
                cv2.circle(debug, tuple(pt.astype(int)), 6, (0, 0, 255), -1)
            _draw_box(debug, rx, ry, rw, rh, (0, 200, 255), "Display (quad+pad)")
            return crop, debug, True, (rx, ry)

    if candidates and cv2.contourArea(candidates[0]) >= min_area:
        cnt = candidates[0]
        bx, by, bw, bh = cv2.boundingRect(cnt)
        bx = max(0, bx - pad_px)
        by = max(0, by - pad_px)
        bw = min(w - bx, bw + 2 * pad_px)
        bh = min(h - by, bh + 2 * pad_px)
        crop = img_bgr[by:by + bh, bx:bx + bw].copy()
        _draw_box(debug, bx, by, bw, bh, (0, 200, 255), "Display (bbox)")
        return crop, debug, True, (bx, by)

    return None, debug, False, (0, 0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def shrink_bbox(
    ox: int,
    oy: int,
    w: int,
    h: int,
    shrink: float,
    min_w: int = 20,
    min_h: int = 10,
) -> tuple[int, int, int, int]:
    """Shrink a bounding box inward by *shrink* fraction (0.0–1.0 exclusive).

    Each edge moves inward by shrink/2 of the corresponding dimension, so
    the total reduction is shrink of width and shrink of height.

    Returns (new_ox, new_oy, new_w, new_h).  The new origin can only move
    inward (ox and oy can only increase), so the shrunk box always lies
    within the original box.  The dimensions are clamped to min_w × min_h.
    """
    dx = int(w * shrink / 2)
    dy = int(h * shrink / 2)
    new_w = max(min_w, w - 2 * dx)
    new_h = max(min_h, h - 2 * dy)
    new_ox = ox + dx
    new_oy = oy + dy
    return new_ox, new_oy, new_w, new_h


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
    useful_ratio: float = 0.75,
) -> tuple[np.ndarray, np.ndarray, bool, str, tuple[int, int]]:
    """Locate a segment display region in a scene image.

    method: "brightness" | "color" | "contour" | "auto"

    Returns (crop, debug_img, found, method_used, offset) where offset is the
    (x, y) pixel origin of the crop within the original image, used to project
    digit boxes back onto the full frame.
    """
    shared = dict(close_ksize=close_ksize, min_area_ratio=min_area_ratio, pad_px=pad_px)
    contour_args = dict(min_area_ratio=min_area_ratio, canny_low=canny_low,
                        canny_high=canny_high, dilate_iters=dilate_iters,
                        pad_px=pad_px)

    if method == "brightness":
        crop, debug, found, offset = _find_by_brightness(img_bgr, **shared)
        method_used = "brightness" if found else "none"
    elif method == "color":
        crop, debug, found, offset = _find_by_color(img_bgr, color_ranges=color_ranges, **shared)
        method_used = "color" if found else "none"
    elif method == "contour":
        crop, debug, found, offset = _find_by_contour(img_bgr, **contour_args)
        method_used = "contour" if found else "none"
    elif method == "auto":
        # A crop that covers ≥ useful_ratio of the original image didn't really
        # isolate anything useful — treat it the same as "not found" and try next.
        img_area = img_bgr.shape[0] * img_bgr.shape[1]

        def _useful(c) -> bool:
            if c is None or (c.shape[0] * c.shape[1]) >= useful_ratio * img_area:
                return False
            # Reject portrait-oriented crops — a 7-segment display is always wider than tall.
            if c.shape[1] > 0 and c.shape[0] / c.shape[1] > 1.2:
                return False
            return True

        crop, debug, found, offset = _find_by_color(img_bgr, color_ranges=color_ranges, **shared)
        method_used = "color"
        if not found or not _useful(crop):
            crop, debug, found, offset = _find_by_brightness(img_bgr, **shared)
            method_used = "brightness"
        if not found or not _useful(crop):
            crop, debug, found, offset = _find_by_top_brightness(img_bgr, **shared)
            method_used = "top_brightness"
        if not found or not _useful(crop):
            crop, debug, found, offset = _find_by_contour(img_bgr, **contour_args)
            method_used = "contour"
        if not found or not _useful(crop):
            method_used = "none"
            found = False
            offset = (0, 0)
    else:
        raise ValueError(f"Unknown method {method!r}. Use brightness/color/contour/auto.")

    if not found:
        crop = img_bgr.copy()
        offset = (0, 0)
    return crop, debug, found, method_used, offset


def compose_debug(localize_debug: np.ndarray, annotated: np.ndarray) -> np.ndarray:
    """Side-by-side: localization result left, annotated crop right."""
    target_h = max(localize_debug.shape[0], annotated.shape[0])

    def pad(img: np.ndarray) -> np.ndarray:
        ph = target_h - img.shape[0]
        return img if ph <= 0 else np.vstack(
            [img, np.zeros((ph, img.shape[1], 3), dtype=np.uint8)])

    divider = np.full((target_h, 2, 3), 60, dtype=np.uint8)
    return np.hstack([pad(localize_debug), divider, pad(annotated)])
