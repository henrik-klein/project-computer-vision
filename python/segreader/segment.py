"""
Projektionsprofil zur Ziffern-Segmentierung.
"""
import numpy as np


def vertical_projection(binary: np.ndarray) -> np.ndarray:
    # Projektionsprofil: Spaltensummen der Vordergrundpixel
    return binary.sum(axis=0).astype(np.float64)


def smooth_projection(projection: np.ndarray, window: int = 1) -> np.ndarray:
    """Apply a uniform 1-D moving-average to reduce projection noise.

    window=1 is the identity (same object returned, no copy).
    Uses np.convolve with mode='same', so the output length equals the input length.
    """
    if window <= 1 or len(projection) == 0:
        return projection
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(projection, kernel, mode="same")


def compute_projection_threshold(
    projection: np.ndarray,
    height: int,
    threshold_factor: float = 0.05,
    fallback_percentile: float = 25.0,
) -> float:
    """Return the active-column threshold with an adaptive fallback.

    Normally returns threshold_factor * height.  When no column meets
    that value but nonzero foreground exists, falls back to the
    fallback_percentile of the nonzero projection values — this prevents
    returning zero boxes for low-contrast or small-crop binary images.
    """
    fixed = float(threshold_factor * height)
    if np.any(projection >= fixed):
        return fixed
    nonzero = projection[projection > 0]
    if len(nonzero) == 0:
        return fixed
    return float(np.percentile(nonzero, fallback_percentile))


def split_run_at_valleys(
    projection: np.ndarray,
    x1: int,
    x2: int,
    valley_threshold: float,
    min_segment_width: int = 5,
) -> list:
    """Split column range [x1, x2) at contiguous below-threshold regions.

    A valley column is any c where projection[c] < valley_threshold.
    Passing the same value used as the active-column threshold guarantees
    that no genuinely-active digit column is ever a valley — only the
    zero-projection gap columns bridged by gap-merging trigger splits.

    Returns a list of (start, end) sub-ranges (end exclusive).
    Returns [(x1, x2)] unchanged when no qualifying split is found or the
    range is too narrow to contain two segments of min_segment_width.
    """
    if x2 - x1 < 2 * min_segment_width:
        return [(x1, x2)]

    sub_ranges: list = []
    seg_start: int | None = None

    for col in range(x1, x2):
        if projection[col] >= valley_threshold:
            if seg_start is None:
                seg_start = col
        else:
            if seg_start is not None and col - seg_start >= min_segment_width:
                sub_ranges.append((seg_start, col))
            seg_start = None

    if seg_start is not None and x2 - seg_start >= min_segment_width:
        sub_ranges.append((seg_start, x2))

    return sub_ranges if len(sub_ranges) > 1 else [(x1, x2)]


def find_digit_boxes(
    binary: np.ndarray,
    projection: np.ndarray,
    threshold_factor: float = 0.05,
    min_gap: int = 3,
    min_width: int = 3,
    smooth_window: int = 1,
    min_segment_width: int = 5,
) -> list:
    """Findet Ziffern-Bounding-Boxes über das vertikale Projektionsprofil.

    y-Grenzen werden pro Box aus den tatsächlich leuchtenden Pixeln im
    jeweiligen Spaltenband bestimmt (nicht das ganze Bild als y-Bereich).

    smooth_window=1 disables smoothing (identity).
    Intra-run valley splitting uses the same threshold as active-column detection,
    so only below-threshold gap columns trigger splits — never active digit columns.
    """
    proj = smooth_projection(projection, smooth_window)
    threshold = compute_projection_threshold(proj, binary.shape[0], threshold_factor)
    active_cols = np.where(proj >= threshold)[0]
    if len(active_cols) == 0:
        return []

    # Spalten-Runs mit Gap-Merging zusammenfassen
    runs: list = []
    current_run = [int(active_cols[0])]
    for col in active_cols[1:]:
        col = int(col)
        if col - current_run[-1] <= min_gap:
            current_run.append(col)
        else:
            runs.append(current_run)
            current_run = [col]
    runs.append(current_run)

    boxes = []
    for run in runs:
        x1_run, x2_run = run[0], run[-1] + 1
        sub_ranges = split_run_at_valleys(proj, x1_run, x2_run, threshold, min_segment_width)
        for x1, x2 in sub_ranges:
            if x2 - x1 < min_width:
                continue
            # y-Grenzen aus dem Spaltenband bestimmen
            col_slice = binary[:, x1:x2]
            row_indices = np.where(col_slice.any(axis=1))[0]
            if len(row_indices) == 0:
                continue
            y1, y2 = int(row_indices[0]), int(row_indices[-1]) + 1
            boxes.append((y1, x1, y2, x2))

    return boxes


def component_summary_from_stats(stats: list, image_area: int) -> dict:
    """Compact component count + largest area ratio for stage-level diagnostics.

    Takes a pre-computed stats list (dicts with at least an 'area' key) and the
    total image pixel area (H*W).  Returns a dict with three keys:
      count         — number of components
      largest_area  — pixel area of the largest component (0 when stats is empty)
      largest_ratio — largest_area / image_area (0.0 when image_area <= 0 or empty)
    """
    if not stats or image_area <= 0:
        return {"count": len(stats) if stats else 0, "largest_area": 0, "largest_ratio": 0.0}
    largest = max(s["area"] for s in stats)
    return {
        "count": len(stats),
        "largest_area": largest,
        "largest_ratio": round(largest / image_area, 4),
    }


def filter_components_with_fallback(
    stats: list,
    image_height: int,
    image_width: int = 0,
    min_area: int = 30,
    colon_aspect_min: float = 2.0,
    colon_area_max: int = 200,
    max_area_ratio: float = 0.15,
) -> tuple:
    """Two-pass filter: normal rules first; if nothing kept, retry without background rule.

    Returns (kept, rejected, fallback_used) where fallback_used is True only when
    the second pass (background rule disabled) recovered at least one component.
    The noise and colon rules always apply in both passes.
    """
    kept, rejected = filter_components(
        stats, image_height, image_width, min_area, colon_aspect_min, colon_area_max, max_area_ratio
    )
    if kept or not stats:
        return kept, rejected, False

    # First pass kept nothing → retry with background-area rule disabled
    fallback_kept, fallback_rejected = filter_components(
        stats, image_height, image_width=0,
        min_area=min_area, colon_aspect_min=colon_aspect_min, colon_area_max=colon_area_max,
    )
    return fallback_kept, fallback_rejected, bool(fallback_kept)


def filter_components(
    stats: list,
    image_height: int,
    image_width: int = 0,
    min_area: int = 30,
    colon_aspect_min: float = 2.0,
    colon_area_max: int = 200,
    max_area_ratio: float = 0.15,
) -> tuple:
    """Entfernt Rausch-Blobs, Doppelpunkt-Punkte und Hintergrund-Outlier.

    Gibt (kept, rejected) zurück. Eingabe-Dicts werden nicht verändert.
    Jedes dict in `rejected` enthält ein 'rejection_reason'-Feld.
    Jedes dict in `kept` ist eine flache Kopie ohne 'rejection_reason'.
    """
    max_area = max_area_ratio * image_height * image_width if image_width > 0 else float("inf")

    kept: list = []
    rejected: list = []
    for s in stats:
        if s["area"] < min_area:
            reason = f"noise: area {s['area']} < {min_area}"
        elif s["aspect_ratio"] > colon_aspect_min and s["area"] < colon_area_max:
            reason = f"colon/dot: h/w={s['aspect_ratio']:.1f}>{colon_aspect_min}, area={s['area']}<{colon_area_max}"
        elif s["area"] > max_area:
            reason = f"background: area {s['area']} > {max_area:.0f}px ({max_area_ratio:.0%} of image)"
        else:
            reason = None

        if reason is None:
            kept.append(dict(s))
        else:
            rejected.append({**s, "rejection_reason": reason})

    return kept, rejected
