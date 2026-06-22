"""
Projektionsprofil zur Ziffern-Segmentierung.
"""
import numpy as np


def vertical_projection(binary: np.ndarray) -> np.ndarray:
    # Projektionsprofil: Spaltensummen der Vordergrundpixel
    return binary.sum(axis=0).astype(np.float64)


def find_digit_boxes(
    binary: np.ndarray,
    projection: np.ndarray,
    threshold_factor: float = 0.05,
    min_gap: int = 3,
    min_width: int = 3,
) -> list:
    """Findet Ziffern-Bounding-Boxes über das vertikale Projektionsprofil.

    y-Grenzen werden pro Box aus den tatsächlich leuchtenden Pixeln im
    jeweiligen Spaltenband bestimmt (nicht das ganze Bild als y-Bereich).
    """
    threshold = threshold_factor * binary.shape[0]
    active_cols = np.where(projection >= threshold)[0]
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
        x1, x2 = run[0], run[-1] + 1
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
