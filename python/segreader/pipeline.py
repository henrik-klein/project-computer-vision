from __future__ import annotations

import cv2
import numpy as np

from . import io, preprocess, morphology, labeling, segment, decode, backend as backend_mod
from .decode import SEGMENT_ZONES
from .localize import find_display
from .visualisation import visualisation



# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(
    path: str,
    backend: str = "python",
    contrast_alpha: float = 1.5,
    morph_kernel_size: int = 3,
    projection_threshold_factor: float = 0.05,
    segment_threshold: float = 0.3,
    min_gap: int = 1,
    max_width: int = 800,
    return_steps: bool = False,
    localize: bool = True,
) -> tuple:
    """Vollständige 7-Segment-Pipeline.

    Gibt (number_string, annotiertes_BGR_Bild) zurück.
    Mit return_steps=True: (number_string, annotiertes_BGR_Bild, steps).
    """
    backend_mod.get_backend(backend)

    steps: list[dict] = []

    def _step(id: str, name: str, desc: str, image: np.ndarray, details: dict, **extra) -> None:
        if return_steps:
            steps.append({"id": id, "name": name, "description": desc,
                          "image": image, "details": details, **extra})

    # ── Bild laden ────────────────────────────────────────────────────────────
    img_bgr = io.load_image(path)
    h0, w0 = img_bgr.shape[:2]

    # ── Optional: große Bilder skalieren ──────────────────────────────────────
    was_resized = w0 > max_width
    if was_resized:
        scale = max_width / w0
        new_w, new_h = int(w0 * scale), int(h0 * scale)
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    _step("original", "Original Image",
          "Raw input image as loaded from disk."
          + (f" Down-scaled from {w0}→{img_bgr.shape[1]} px (max_width={max_width})."
             if was_resized else " No resize needed."),
          img_bgr.copy(),
          {"width": img_bgr.shape[1], "height": img_bgr.shape[0],
           "original_width": w0, "was_resized": was_resized})

    # ── Lokalisierung der Anzeigeregion ───────────────────────────────────────
    if localize:
        crop, debug_img, found, method_used, crop_offset = find_display(img_bgr, method="auto")
    else:
        crop, debug_img, found, method_used, crop_offset = img_bgr.copy(), img_bgr.copy(), False, "disabled", (0, 0)

    img_area = img_bgr.shape[0] * img_bgr.shape[1]
    crop_area = crop.shape[0] * crop.shape[1]
    coverage = round(crop_area / img_area, 3)

    _step("localize", "Display Localization",
          "Localization disabled — full image used." if not localize else
          "Auto strategy tries color → brightness → contour in order. "
          "A method is accepted only if its crop covers less than 75 % of the original image — "
          "larger crops mean nothing was actually isolated and the next method is tried. "
          f"Winner: '{method_used}' ({'found' if found else 'not found — full image used'}, "
          f"crop covers {coverage:.1%} of original).",
          visualisation.localize(debug_img, method_used, found),
          {"method_used": method_used, "found": found,
           "crop_width": crop.shape[1], "crop_height": crop.shape[0],
           "coverage": coverage})

    # ── Punktoperation: Graustufenkonvertierung ───────────────────────────────
    gray = preprocess.to_grayscale(crop)

    _step("grayscale", "Convert to Grayscale",
          "Weighted luma sum Y = 0.114·B + 0.587·G + 0.299·R. "
          "Colour is irrelevant for segment detection; only brightness matters.",
          visualisation.gray(gray),
          {"formula": "Y = 0.114·B + 0.587·G + 0.299·R",
           "min": int(gray.min()), "max": int(gray.max()), "mean": round(float(gray.mean()), 1)})

    # ── Punktoperation: Kontrastverstärkung ───────────────────────────────────
    gray = preprocess.adjust_contrast(gray, alpha=contrast_alpha)

    _step("contrast", "Linear Contrast Stretch",
          f"Point op: output = clip(α·input, 0, 255) with α={contrast_alpha}. "
          "Spreads the histogram so dark backgrounds approach 0 and bright segments approach 255.",
          visualisation.gray(gray),
          {"alpha": contrast_alpha,
           "min": int(gray.min()), "max": int(gray.max()), "mean": round(float(gray.mean()), 2)})

    # ── Histogramm / Otsu-Schwellwert: Binarisierung ─────────────────────────
    otsu_t = visualisation.otsu_threshold(gray) if return_steps else 0
    binary = preprocess.otsu_binarize(gray)
    fg_ratio = float(binary.mean())
    inverted = fg_ratio > 0.5

    _step("binary", "Otsu Binarisation",
          "Otsu picks T that maximises between-class variance σ²_B = w₀·w₁·(μ₀−μ₁)². "
          f"T={otsu_t}. Foreground ratio {fg_ratio:.1%} → "
          f"{'polarity inverted (bright background)' if inverted else 'no inversion needed'}.",
          visualisation.binary(binary),
          {"otsu_threshold": otsu_t, "foreground_ratio": round(fg_ratio, 4),
           "polarity_inverted": inverted, "foreground_pixels": int(binary.sum())})

    # ── Morphologische Filter: Opening entfernt Rauschen ─────────────────────
    kernel = morphology.make_rect_kernel(morph_kernel_size, morph_kernel_size)
    px_before = int(binary.sum())
    binary = morphology.opening(binary, kernel)
    px_after_open = int(binary.sum())

    _step("opening", "Morphological Opening (Denoise)",
          f"opening = dilate(erode(B,K), K) with {morph_kernel_size}×{morph_kernel_size} kernel. "
          "Erosion destroys blobs smaller than K; dilation restores survivors. "
          "Permanently removes isolated noise pixels.",
          visualisation.binary(binary),
          {"kernel_size": morph_kernel_size, "pixels_before": px_before,
           "pixels_after": px_after_open, "pixels_removed": px_before - px_after_open})

    # ── Sequential Labeling — Zwei-Pass mit Union-Find ────────────────────────
    labels = labeling.label_components(binary)
    all_stats = labeling.get_component_stats(labels)
    kept_stats, rejected_stats = segment.filter_components(all_stats, binary.shape[0], binary.shape[1])
    kept_set = {s["label"] for s in kept_stats}

    _step("components", "Connected Component Labeling",
          "Two-pass Union-Find (4-connectivity) assigns a unique label to every connected blob. "
          "Each colour is one component. Grey = will be rejected.",
          visualisation.components(labels, all_stats, kept_set),
          {"total_components": len(all_stats),
           "components": [{"label": s["label"], "area": s["area"],
                           "bbox": list(s["bbox"]),
                           "aspect_ratio": round(s["aspect_ratio"], 2),
                           "kept": s["label"] in kept_set}
                          for s in all_stats]})

    # ── Rauschen und Doppelpunkt herausfiltern ────────────────────────────────
    rejected = rejected_stats

    _step("filtering", "Component Filtering",
          "Rule ①: area < 30 px → noise (purple). "
          "Rule ②: h/w > 2.0 AND area < 200 → colon/dot (purple). "
          "Rule ③: area > 15 % of image → background outlier (orange). "
          "Green = kept.",
          visualisation.filter_result(labels, kept_stats + rejected_stats, kept_set),
          {"total": len(all_stats), "kept": len(kept_stats),
           "rejected_count": len(rejected),
           "rejected": [{"label": s["label"], "area": s["area"],
                         "aspect_ratio": round(s["aspect_ratio"], 2),
                         "reason": visualisation.rejection_reason(s)}
                        for s in rejected]})

    # ── Bereinigte Binärmaske nur aus behaltenen Komponenten aufbauen ─────────
    clean_binary = np.zeros_like(binary)
    for s in kept_stats:
        clean_binary[labels == s["label"]] = 1

    _step("clean", "Clean Binary Image",
          "Only kept components remain. This is the image fed into digit segmentation.",
          visualisation.binary(clean_binary),
          {"foreground_pixels": int(clean_binary.sum()),
           "foreground_ratio": round(float(clean_binary.mean()), 4)})

    # ── Projektionsprofil zur Ziffern-Segmentierung ───────────────────────────
    proj = segment.vertical_projection(clean_binary)
    thresh_px = projection_threshold_factor * clean_binary.shape[0]
    digit_boxes = segment.find_digit_boxes(
        clean_binary, proj,
        threshold_factor=projection_threshold_factor,
        min_gap=min_gap,
    )

    _step("projection", "Vertical Projection Profile",
          "proj[x] = Σ_y clean[y,x]. Peaks = digit columns; valleys = inter-digit gaps. "
          f"Threshold = {projection_threshold_factor}·h = {thresh_px:.1f} px (blue). "
          "Connected above-threshold runs → digit boxes (yellow).",
          visualisation.projection(proj, thresh_px, digit_boxes),
          {"threshold_factor": projection_threshold_factor,
           "threshold_px": round(thresh_px, 2),
           "max_projection": round(float(proj.max()), 1),
           "num_digit_boxes": len(digit_boxes)})

    # ── 7-Bit-Muster → Lookup-Tabelle → Ziffer ───────────────────────────────
    number_str, digits = decode.decode_number(clean_binary, digit_boxes, segment_threshold)

    _step("boxes", "Digit Bounding Boxes",
          "Each projection run → (y1, x1, y2, x2). Vertical bounds come from actual "
          "pixel rows in the column band, not the full image height.",
          visualisation.boxes(clean_binary, digit_boxes, digits),
          {"num_digits": len(digit_boxes),
           "boxes": [{"index": i, "y1": int(y1), "x1": int(x1), "y2": int(y2), "x2": int(x2)}
                     for i, (y1, x1, y2, x2) in enumerate(digit_boxes)]})

    # ── Segment-Zonen-Auswertung (nur für return_steps) ──────────────────────
    if return_steps:
        digit_records: list[dict] = []
        for i, (y1, x1, y2, x2) in enumerate(digit_boxes):
            patch = clean_binary[y1:y2, x1:x2]
            Hp, Wp = patch.shape
            zones_data: dict[str, dict] = {}
            for seg, (ys, ye, xs, xe) in SEGMENT_ZONES.items():
                r1 = max(int(ys * Hp), 0); r2 = max(int(ye * Hp), 1)
                c1 = max(int(xs * Wp), 0); c2 = max(int(xe * Wp), 1)
                z = patch[r1:r2, c1:c2]
                fill = float(z.mean()) if z.size > 0 else 0.0
                zones_data[seg] = {"fill": round(fill, 4),
                                   "active": fill >= segment_threshold}
            digit_records.append({
                "index": i,
                "box": {"y1": int(y1), "x1": int(x1), "y2": int(y2), "x2": int(x2)},
                "patch": visualisation.binary(patch),
                "zone_vis": visualisation.zone_overlay(patch, zones_data),
                "zones": zones_data,
                "active_segments": sorted(s for s, info in zones_data.items() if info["active"]),
                "narrow_digit_shortcut": Wp > 0 and Hp / Wp > 3,
                "decoded": digits[i],
            })
        steps.append({
            "id": "decoding",
            "name": "7-Segment Zone Sampling & Lookup",
            "description": (
                "Each digit patch is divided into 7 fixed zones (a–g). "
                f"Zone mean ≥ {segment_threshold} → segment ON. "
                "Active-segment frozenset is looked up in a 10-entry table → digit. "
                "Special case: height/width > 3 → shortcut classify as '1'."
            ),
            "image": visualisation.boxes(clean_binary, digit_boxes, digits),
            "digits": digit_records,
            "details": {"segment_threshold": segment_threshold,
                        "num_digits": len(digit_boxes)},
        })

    # ── Annotiertes Ergebnisbild — drawn on the full (non-localized) image ───
    ox, oy = crop_offset
    full_boxes = [(y1 + oy, x1 + ox, y2 + oy, x2 + ox) for y1, x1, y2, x2 in digit_boxes]
    annotated = io.draw_annotations(img_bgr, full_boxes, digits, number_str)

    _step("result", "Final Result",
          f"Decoded: {number_str}. Green boxes = recognised digits; red = unknown ('?').",
          annotated.copy(),
          {"number": number_str, "digits": digits, "has_unknowns": "?" in number_str})

    # ── Fallback: retry without localization if result contains unknowns ─────
    if localize and "?" in number_str:
        return run_pipeline(
            path,
            backend=backend,
            contrast_alpha=contrast_alpha,
            morph_kernel_size=morph_kernel_size,
            projection_threshold_factor=projection_threshold_factor,
            segment_threshold=segment_threshold,
            min_gap=min_gap,
            max_width=max_width,
            return_steps=return_steps,
            localize=False,
        )

    if return_steps:
        return number_str, annotated, steps
    return number_str, annotated
