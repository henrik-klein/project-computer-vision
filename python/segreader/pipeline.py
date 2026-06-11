import numpy as np

from . import io, preprocess, morphology, labeling, segment, decode, backend as backend_mod


def run_pipeline(
    path: str,
    backend: str = "python",
    contrast_alpha: float = 1.5,
    morph_kernel_size: int = 3,
    projection_threshold_factor: float = 0.05,
    segment_threshold: float = 0.3,
    min_gap: int = 3,
    max_width: int = 800,
) -> tuple:
    """Vollständige 7-Segment-Pipeline.

    Gibt (number_string, annotiertes_BGR_Bild) zurück.
    """
    backend_mod.get_backend(backend)

    # Bild laden
    img_bgr = io.load_image(path)

    # Optional: große Bilder skalieren (reine Python-Labeling-Schleife sonst zu langsam)
    h, w = img_bgr.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_w, new_h = int(w * scale), int(h * scale)
        import cv2
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Punktoperation: Graustufenkonvertierung
    gray = preprocess.to_grayscale(img_bgr)

    # Punktoperation: Kontrastverstärkung
    gray = preprocess.adjust_contrast(gray, alpha=contrast_alpha)

    # Histogramm / Otsu-Schwellwert: Binarisierung
    binary = preprocess.otsu_binarize(gray)

    # Morphologische Filter: Opening entfernt Rauschen
    kernel = morphology.make_rect_kernel(morph_kernel_size, morph_kernel_size)
    binary = morphology.opening(binary, kernel)

    # Morphologische Filter: Closing schließt Lücken in Segmenten
    binary = morphology.closing(binary, kernel)

    # Sequential Labeling — Zwei-Pass mit Union-Find
    labels = labeling.label_components(binary)

    # Rauschen und Doppelpunkt herausfiltern
    stats = labeling.get_component_stats(labels)
    kept_stats = segment.filter_components(stats, binary.shape[0])

    # Bereinigte Binärmaske nur aus behaltenen Komponenten aufbauen
    # (verhindert Doppelpunkt-Phantome im Projektionsprofil)
    clean_binary = np.zeros_like(binary)
    for s in kept_stats:
        clean_binary[labels == s["label"]] = 1

    # Projektionsprofil zur Ziffern-Segmentierung
    proj = segment.vertical_projection(clean_binary)
    digit_boxes = segment.find_digit_boxes(
        clean_binary, proj,
        threshold_factor=projection_threshold_factor,
        min_gap=min_gap,
    )

    # 7-Bit-Muster → Lookup-Tabelle → Ziffer
    number_str, digits = decode.decode_number(clean_binary, digit_boxes, segment_threshold)

    annotated = io.draw_annotations(img_bgr, digit_boxes, digits, number_str)
    return number_str, annotated
