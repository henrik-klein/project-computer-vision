"""Run the segment reader on all images in a directory and save results.

Usage:
    uv run python evaluate.py [image_dir] [options]

Options:
    -o / --output          Path for result JSON (default: data/testing/result.json)
    -b / --backend         python (default)
    -t / --threshold       Segment threshold 0.0-1.0 (default: 0.40)
    -w / --workers         Image-level worker threads (default: cpu count)
    --ground-truth         Path to samples.json; enables per-image and summary metrics
    --diagnostics          Add per-image pipeline diagnostics and extended summary
    --sweep-thresholds     Comma-separated thresholds to sweep (requires --ground-truth)

The result JSON embeds each overlay image as base64 so report.html is
self-contained and needs no HTTP server.
"""
import argparse
import base64
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "python"))

from segreader import run_pipeline
from segreader.metrics import (
    build_per_image_diagnostics,
    compare_prediction,
    compute_diagnostic_summary,
    compute_summary,
    compute_sweep_row,
    extract_pipeline_info,
    load_ground_truth,
    parse_thresholds,
)
from segreader.steps_io import save_processed_steps

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

# Each binarization variant: (binarization_mode, local_mean_window, local_mean_offset, label)
BINARIZATION_VARIANTS = [
    ("otsu_bright",        0,  0, "otsu_bright"),   # current default
    ("otsu_dark",          0,  0, "otsu_dark"),
    ("local_mean_dark",   31, 10, "lm_dark_w31"),
    ("local_mean_dark",   51, 10, "lm_dark_w51"),
    ("local_mean_dark",   71, 10, "lm_dark_w71"),
    ("local_mean_bright", 31, 10, "lm_bright_w31"),
]

# Each localization variant: (localize, method, fallback, useful_ratio, shrink, label)
LOCALIZATION_VARIANTS = [
    (True,  "auto",        True,  0.75, 0.00, "auto"),        # current default
    (False, "auto",        True,  0.75, 0.00, "none"),         # full image always
    (True,  "color",       True,  0.75, 0.00, "color"),
    (True,  "brightness",  True,  0.75, 0.00, "brightness"),
    (True,  "contour",     True,  0.75, 0.00, "contour"),
    (True,  "auto",        False, 0.75, 0.00, "auto_no_fb"),   # no '?' retry
    (True,  "auto",        True,  0.85, 0.00, "auto_r85"),     # relax area thresh to 85%
    (True,  "auto",        True,  0.90, 0.00, "auto_r90"),     # relax area thresh to 90%
    (True,  "auto",        True,  0.75, 0.05, "auto_sh5"),     # auto + 5% inward shrink
    (True,  "auto",        True,  0.75, 0.10, "auto_sh10"),    # auto + 10% inward shrink
    (True,  "contour",     True,  0.75, 0.05, "cont_sh5"),     # contour + 5% shrink
]

# Each variant: (morph_type, kernel_height, kernel_width, label)
MORPH_VARIANTS = [
    ("none",            1, 1, "none"),
    ("opening",         3, 3, "open3x3"),   # current default
    ("opening",         2, 2, "open2x2"),
    ("opening",         1, 3, "open1x3"),   # horizontal kernel
    ("opening",         3, 1, "open3x1"),   # vertical kernel
    ("closing",         3, 3, "close3x3"),
    ("close_then_open", 3, 3, "close+open"),
    ("open_then_close", 3, 3, "open+close"),
]

_print_lock = threading.Lock()


def image_to_b64(img) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Failed to encode image")
    return base64.b64encode(buf.tobytes()).decode()


def _process_image(
    path: str, stem: str, args, collect_pipeline_info: bool = False
) -> tuple[str, dict, str]:
    """Process one image; return (stem, entry_dict, log_line).

    When collect_pipeline_info is True, key pipeline metadata is extracted
    from the steps list and stored under entry["pipeline_info"] so that the
    caller can build per-image diagnostics without reading back the steps JSON.
    """
    number_str, annotated, steps = run_pipeline(
        path,
        backend=args.backend,
        segment_threshold=args.threshold,
        localize=not args.no_localize,
        return_steps=True,
    )

    pipeline_info = extract_pipeline_info(steps) if collect_pipeline_info else None

    steps_json_path = None
    try:
        steps_json_path = save_processed_steps(
            steps, path, number_str,
            {"segment_threshold": args.threshold, "backend": args.backend},
        )
    except Exception:
        pass

    entry = {
        "predicted":   number_str,
        "overlay_b64": image_to_b64(annotated),
        "steps_json":  steps_json_path,
    }
    if pipeline_info is not None:
        entry["pipeline_info"] = pipeline_info

    log = f"-> {number_str!r}"
    return stem, entry, log


def _quick_process_image(
    path: str, stem: str, backend: str, threshold: float, no_localize: bool
) -> tuple[str, dict]:
    """Lightweight pipeline run: no step saving, no image encoding. For sweep only."""
    try:
        number_str, _ = run_pipeline(
            path,
            backend=backend,
            segment_threshold=threshold,
            localize=not no_localize,
            return_steps=False,
        )
        return stem, {"predicted": number_str}
    except Exception as exc:
        return stem, {"predicted": None, "error": str(exc)}


def _run_sweep(
    image_files: list[str],
    image_dir: str,
    ground_truth: dict,
    thresholds: list[float],
    args,
    n_workers: int,
) -> list[dict]:
    """Run lightweight evaluation at each threshold; return list of sweep rows."""
    rows = []
    for threshold in thresholds:
        results: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            future_to_stem = {
                ex.submit(
                    _quick_process_image,
                    os.path.join(image_dir, f),
                    os.path.splitext(f)[0],
                    args.backend,
                    threshold,
                    args.no_localize,
                ): os.path.splitext(f)[0]
                for f in image_files
            }
            for future in as_completed(future_to_stem):
                stem_r, entry = future.result()
                results[stem_r] = entry
        rows.append(compute_sweep_row(results, ground_truth, threshold))
    rows.sort(key=lambda r: r["threshold"])
    return rows


def _print_sweep_table(rows: list[dict]) -> None:
    header = (
        f"  {'thresh':>6}  {'correct':>7}  {'eval':>5}  "
        f"{'acc%':>6}  {'?-digits':>8}  {'len-mm':>6}  {'wrong':>5}"
    )
    print()
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        acc = f"{r['accuracy_pct']:.1f}" if r["accuracy_pct"] is not None else "n/a"
        print(
            f"  {r['threshold']:.2f}    {r['correct']:>6}   {r['evaluated']:>4}  "
            f"{acc:>5}%   {r['contains_unknown_count']:>7}    "
            f"{r['length_mismatch_count']:>5}  {r['wrong_digits_count']:>5}"
        )


def _quick_process_morph(
    path: str, stem: str, backend: str, threshold: float, no_localize: bool,
    morph_type: str, kh: int, kw: int,
) -> tuple[str, dict]:
    """Lightweight pipeline run for the morphology sweep."""
    try:
        number_str, _ = run_pipeline(
            path, backend=backend, segment_threshold=threshold,
            localize=not no_localize, return_steps=False,
            morph_type=morph_type, morph_kernel_height=kh, morph_kernel_width=kw,
        )
        return stem, {"predicted": number_str}
    except Exception as exc:
        return stem, {"predicted": None, "error": str(exc)}


def _run_morph_sweep(
    image_files: list[str],
    image_dir: str,
    ground_truth: dict,
    args,
    n_workers: int,
) -> list[dict]:
    """Run lightweight evaluation for each morphology variant."""
    rows = []
    for morph_type, kh, kw, label in MORPH_VARIANTS:
        results: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            future_to_stem = {
                ex.submit(
                    _quick_process_morph,
                    os.path.join(image_dir, f),
                    os.path.splitext(f)[0],
                    args.backend,
                    args.threshold,
                    args.no_localize,
                    morph_type, kh, kw,
                ): os.path.splitext(f)[0]
                for f in image_files
            }
            for future in as_completed(future_to_stem):
                stem_r, entry = future.result()
                results[stem_r] = entry
        row = compute_sweep_row(results, ground_truth, args.threshold)
        row["label"] = label
        rows.append(row)
    return rows


def _print_morph_table(rows: list[dict]) -> None:
    header = (
        f"  {'variant':<12}  {'correct':>7}  {'eval':>5}  "
        f"{'acc%':>6}  {'?-digits':>8}  {'len-mm':>6}  {'wrong':>5}"
    )
    print()
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        acc = f"{r['accuracy_pct']:.1f}" if r["accuracy_pct"] is not None else "n/a"
        marker = " <--current" if r["label"] == "open3x3" else ""
        print(
            f"  {r['label']:<12}  {r['correct']:>6}   {r['evaluated']:>4}  "
            f"{acc:>5}%   {r['contains_unknown_count']:>7}    "
            f"{r['length_mismatch_count']:>5}  {r['wrong_digits_count']:>5}{marker}"
        )


def _quick_process_binarize(
    path: str, stem: str, backend: str, threshold: float, no_localize: bool,
    binarization: str, lm_window: int, lm_offset: int,
) -> tuple[str, dict]:
    """Lightweight pipeline run for the binarization sweep."""
    try:
        number_str, _ = run_pipeline(
            path, backend=backend, segment_threshold=threshold,
            localize=not no_localize, return_steps=False,
            binarization=binarization,
            local_mean_window=lm_window, local_mean_offset=lm_offset,
        )
        return stem, {"predicted": number_str}
    except Exception as exc:
        return stem, {"predicted": None, "error": str(exc)}


def _run_binarization_sweep(
    image_files: list[str],
    image_dir: str,
    ground_truth: dict,
    args,
    n_workers: int,
) -> list[dict]:
    """Run lightweight evaluation for each binarization variant."""
    rows = []
    for binarization, lm_window, lm_offset, label in BINARIZATION_VARIANTS:
        results: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            future_to_stem = {
                ex.submit(
                    _quick_process_binarize,
                    os.path.join(image_dir, f),
                    os.path.splitext(f)[0],
                    args.backend,
                    args.threshold,
                    args.no_localize,
                    binarization, lm_window, lm_offset,
                ): os.path.splitext(f)[0]
                for f in image_files
            }
            for future in as_completed(future_to_stem):
                stem_r, entry = future.result()
                results[stem_r] = entry
        row = compute_sweep_row(results, ground_truth, args.threshold)
        row["label"] = label
        rows.append(row)
    return rows


def _print_binarization_table(rows: list[dict]) -> None:
    header = (
        f"  {'variant':<14}  {'correct':>7}  {'eval':>5}  "
        f"{'acc%':>6}  {'?-digits':>8}  {'len-mm':>6}  {'wrong':>5}"
    )
    print()
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        acc = f"{r['accuracy_pct']:.1f}" if r["accuracy_pct"] is not None else "n/a"
        marker = " <--current" if r["label"] == "otsu_bright" else ""
        print(
            f"  {r['label']:<14}  {r['correct']:>6}   {r['evaluated']:>4}  "
            f"{acc:>5}%   {r['contains_unknown_count']:>7}    "
            f"{r['length_mismatch_count']:>5}  {r['wrong_digits_count']:>5}{marker}"
        )


def _quick_process_localize(
    path: str, stem: str, backend: str, threshold: float,
    localize: bool, localize_method: str, localize_fallback: bool,
    localize_useful_ratio: float, localize_shrink: float,
) -> tuple[str, dict]:
    """Lightweight pipeline run for the localization sweep."""
    try:
        number_str, _ = run_pipeline(
            path, backend=backend, segment_threshold=threshold,
            localize=localize,
            localize_method=localize_method,
            localize_fallback=localize_fallback,
            localize_useful_ratio=localize_useful_ratio,
            localize_shrink=localize_shrink,
            return_steps=False,
        )
        return stem, {"predicted": number_str}
    except Exception as exc:
        return stem, {"predicted": None, "error": str(exc)}


def _run_localization_sweep(
    image_files: list[str],
    image_dir: str,
    ground_truth: dict,
    args,
    n_workers: int,
) -> list[dict]:
    """Run lightweight evaluation for each localization variant."""
    rows = []
    for loc, method, fallback, useful_ratio, shrink, label in LOCALIZATION_VARIANTS:
        results: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            future_to_stem = {
                ex.submit(
                    _quick_process_localize,
                    os.path.join(image_dir, f),
                    os.path.splitext(f)[0],
                    args.backend,
                    args.threshold,
                    loc, method, fallback, useful_ratio, shrink,
                ): os.path.splitext(f)[0]
                for f in image_files
            }
            for future in as_completed(future_to_stem):
                stem_r, entry = future.result()
                results[stem_r] = entry
        row = compute_sweep_row(results, ground_truth, args.threshold)
        row["label"] = label
        rows.append(row)
    return rows


def _print_localization_table(rows: list[dict]) -> None:
    header = (
        f"  {'variant':<12}  {'correct':>7}  {'eval':>5}  "
        f"{'acc%':>6}  {'?-digits':>8}  {'len-mm':>6}  {'wrong':>5}"
    )
    print()
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        acc = f"{r['accuracy_pct']:.1f}" if r["accuracy_pct"] is not None else "n/a"
        marker = " <--current" if r["label"] == "auto" else ""
        print(
            f"  {r['label']:<12}  {r['correct']:>6}   {r['evaluated']:>4}  "
            f"{acc:>5}%   {r['contains_unknown_count']:>7}    "
            f"{r['length_mismatch_count']:>5}  {r['wrong_digits_count']:>5}{marker}"
        )


def main() -> None:
    cpu = os.cpu_count() or 4

    parser = argparse.ArgumentParser(description="Evaluate segment reader on a test set")
    parser.add_argument("image_dir", nargs="?", default=os.path.join("data", "testing", "images"))
    parser.add_argument("-o", "--output", default=os.path.join("data", "testing", "result.json"))
    parser.add_argument("-b", "--backend", default="python")
    parser.add_argument("-t", "--threshold", type=float, default=0.40)
    parser.add_argument(
        "-w", "--workers", type=int, default=None,
        help=f"Anzahl paralleler Bild-Worker (Standard: CPU-Anzahl = {cpu})",
    )
    parser.add_argument(
        "--no-localize", action="store_true",
        help="Skip display localization — process full image directly",
    )
    parser.add_argument(
        "--ground-truth", default=None, metavar="PATH",
        help="Path to samples.json (stem->expected); enables per-image correctness and summary",
    )
    parser.add_argument(
        "--diagnostics", action="store_true",
        help="Add per-image pipeline diagnostics and extended error-category summary",
    )
    parser.add_argument(
        "--sweep-thresholds", default=None, metavar="T1,T2,...",
        help="Comma-separated thresholds to sweep (requires --ground-truth); skips normal eval",
    )
    parser.add_argument(
        "--sweep-morphology", action="store_true",
        help="Sweep morphology variants (requires --ground-truth); skips normal eval",
    )
    parser.add_argument(
        "--sweep-binarization", action="store_true",
        help="Sweep binarization variants (requires --ground-truth); skips normal eval",
    )
    parser.add_argument(
        "--sweep-localization", action="store_true",
        help="Sweep localization variants (requires --ground-truth); skips normal eval",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        print(f"Directory not found: {args.image_dir}", file=sys.stderr)
        sys.exit(1)

    image_files = sorted(
        f for f in os.listdir(args.image_dir)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    )

    if not image_files:
        print("No images found.", file=sys.stderr)
        sys.exit(1)

    n_workers = min(args.workers or cpu, len(image_files))

    # ── Localization sweep mode ───────────────────────────────────────────────
    if args.sweep_localization:
        if not args.ground_truth:
            print("--sweep-localization requires --ground-truth", file=sys.stderr)
            sys.exit(1)
        try:
            gt_data = load_ground_truth(args.ground_truth)
        except Exception as exc:
            print(f"Could not load ground truth: {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"Localization sweep: {len(LOCALIZATION_VARIANTS)} variant(s) x {len(image_files)} image(s) ...")
        rows = _run_localization_sweep(image_files, args.image_dir, gt_data, args, n_workers)
        _print_localization_table(rows)
        return

    # ── Binarization sweep mode ───────────────────────────────────────────────
    if args.sweep_binarization:
        if not args.ground_truth:
            print("--sweep-binarization requires --ground-truth", file=sys.stderr)
            sys.exit(1)
        try:
            gt_data = load_ground_truth(args.ground_truth)
        except Exception as exc:
            print(f"Could not load ground truth: {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"Binarization sweep: {len(BINARIZATION_VARIANTS)} variant(s) x {len(image_files)} image(s) ...")
        rows = _run_binarization_sweep(image_files, args.image_dir, gt_data, args, n_workers)
        _print_binarization_table(rows)
        return

    # ── Morphology sweep mode ─────────────────────────────────────────────────
    if args.sweep_morphology:
        if not args.ground_truth:
            print("--sweep-morphology requires --ground-truth", file=sys.stderr)
            sys.exit(1)
        try:
            gt_data = load_ground_truth(args.ground_truth)
        except Exception as exc:
            print(f"Could not load ground truth: {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"Morphology sweep: {len(MORPH_VARIANTS)} variant(s) × {len(image_files)} image(s) …")
        rows = _run_morph_sweep(image_files, args.image_dir, gt_data, args, n_workers)
        _print_morph_table(rows)
        return

    # ── Threshold sweep mode (separate from normal eval) ──────────────────────
    if args.sweep_thresholds:
        if not args.ground_truth:
            print("--sweep-thresholds requires --ground-truth", file=sys.stderr)
            sys.exit(1)
        try:
            thresholds = parse_thresholds(args.sweep_thresholds)
        except ValueError as exc:
            print(f"Invalid --sweep-thresholds: {exc}", file=sys.stderr)
            sys.exit(1)
        try:
            gt_data = load_ground_truth(args.ground_truth)
        except Exception as exc:
            print(f"Could not load ground truth: {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"Sweep: {len(thresholds)} threshold(s) × {len(image_files)} image(s) …")
        rows = _run_sweep(image_files, args.image_dir, gt_data, thresholds, args, n_workers)
        _print_sweep_table(rows)
        return  # sweep is its own mode

    # ── Normal evaluation mode ────────────────────────────────────────────────
    collect_info = bool(args.diagnostics)
    print(f"Processing {len(image_files)} image(s) with {n_workers} worker(s) …\n")

    results: dict[str, dict] = {}
    errors = 0

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        future_map = {
            ex.submit(
                _process_image,
                os.path.join(args.image_dir, filename),
                os.path.splitext(filename)[0],
                args,
                collect_info,
            ): filename
            for filename in image_files
        }
        for future in as_completed(future_map):
            filename = future_map[future]
            try:
                stem, entry, log = future.result()
                results[stem] = entry
                with _print_lock:
                    print(f"  {filename} {log}")
            except Exception as exc:
                stem = os.path.splitext(filename)[0]
                results[stem] = {"predicted": None, "error": str(exc), "overlay_b64": None}
                errors += 1
                with _print_lock:
                    print(f"  {filename} ERROR: {exc}")

    # Restore original file order in the output JSON.
    ordered = {
        os.path.splitext(f)[0]: results[os.path.splitext(f)[0]]
        for f in image_files
        if os.path.splitext(f)[0] in results
    }

    # ── Ground-truth comparison ───────────────────────────────────────────────
    ground_truth = None
    summary = None
    if args.ground_truth:
        try:
            ground_truth = load_ground_truth(args.ground_truth)
        except Exception as exc:
            print(f"\nWarning: could not load ground truth: {exc}", file=sys.stderr)

    if ground_truth is not None:
        for stem, entry in ordered.items():
            if stem in ground_truth:
                entry.update(compare_prediction(entry.get("predicted"), ground_truth[stem]))
        summary = compute_summary(ordered, ground_truth)

    # ── Per-image diagnostics enrichment ──────────────────────────────────────
    if args.diagnostics:
        for stem, entry in ordered.items():
            expected = ground_truth.get(stem) if ground_truth is not None else None
            diag = build_per_image_diagnostics(entry, expected)
            entry.update(diag)
            entry.pop("pipeline_info", None)  # absorbed into diag fields
        if ground_truth is not None:
            summary = compute_diagnostic_summary(ordered, ground_truth)

    # ── Assemble output JSON ──────────────────────────────────────────────────
    meta = {
        "timestamp":  datetime.now().isoformat(timespec="seconds"),
        "image_dir":  args.image_dir,
        "backend":    args.backend,
        "threshold":  args.threshold,
        "workers":    n_workers,
    }
    if args.ground_truth:
        meta["ground_truth"] = args.ground_truth
    if args.diagnostics:
        meta["diagnostics"] = True

    output: dict = {"meta": meta}
    if summary is not None:
        output["summary"] = summary
    output["results"] = ordered

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f)

    total = len(results)
    suffix = ""
    if summary is not None:
        acc = (f"{summary['accuracy_pct']}%" if summary["accuracy_pct"] is not None
               else "n/a")
        suffix = f", {summary['correct']}/{summary['evaluated']} correct ({acc})"
    print(f"\nDone. {total} image(s), {errors} error(s){suffix}. Saved to {args.output}")


if __name__ == "__main__":
    main()
