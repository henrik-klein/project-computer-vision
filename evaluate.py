"""Run the segment reader on all images in a directory and save results.

Usage:
    uv run python evaluate.py [image_dir] [options]

Options:
    -o / --output          Path for result JSON (default: data/testing/result.json)
    -b / --backend         python (default)
    -t / --threshold       Segment threshold 0.0-1.0 (default: 0.3)
    -w / --workers         Image-level worker threads (default: cpu count)
    --all-combinations     Try every localize method × channel and store results

The result JSON embeds each overlay image as base64 so report.html is
self-contained and needs no HTTP server.
"""
import argparse
import base64
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "python"))

from segreader import run_pipeline, run_all_combinations

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

_print_lock = threading.Lock()


def image_to_b64(img) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Failed to encode image")
    return base64.b64encode(buf.tobytes()).decode()


def _process_image(path: str, stem: str, args) -> tuple[str, dict, str]:
    """Process one image: pipeline + optional all-combinations sweep.

    Returns (stem, entry_dict, log_line).
    """
    number_str, annotated, details = run_pipeline(
        path,
        backend=args.backend,
        localize=args.localize,
        segment_threshold=args.threshold,
        min_digit_width_ratio=args.min_digit_width,
        return_meta=True,
    )
    meta = details["meta"]
    attempts_serialized = [
        {
            "localize_method": a["localize_method"],
            "threshold":       a["threshold"],
            "channel":         a["channel"],
            "number_str":      a["number_str"],
            "overlay_b64":     image_to_b64(a["annotated"]),
        }
        for a in details["attempts"]
    ]
    localize_sel_serialized = [
        {
            "method":    c["method"],
            "found":     c["found"],
            "coverage":  c["coverage"],
            "probe_str": c["probe_str"],
            "chosen":    c["chosen"],
            "debug_b64": image_to_b64(c["debug"]),
        }
        for c in details.get("localize_selection", [])
    ]
    tophat_viz = details.get("tophat_viz")
    # Generate steps.json via main.py --no-save --processed
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _main_script = os.path.join(_script_dir, "main.py")
    steps_json_path = None
    try:
        subprocess.run(
            [sys.executable, _main_script, path, "--no-save", "--processed"],
            check=True, capture_output=True, cwd=_script_dir,
        )
        _candidate = os.path.join("data", "processed", stem, "steps.json")
        if os.path.isfile(os.path.join(_script_dir, _candidate)):
            steps_json_path = _candidate
    except Exception:
        pass

    entry = {
        "predicted":          number_str,
        "overlay_b64":        image_to_b64(annotated),
        "meta":               meta,
        "attempts":           attempts_serialized,
        "localize_selection": localize_sel_serialized,
        "tophat_b64":         image_to_b64(tophat_viz) if tophat_viz is not None else None,
        "steps_json":         steps_json_path,
    }

    if args.all_combinations:
        combos = run_all_combinations(
            path,
            backend=args.backend,
            segment_threshold=args.threshold,
            min_digit_width_ratio=args.min_digit_width,
            combo_workers=args.combo_workers,
        )
        entry["all_combinations"] = [
            {
                "localize_method":   c["localize_method"],
                "channel":           c["channel"],
                "threshold":         c["threshold"],
                "number_str":        c["number_str"],
                "overlay_b64":       image_to_b64(c["annotated"]),
                "threshold_results": c["threshold_results"],
            }
            for c in combos
        ]

    log = (
        f"→ {number_str!r}  "
        f"(method={meta['localize_method']} t={meta['threshold']} "
        f"ch={meta['channel']} attempts={len(attempts_serialized)})"
    )
    return stem, entry, log


def main() -> None:
    cpu = os.cpu_count() or 4

    parser = argparse.ArgumentParser(description="Evaluate segment reader on a test set")
    parser.add_argument("image_dir", nargs="?", default=os.path.join("data", "testing", "images"))
    parser.add_argument("-o", "--output", default=os.path.join("data", "testing", "result.json"))
    parser.add_argument("-b", "--backend", default="python")
    parser.add_argument("-t", "--threshold", type=float, default=0.3)
    parser.add_argument("--min-digit-width", type=float, default=0.08,
                        help="Mindestbreite einer Ziffern-Box relativ zur Bildhöhe (Standard: 0.08)")
    parser.add_argument("-l", "--localize", action="store_true",
                        help="Display-Region vor der Dekodierung automatisch lokalisieren")
    parser.add_argument("--localize-method", default="auto",
                        choices=["brightness", "color", "contour", "auto"],
                        help="Lokalisierungsstrategie: auto (Standard), brightness, color, contour")
    parser.add_argument("--all-combinations", action="store_true",
                        help="Alle Lokalisierungsmethoden × Kanäle evaluieren und im JSON speichern")
    parser.add_argument(
        "-w", "--workers", type=int, default=None,
        help=f"Anzahl paralleler Bild-Worker (Standard: CPU-Anzahl = {cpu})",
    )
    args = parser.parse_args()

    # Number of threads to use inside run_all_combinations per image.
    # With N image workers each spawning combo_workers threads the total is bounded.
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
    # Keep total thread count ≤ cpu * 4 so we don't flood the OS scheduler.
    args.combo_workers = max(1, (cpu * 4) // n_workers)

    combo_info = f", {args.combo_workers} combo-threads/image" if args.all_combinations else ""
    print(f"Processing {len(image_files)} image(s) with {n_workers} worker(s){combo_info} …\n")

    results: dict[str, dict] = {}
    errors = 0

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        future_map = {
            ex.submit(
                _process_image,
                os.path.join(args.image_dir, filename),
                os.path.splitext(filename)[0],
                args,
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

    output = {
        "meta": {
            "timestamp":  datetime.now().isoformat(timespec="seconds"),
            "image_dir":  args.image_dir,
            "backend":    args.backend,
            "threshold":  args.threshold,
            "workers":    n_workers,
        },
        "results": ordered,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f)

    total = len(results)
    print(f"\nDone. {total} image(s), {errors} error(s). Saved to {args.output}")


if __name__ == "__main__":
    main()
