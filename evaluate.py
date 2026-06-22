"""Run the segment reader on all images in a directory and save results.

Usage:
    uv run python evaluate.py [image_dir] [options]

Options:
    -o / --output          Path for result JSON (default: data/testing/result.json)
    -b / --backend         python (default)
    -t / --threshold       Segment threshold 0.0-1.0 (default: 0.3)
    -w / --workers         Image-level worker threads (default: cpu count)

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
from segreader.steps_io import save_processed_steps

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

_print_lock = threading.Lock()


def image_to_b64(img) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Failed to encode image")
    return base64.b64encode(buf.tobytes()).decode()


def _process_image(path: str, stem: str, args) -> tuple[str, dict, str]:
    """Process one image and return (stem, entry_dict, log_line)."""
    number_str, annotated, steps = run_pipeline(
        path,
        backend=args.backend,
        segment_threshold=args.threshold,
        localize=not args.no_localize,
        return_steps=True,
    )

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

    log = f"-> {number_str!r}"
    return stem, entry, log


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
