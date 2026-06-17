"""Sieben-Segment-Reader — Einstiegspunkt.

Verwendung:
    python main.py <bildpfad> [Optionen]

Optionen:
    -b / --backend      python (Standard)
    -o / --output       Pfad für das annotierte Ausgabebild
    -t / --threshold    Segment-Schwellwert 0.0-1.0 (Standard: 0.3)
    -n / --no-save      Annotiertes Bild nicht speichern
    -v / --verbose      Einzelne Ziffern ausgeben
    -p / --processed    Zwischenschritte als PNG + steps.json nach data/processed/<name>/ speichern

Exit-Codes: 0=Erfolg, 1=Datei nicht gefunden, 2=keine Ziffern, 3=unbekanntes Backend
"""
import argparse
import base64
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "python"))

from segreader import run_pipeline
from segreader.io import save_image


def _b64(img) -> str:
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode()


def _save_processed(steps: list, image_path: str) -> str:
    """Save each step as PNG and write steps.json. Returns the output directory."""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.join("data", "processed", stem)
    os.makedirs(out_dir, exist_ok=True)

    json_steps = []
    for i, step in enumerate(steps):
        # Save PNG
        if "image" in step and step["image"] is not None:
            fname = f"{i:02d}_{step['id']}.png"
            cv2.imwrite(os.path.join(out_dir, fname), step["image"])

        # Build JSON-serialisable step (replace ndarray → base64)
        s = {k: v for k, v in step.items() if k not in ("image", "digits")}
        s["image_b64"] = _b64(step["image"]) if step.get("image") is not None else None

        if "digits" in step:
            serialised_digits = []
            for d in step["digits"]:
                # Save per-digit PNGs
                patch_fname = f"{i:02d}_digit{d['index']}_patch.png"
                zone_fname = f"{i:02d}_digit{d['index']}_zones.png"
                cv2.imwrite(os.path.join(out_dir, patch_fname), d["patch"])
                cv2.imwrite(os.path.join(out_dir, zone_fname), d["zone_vis"])

                sd = {k: v for k, v in d.items() if k not in ("patch", "zone_vis")}
                sd["patch_b64"] = _b64(d["patch"])
                sd["zone_vis_b64"] = _b64(d["zone_vis"])
                serialised_digits.append(sd)
            s["digits"] = serialised_digits

        json_steps.append(s)

    return out_dir, json_steps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sieben-Segment-Anzeige aus einem Bild dekodieren"
    )
    parser.add_argument("image_path", nargs="?", default="data/samples/pic1.png",
                        help="Pfad zum Eingabebild")
    parser.add_argument("-b", "--backend", default="python", help="Verarbeitungs-Backend")
    parser.add_argument("-o", "--output", default=None, help="Pfad für annotiertes Ausgabebild")
    parser.add_argument("-t", "--threshold", type=float, default=0.3,
                        help="Segment-Abtastschwelle (0.0-1.0, Standard: 0.3)")
    parser.add_argument("-n", "--no-save", action="store_true",
                        help="Annotiertes Bild nicht speichern")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Einzelne Ziffern ausgeben")
    parser.add_argument("-p", "--processed", action="store_true", default=False,
                        help="Jeden Verarbeitungsschritt als PNG + steps.json nach "
                             "data/processed/<name>/ speichern")
    args = parser.parse_args()

    if not os.path.isfile(args.image_path):
        print(f"Fehler: Datei nicht gefunden: {args.image_path}", file=sys.stderr)
        sys.exit(1)

    try:
        result = run_pipeline(
            args.image_path,
            backend=args.backend,
            segment_threshold=args.threshold,
            return_steps=args.processed,
        )
    except ValueError as e:
        print(f"Fehler: {e}", file=sys.stderr)
        sys.exit(3)

    if args.processed:
        number_str, annotated, steps = result
    else:
        number_str, annotated = result

    if not number_str.replace("?", ""):
        print("Warnung: Keine Ziffern erkannt.", file=sys.stderr)
        sys.exit(2)

    if args.verbose:
        print(f"Erkannte Zahl: {number_str}")
    else:
        print(number_str)

    if not args.no_save:
        out_path = args.output
        if out_path is None:
            base, ext = os.path.splitext(args.image_path)
            out_path = f"{base}_annotated{ext}"
        save_image(out_path, annotated)
        print(f"Annotiertes Bild gespeichert: {out_path}")

    if args.processed:
        out_dir, json_steps = _save_processed(steps, args.image_path)
        payload = {
            "image_path": args.image_path,
            "result": number_str,
            "params": {"segment_threshold": args.threshold, "backend": args.backend},
            "steps": json_steps,
        }
        json_path = os.path.join(out_dir, "steps.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Verarbeitungsschritte: {out_dir}/  ({len(steps)} PNGs + steps.json)")


if __name__ == "__main__":
    main()
