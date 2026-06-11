"""Sieben-Segment-Reader — Einstiegspunkt.

Verwendung:
    uv run python main.py <bildpfad> [Optionen]

Optionen:
    -b / --backend    python (Standard; später: rust)
    -o / --output     Pfad für das annotierte Ausgabebild
    -t / --threshold  Segment-Schwellwert 0.0-1.0 (Standard: 0.3)
    -n / --no-save    Annotiertes Bild nicht speichern
    -v / --verbose    Einzelne Ziffern ausgeben

Exit-Codes: 0=Erfolg, 1=Datei nicht gefunden, 2=keine Ziffern, 3=unbekanntes Backend
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "python"))

from segreader import run_pipeline
from segreader.io import save_image


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sieben-Segment-Anzeige aus einem Bild dekodieren"
    )
    parser.add_argument("image_path", nargs="?", default="data/samples/pic1.png",
                        help="Pfad zum Eingabebild (Standard: data/samples/pic1.png)")
    parser.add_argument("-b", "--backend", default="python", help="Verarbeitungs-Backend")
    parser.add_argument("-o", "--output", default=None, help="Pfad für annotiertes Ausgabebild")
    parser.add_argument("-t", "--threshold", type=float, default=0.3,
                        help="Segment-Abtastschwelle (0.0-1.0, Standard: 0.3)")
    parser.add_argument("-n", "--no-save", action="store_true",
                        help="Annotiertes Bild nicht speichern")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Einzelne Ziffern ausgeben")
    args = parser.parse_args()

    if not os.path.isfile(args.image_path):
        print(f"Fehler: Datei nicht gefunden: {args.image_path}", file=sys.stderr)
        sys.exit(1)

    try:
        number_str, annotated = run_pipeline(
            args.image_path,
            backend=args.backend,
            segment_threshold=args.threshold,
        )
    except ValueError as e:
        print(f"Fehler: {e}", file=sys.stderr)
        sys.exit(3)

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


if __name__ == "__main__":
    main()
