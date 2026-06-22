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
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "python"))

from segreader import run_pipeline
from segreader.io import save_image
from segreader.steps_io import save_processed_steps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sieben-Segment-Anzeige aus einem Bild dekodieren"
    )
    parser.add_argument("image_path", nargs="?", default="data/samples/pic1.png",
                        help="Pfad zum Eingabebild")
    parser.add_argument("-b", "--backend", default="python", help="Verarbeitungs-Backend")
    parser.add_argument("-o", "--output", default=None, help="Pfad für annotiertes Ausgabebild")
    parser.add_argument("-t", "--threshold", type=float, default=0.40,
                        help="Segment-Abtastschwelle (0.0-1.0, Standard: 0.40)")
    parser.add_argument("-n", "--no-save", action="store_true",
                        help="Annotiertes Bild nicht speichern")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Einzelne Ziffern ausgeben")
    parser.add_argument("-p", "--processed", action="store_true", default=False,
                        help="Jeden Verarbeitungsschritt als PNG + steps.json nach "
                             "data/processed/<name>/ speichern")
    parser.add_argument("--no-localize", action="store_true",
                        help="Lokalisierung überspringen — ganzes Bild verwenden")
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
            localize=not args.no_localize,
        )
    except ValueError as e:
        print(f"Fehler: {e}", file=sys.stderr)
        sys.exit(3)

    if args.processed:
        number_str, annotated, steps = result
    else:
        number_str, annotated = result
        
    if args.processed:
        json_path = save_processed_steps(
            steps, args.image_path, number_str,
            {"segment_threshold": args.threshold, "backend": args.backend},
        )
        print(f"Verarbeitungsschritte: {os.path.dirname(json_path)}/  ({len(steps)} PNGs + steps.json)")

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
