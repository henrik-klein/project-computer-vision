"""
7-Segment-Dekodierung: Zonen abtasten -> 7-Bit-Muster -> Lookup-Tabelle -> Ziffer.
"""
import numpy as np

# Relative Zonen-Koordinaten (y_start, y_end, x_start, x_end) als Bruchteil der Box-Größe.
# Segment-Konvention: a=oben, b=oben-rechts, c=unten-rechts, d=unten, e=unten-links,
#                     f=oben-links, g=Mitte
#
# Wichtig: zwischen jedem horizontalen und dem angrenzenden vertikalen Segment gibt es
# einen expliziten Lückenbereich (~2 %), damit Nachbar-Segmente nicht in die falsche Zone
# bluten (besonders 'f'/'b' → 'g' und 'g' → 'e'/'c').
SEGMENT_ZONES: dict = {
    #        y_start  y_end   x_start  x_end
    "a": (0.02,    0.16,    0.20,    0.80),   # oben horizontal
    "b": (0.18,    0.42,    0.70,    0.98),   # oben-rechts vertikal
    "c": (0.58,    0.82,    0.70,    0.98),   # unten-rechts vertikal
    "d": (0.84,    0.98,    0.20,    0.80),   # unten horizontal
    "e": (0.58,    0.82,    0.02,    0.30),   # unten-links vertikal
    "f": (0.18,    0.42,    0.02,    0.30),   # oben-links vertikal
    "g": (0.44,    0.56,    0.20,    0.80),   # Mitte horizontal
}

# Lookup-Tabelle: aktive Segmente -> Ziffer
LOOKUP_TABLE: dict = {
    frozenset("abcdef"):  "0",
    frozenset("bc"):      "1",
    frozenset("abdeg"):   "2",
    frozenset("abcdg"):   "3",
    frozenset("bcfg"):    "4",
    frozenset("acdfg"):   "5",
    frozenset("acdefg"):  "6",
    frozenset("abc"):     "7",
    frozenset("abcdefg"): "8",
    frozenset("abcdfg"):  "9",
}


def sample_segments(patch: np.ndarray, threshold: float = 0.3) -> frozenset:
    """Bestimmt aktive Segmente durch Mittelwert-Abtastung der 7 Zonen."""
    H, W = patch.shape
    active = set()
    for seg, (ys, ye, xs, xe) in SEGMENT_ZONES.items():
        r1, r2 = max(int(ys * H), 0), max(int(ye * H), 1)
        c1, c2 = max(int(xs * W), 0), max(int(xe * W), 1)
        zone = patch[r1:r2, c1:c2]
        if zone.size > 0 and zone.mean() >= threshold:
            active.add(seg)
    return frozenset(active)


def decode_digit(patch: np.ndarray, threshold: float = 0.3) -> str:
    """Dekodiert eine einzelne Ziffer aus ihrem binären Patch.

    Sonderfall "1": sehr schmale Box (height/width > 3) wird direkt klassifiziert,
    da das Zonen-Sampling auf einer vollen Ziffernzelle aufgebaut ist und bei
    engen Boxen falsche Mittelwerte liefert.
    """
    H, W = patch.shape
    if W > 0 and H / W > 3:
        return "1"
    active = sample_segments(patch, threshold)
    return LOOKUP_TABLE.get(active, "?")


def decode_number(
    binary: np.ndarray,
    digit_boxes: list,
    threshold: float = 0.3,
) -> tuple:
    """Dekodiert alle Ziffern-Boxen und gibt (Gesamtzahl, Liste pro Ziffer) zurück."""
    digits = []
    for y1, x1, y2, x2 in digit_boxes:
        patch = binary[y1:y2, x1:x2]
        digits.append(decode_digit(patch, threshold))
    return "".join(digits), digits
