"""
Sequential Labeling — eigene Python/numpy-Implementierung (Baseline für Rust-Vergleich).
Zwei-Pass-Algorithmus mit Union-Find. Kein cv2.connectedComponents / scipy.
"""
import numpy as np


class UnionFind:
    def __init__(self, size: int) -> None:
        self._parent = list(range(size))
        self._rank = [0] * size

    def find(self, x: int) -> int:
        # Pfadkompression
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        # Union by rank
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1


def label_components(binary: np.ndarray) -> np.ndarray:
    """Zwei-Pass Connected-Component-Labeling mit 4-Konnektivität."""
    H, W = binary.shape
    # obere Schranke: jedes Pixel könnte ein eigenes Label bekommen
    uf = UnionFind(H * W + 1)
    labels = np.zeros((H, W), dtype=np.int32)
    next_label = 1

    # Pass 1: provisorische Labels vergeben, Äquivalenzen erfassen
    for r in range(H):
        for c in range(W):
            if not binary[r, c]:
                continue
            north = labels[r - 1, c] if r > 0 else 0
            west = labels[r, c - 1] if c > 0 else 0

            if north == 0 and west == 0:
                labels[r, c] = next_label
                next_label += 1
            elif north == 0:
                labels[r, c] = west
            elif west == 0:
                labels[r, c] = north
            else:
                labels[r, c] = min(north, west)
                if north != west:
                    uf.union(north, west)

    # Pass 2: Äquivalenzen auflösen, kompakte sequentielle IDs vergeben
    root_to_id: dict = {}
    counter = 1
    for r in range(H):
        for c in range(W):
            lbl = labels[r, c]
            if lbl == 0:
                continue
            root = uf.find(lbl)
            if root not in root_to_id:
                root_to_id[root] = counter
                counter += 1
            labels[r, c] = root_to_id[root]

    return labels


def get_component_stats(labels: np.ndarray) -> list:
    """Gibt Statistiken (area, bbox, aspect_ratio) pro Komponente zurück, sortiert nach x1."""
    unique = [lbl for lbl in np.unique(labels) if lbl != 0]
    stats = []
    for lbl in unique:
        mask = labels == lbl
        rows, cols = np.where(mask)
        y1, y2 = int(rows.min()), int(rows.max()) + 1
        x1, x2 = int(cols.min()), int(cols.max()) + 1
        h, w = y2 - y1, x2 - x1
        stats.append({
            "label": int(lbl),
            "area": int(mask.sum()),
            "bbox": (y1, x1, y2, x2),
            "aspect_ratio": h / max(w, 1),
        })
    stats.sort(key=lambda s: s["bbox"][1])
    return stats
