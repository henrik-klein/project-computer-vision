from __future__ import annotations

import cv2
import numpy as np

from .decode import SEGMENT_ZONES
from .preprocess import _compute_otsu_threshold


class visualisation:
    """Static helpers that convert pipeline intermediates into displayable BGR images."""

    COMP_COLORS: list[tuple] = [
        (255, 100, 100), (100, 255, 100), (100, 100, 255),
        (255, 255,  80), (255,  80, 255), ( 80, 255, 255),
        (255, 160,  80), (160, 255,  80), ( 80, 160, 255),
        (200, 200, 100), (100, 200, 200), (200, 100, 200),
    ]

    @staticmethod
    def gray(gray: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def binary(binary: np.ndarray) -> np.ndarray:
        return cv2.cvtColor((binary * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    @staticmethod
    def components(labels: np.ndarray, stats: list, kept_set: set) -> np.ndarray:
        vis = np.zeros((*labels.shape, 3), dtype=np.uint8)
        for i, s in enumerate(stats):
            color = (visualisation.COMP_COLORS[i % len(visualisation.COMP_COLORS)]
                     if s["label"] in kept_set else (60, 60, 60))
            vis[labels == s["label"]] = color
        return vis

    @staticmethod
    def filter_result(labels: np.ndarray, stats: list, kept_set: set) -> np.ndarray:
        vis = np.zeros((*labels.shape, 3), dtype=np.uint8)
        for s in stats:
            if s["label"] in kept_set:
                color = (0, 200, 60)       # green  — kept
            elif s.get("rejection_reason", "").startswith("background"):
                color = (0, 140, 255)      # orange — background outlier
            else:
                color = (60, 0, 200)       # purple — noise / colon
            vis[labels == s["label"]] = color
        return vis

    @staticmethod
    def projection(proj: np.ndarray, thresh_px: float, digit_boxes: list) -> np.ndarray:
        W = max(len(proj), 1)
        chart = np.full((180, W, 3), 18, dtype=np.uint8)
        max_val = float(proj.max()) if proj.max() > 0 else 1.0
        for x, val in enumerate(proj):
            bar_h = int(val / max_val * 176)
            if bar_h > 0:
                cv2.line(chart, (x, 178), (x, 178 - bar_h), (80, 200, 80), 1)
        thresh_y = max(1, min(177, 178 - int(thresh_px / max_val * 176)))
        cv2.line(chart, (0, thresh_y), (W - 1, thresh_y), (80, 80, 255), 1)
        cv2.putText(chart, f"thr={thresh_px:.1f}px", (4, thresh_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 255), 1)
        for (_, x1, _, x2) in digit_boxes:
            cv2.rectangle(chart, (x1, 1), (x2, 177), (200, 180, 0), 1)
        return chart

    @staticmethod
    def zone_overlay(patch: np.ndarray, zones_data: dict, scale: int = 7) -> np.ndarray:
        H, W = patch.shape
        if H == 0 or W == 0:
            return np.zeros((60, 40, 3), dtype=np.uint8)
        vis = cv2.cvtColor(
            cv2.resize((patch * 55).astype(np.uint8), (W * scale, H * scale),
                       interpolation=cv2.INTER_NEAREST),
            cv2.COLOR_GRAY2BGR,
        )
        for seg, info in zones_data.items():
            ys, ye, xs, xe = SEGMENT_ZONES[seg]
            r1 = int(ys * H * scale); r2 = int(ye * H * scale)
            c1 = int(xs * W * scale); c2 = int(xe * W * scale)
            color = (0, 220, 60) if info["active"] else (60, 30, 180)
            overlay = vis.copy()
            cv2.rectangle(overlay, (c1, r1), (c2, r2), color, -1)
            cv2.addWeighted(overlay, 0.38, vis, 0.62, 0, vis)
            cv2.rectangle(vis, (c1, r1), (c2, r2), color, 1)
            mx, my = (c1 + c2) // 2, (r1 + r2) // 2
            cv2.putText(vis, seg, (mx - 3, my + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1)
        return vis

    @staticmethod
    def boxes(binary: np.ndarray, digit_boxes: list, digits: list) -> np.ndarray:
        out = visualisation.binary(binary)
        for i, ((y1, x1, y2, x2), d) in enumerate(zip(digit_boxes, digits)):
            color = (0, 230, 0) if d != "?" else (30, 30, 230)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, f"#{i}:{d}", (x1, max(y1 - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return out

    @staticmethod
    def localize(debug_img: np.ndarray, method_used: str, found: bool) -> np.ndarray:
        """Overlay a status banner on the debug image returned by find_display.

        The debug_img already contains the bounding-box annotation drawn by
        localize.py. This method adds a top stripe that documents which of the
        three auto strategies succeeded (color → brightness → contour) and
        whether a display region was actually found.
        """
        out = debug_img.copy()
        h, w = out.shape[:2]
        cv2.rectangle(out, (0, 0), (w, 30), (0, 0, 0), -1)
        color = (0, 200, 60) if found else (60, 100, 220)
        label = (f"auto → {method_used}  |  display found ✓"
                 if found else
                 f"auto → {method_used}  |  not found — full image used")
        cv2.putText(out, label, (8, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
        return out

    @staticmethod
    def otsu_threshold(gray: np.ndarray) -> int:
        return _compute_otsu_threshold(gray)

    @staticmethod
    def rejection_reason(s: dict) -> str:
        return s.get("rejection_reason", "unknown")
