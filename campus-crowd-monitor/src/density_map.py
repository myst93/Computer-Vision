"""
density_map.py
--------------
Unit 1 & 3 — Gaussian Kernel Density Estimation + Heatmap

Covers:
  - Kernel density estimation (Gaussian kernel smoothing)
  - Heatmap generation and colour mapping
  - Temporal smoothing across frames
  - Density-based alert system
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# 1. Point-spread Gaussian kernel density map
# ---------------------------------------------------------------------------

def make_density_map(frame_shape: Tuple[int, int],
                     boxes: np.ndarray,
                     sigma: float = 30.0) -> np.ndarray:
    """
    Generate a density map by placing a 2-D Gaussian kernel at the foot-point
    of each detected bounding box, then summing all contributions.

    This is the same "ground-truth density map" approach used in crowd-counting
    research (Zhang et al., MCNN, CVPR 2016).

    Args:
        frame_shape — (height, width) of the output density map
        boxes       — (N, 4) array of [x, y, w, h] detections
        sigma       — Gaussian spread in pixels; larger = smoother heatmap

    Returns:
        density (float32) — unnormalised density map, same size as frame
    """
    h, w    = frame_shape
    density = np.zeros((h, w), dtype=np.float32)

    for (bx, by, bw, bh) in boxes:
        # Foot-point = bottom-centre of bounding box
        px = int(bx + bw // 2)
        py = int(by + bh)
        px = np.clip(px, 0, w - 1)
        py = np.clip(py, 0, h - 1)
        density[py, px] += 1.0

    # Convolve point-mass map with Gaussian kernel
    # sigma controls the spatial spread of each person's influence
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
    density = cv2.GaussianBlur(density, (ksize, ksize), sigmaX=sigma)
    return density


def make_density_from_mask(fg_mask: np.ndarray,
                           sigma: float = 25.0) -> np.ndarray:
    """
    Alternative: build density map directly from a foreground mask.
    Each foreground pixel contributes weight 1.0 before Gaussian smoothing.
    Useful when no bounding boxes are available (e.g. very crowded scenes).
    """
    density = fg_mask.astype(np.float32) / 255.0
    ksize   = int(6 * sigma + 1) | 1   # ensure odd
    density = cv2.GaussianBlur(density, (ksize, ksize), sigmaX=sigma)
    return density


# ---------------------------------------------------------------------------
# 2. Heatmap visualisation
# ---------------------------------------------------------------------------

def density_to_heatmap(density: np.ndarray,
                       colormap: int = cv2.COLORMAP_JET,
                       vmax: Optional[float] = None) -> np.ndarray:
    """
    Convert a float density map to an 8-bit colour heatmap image.

    Args:
        density  — float32 density map
        colormap — OpenCV colormap (COLORMAP_JET, COLORMAP_HOT, etc.)
        vmax     — clip value for normalisation; None = use frame maximum

    Returns:
        heatmap — uint8 BGR colour image
    """
    if vmax is None:
        vmax = density.max() if density.max() > 0 else 1.0

    normalised = np.clip(density / vmax, 0.0, 1.0)
    gray_8bit  = np.uint8(normalised * 255)
    heatmap    = cv2.applyColorMap(gray_8bit, colormap)
    return heatmap


def overlay_heatmap(frame: np.ndarray,
                    density: np.ndarray,
                    alpha: float = 0.55,
                    colormap: int = cv2.COLORMAP_JET,
                    vmax: Optional[float] = None) -> np.ndarray:
    """
    Blend a density heatmap onto the original frame.

    Args:
        frame   — BGR camera frame
        density — float32 density map (same spatial size as frame)
        alpha   — heatmap opacity (0 = invisible, 1 = full heatmap)

    Returns:
        blended BGR image
    """
    heatmap = density_to_heatmap(density, colormap, vmax)
    heatmap_resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
    blended = cv2.addWeighted(heatmap_resized, alpha, frame, 1 - alpha, 0)
    return blended


# ---------------------------------------------------------------------------
# 3. Temporal smoothing
# ---------------------------------------------------------------------------

class TemporalDensityAverager:
    """
    Maintain a rolling average of density maps over the last N frames.

    Averaging reduces flicker caused by missed detections in individual frames
    (a person briefly occluded, slight detector failure, etc.).
    """

    def __init__(self, window: int = 10):
        self.window = window
        self._buffer: List[np.ndarray] = []

    def update(self, density: np.ndarray) -> np.ndarray:
        """Push a new density map and return the rolling average."""
        self._buffer.append(density.copy())
        if len(self._buffer) > self.window:
            self._buffer.pop(0)
        return np.mean(self._buffer, axis=0).astype(np.float32)

    def reset(self):
        self._buffer.clear()


# ---------------------------------------------------------------------------
# 4. Alert system
# ---------------------------------------------------------------------------

class OccupancyAlertSystem:
    """
    Threshold-based alert system that monitors crowd density and
    raises a visual/log alert when a zone exceeds capacity.

    Designed to be simple enough to explain in a project report
    while demonstrating practical utility.
    """

    LEVELS = {
        "Free":     {"colour": (0, 200, 80),  "threshold": 0},
        "Moderate": {"colour": (0, 165, 255), "threshold": 5},
        "Crowded":  {"colour": (0,  50, 220), "threshold": 15},
    }

    def __init__(self,
                 moderate_threshold: int = 5,
                 crowded_threshold: int  = 15):
        self.moderate_threshold = moderate_threshold
        self.crowded_threshold  = crowded_threshold
        self._history: List[dict] = []   # log of alert events

    def evaluate(self, zone_counts: dict) -> dict:
        """
        Evaluate occupancy for each zone.
        Returns {zone_name: {"count": int, "level": str, "colour": tuple}}
        """
        result = {}
        for zone, count in zone_counts.items():
            if count >= self.crowded_threshold:
                level = "Crowded"
            elif count >= self.moderate_threshold:
                level = "Moderate"
            else:
                level = "Free"
            colour = self.LEVELS[level]["colour"]
            result[zone] = {"count": count, "level": level, "colour": colour}
        return result

    def draw_alert_panel(self, frame: np.ndarray,
                         evaluation: dict,
                         panel_x: int = 10,
                         panel_y: int = 10) -> np.ndarray:
        """
        Draw a compact status panel in the corner of the frame listing
        each zone's count and colour-coded occupancy level.
        """
        out        = frame.copy()
        line_h     = 24
        panel_w    = 280
        panel_h    = len(evaluation) * line_h + 16
        # Semi-transparent dark background
        roi        = out[panel_y:panel_y + panel_h, panel_x:panel_x + panel_w]
        dark       = np.zeros_like(roi)
        cv2.addWeighted(dark, 0.55, roi, 0.45, 0, roi)
        out[panel_y:panel_y + panel_h, panel_x:panel_x + panel_w] = roi

        for i, (zone, info) in enumerate(evaluation.items()):
            y   = panel_y + 14 + i * line_h
            dot = (panel_x + 10, y - 4)
            cv2.circle(out, dot, 6, info["colour"], -1)
            text = f"{zone}: {info['count']}p  [{info['level']}]"
            cv2.putText(out, text, (panel_x + 22, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (230, 230, 230), 1, cv2.LINE_AA)

        return out
