"""
segmentation.py
---------------
Unit 3 — Image Segmentation

Covers:
  - Region growing
  - Graph-Cut (GrabCut) foreground segmentation
  - Watershed segmentation
  - Mean-shift clustering
  - Zone polygon overlay (occupied vs free)
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional


# ---------------------------------------------------------------------------
# 1. Region Growing  (Syllabus: Region Growing)
# ---------------------------------------------------------------------------

def region_growing(gray: np.ndarray,
                   seed: Tuple[int, int],
                   threshold: int = 15) -> np.ndarray:
    """
    Simple 4-connected region growing from a seed pixel.

    Starting from `seed`, we add any neighbouring pixel whose intensity
    differs from the seed by less than `threshold`.  The result is a binary
    mask of the grown region.

    Complexity: O(pixels) with a queue-based BFS.
    For crowd monitoring: seed can be placed in a 'free seat' to measure
    the unoccupied floor area.
    """
    h, w    = gray.shape
    visited = np.zeros((h, w), dtype=bool)
    mask    = np.zeros((h, w), dtype=np.uint8)

    seed_val = int(gray[seed[1], seed[0]])
    queue    = [seed]
    visited[seed[1], seed[0]] = True

    while queue:
        cx, cy = queue.pop(0)
        mask[cy, cx] = 255

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                visited[ny, nx] = True
                if abs(int(gray[ny, nx]) - seed_val) < threshold:
                    queue.append((nx, ny))

    return mask


# ---------------------------------------------------------------------------
# 2. GrabCut (Graph-Cut) foreground segmentation  (Syllabus: Graph-Cut)
# ---------------------------------------------------------------------------

def grabcut_segment(frame: np.ndarray,
                    rect: Tuple[int, int, int, int],
                    n_iter: int = 5) -> np.ndarray:
    """
    GrabCut algorithm — iterative graph-cut based segmentation.

    `rect` defines a bounding rectangle (x, y, w, h) that definitely contains
    the foreground object.  GrabCut models foreground and background pixel
    distributions as Gaussian Mixture Models (GMMs) and alternates between:
      1. Classifying pixels using min-cut on a graph
      2. Re-estimating the GMM parameters

    Returns a binary mask (255 = probable/definite foreground).

    Use case for project: segment a single detected person blob from its
    local background for more accurate bounding-box refinement.
    """
    mask  = np.zeros(frame.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(frame, mask, rect, bgdModel, fgdModel,
                n_iter, cv2.GC_INIT_WITH_RECT)

    # Pixels marked GC_FGD (1) or GC_PR_FGD (3) = foreground
    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
                       255, 0).astype(np.uint8)
    return fg_mask


# ---------------------------------------------------------------------------
# 3. Watershed segmentation  (Syllabus: edge-based segmentation)
# ---------------------------------------------------------------------------

def watershed_segment(frame: np.ndarray,
                      fg_mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Marker-based Watershed segmentation to separate touching people blobs.

    Steps:
      1. Morphological operations to find sure background and sure foreground
      2. Unknown region = sure_bg - sure_fg
      3. Label connected components in sure_fg as markers
      4. Run watershed — boundaries are drawn where competing markers meet

    Returns:
        markers  — labelled image (each unique label = one person)
        n_labels — number of detected segments (= people count estimate)
    """
    # Sure background: dilate the fg_mask
    kernel    = np.ones((3, 3), np.uint8)
    sure_bg   = cv2.dilate(fg_mask, kernel, iterations=3)

    # Sure foreground: distance transform + threshold
    dist_transform = cv2.distanceTransform(fg_mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform,
                               0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Connected component markers
    n_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1           # background = 1, not 0
    markers[unknown == 255] = 0     # unknown region = 0

    # Run watershed
    markers = cv2.watershed(frame, markers)

    return markers, n_labels - 1    # -1 to exclude background label


def draw_watershed_overlay(frame: np.ndarray,
                           markers: np.ndarray) -> np.ndarray:
    """Colour each watershed segment and draw boundaries in red."""
    out = frame.copy()
    # Boundaries are marked as -1 by watershed
    out[markers == -1] = [0, 0, 255]

    # Colour each label region (skip label 1 = background)
    n = markers.max()
    colours = np.random.randint(50, 200, (n + 2, 3), dtype=np.uint8)
    for label in range(2, n + 1):
        out[markers == label] = (
            out[markers == label].astype(np.float32) * 0.5 +
            colours[label].astype(np.float32) * 0.5
        ).astype(np.uint8)

    return out


# ---------------------------------------------------------------------------
# 4. Mean-Shift clustering  (Syllabus: Mean-Shift)
# ---------------------------------------------------------------------------

def meanshift_segment(frame: np.ndarray,
                      spatial_radius: int = 21,
                      color_radius: int = 51,
                      min_density: int = 50) -> np.ndarray:
    """
    Mean-Shift image segmentation (pyrMeanShiftFiltering).

    Groups pixels into spatially and chromatically similar clusters without
    specifying the number of clusters in advance — useful for identifying
    distinct crowd zones without knowing how many zones exist.

    Returns the colour-quantised segmented image.
    """
    return cv2.pyrMeanShiftFiltering(frame,
                                     sp=spatial_radius,
                                     sr=color_radius,
                                     minSize=min_density)


# ---------------------------------------------------------------------------
# 5. Zone overlay  (custom — for dashboard)
# ---------------------------------------------------------------------------

DEFAULT_ZONES: Dict[str, Tuple[int, int, int, int]] = {
    "Zone A (tables 1-4)": (0,   0,   320, 240),
    "Zone B (tables 5-8)": (320, 0,   640, 240),
    "Zone C (corridor)":   (0,   240, 640, 480),
}

OCCUPANCY_COLOURS = {
    "Free":     (0,   200, 80),    # green
    "Moderate": (0,   165, 255),   # orange
    "Crowded":  (0,   0,   220),   # red
}


def draw_zone_overlay(frame: np.ndarray,
                      zone_counts: Dict[str, int],
                      zones: Dict[str, Tuple[int, int, int, int]] = None,
                      thresholds: Tuple[int, int] = (5, 15),
                      alpha: float = 0.35) -> np.ndarray:
    """
    Draw semi-transparent coloured overlays for each zone based on occupancy.

    zone_counts: {zone_name: person_count}
    zones:       {zone_name: (x1, y1, x2, y2)}  — defaults to DEFAULT_ZONES
    alpha:       transparency of the colour overlay (0=invisible, 1=opaque)
    """
    from detector import occupancy_label  # local import to avoid circular

    if zones is None:
        zones = DEFAULT_ZONES

    overlay = frame.copy()
    out     = frame.copy()

    for name, (x1, y1, x2, y2) in zones.items():
        count  = zone_counts.get(name, 0)
        label  = occupancy_label(count, thresholds)
        colour = OCCUPANCY_COLOURS[label]

        cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, -1)  # filled
        cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)

        # Reset overlay for next zone
        overlay = out.copy()

        # Zone label text
        text = f"{name}  [{label}]  {count}p"
        cv2.putText(out, text, (x1 + 8, y1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)   # border

    return out
