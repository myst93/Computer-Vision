"""
detector.py
-----------
Unit 3 & 4 — Person Detection (HOG+SVM) + Pattern Analysis

Covers:
  - HOG+SVM person detection (classical CV approach)
  - Non-maximum suppression (NMS)
  - Optional: lightweight PyTorch MobileNet backbone
  - Count estimation per zone
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Non-maximum suppression  (NMS)
# ---------------------------------------------------------------------------

def non_max_suppression(boxes: np.ndarray,
                        weights: np.ndarray,
                        overlap_thresh: float = 0.45) -> np.ndarray:
    """
    Suppress overlapping bounding boxes, keeping only the highest-confidence
    detection when boxes overlap by more than overlap_thresh (IoU).

    HOG detectMultiScale returns many overlapping windows at different scales.
    NMS is the standard post-processing step to reduce duplicates.

    Algorithm (Felzenszwalb et al.):
      1. Sort boxes by score descending
      2. Greedily pick the top box, discard any box with IoU > threshold
      3. Repeat until no boxes remain
    """
    if len(boxes) == 0:
        return np.array([])

    x1 = boxes[:, 0].astype(float)
    y1 = boxes[:, 1].astype(float)
    x2 = (boxes[:, 0] + boxes[:, 2]).astype(float)
    y2 = (boxes[:, 1] + boxes[:, 3]).astype(float)
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = weights.argsort()[::-1]
    keep  = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds  = np.where(iou <= overlap_thresh)[0]
        order = order[inds + 1]

    return boxes[keep]


# ---------------------------------------------------------------------------
# HOG + OpenCV pre-trained SVM detector
# ---------------------------------------------------------------------------

class HOGPersonDetector:
    """
    Person detector based on HOG features + pre-trained linear SVM.
    Uses OpenCV's default people detector (trained on INRIA dataset).

    For the project report, describe this as:
      "We use Dalal & Triggs (2005) HOG descriptor with a linear SVM
       trained on the INRIA pedestrian dataset, accessed via OpenCV's
       HOGDescriptor_getDefaultPeopleDetector()."
    """

    def __init__(self,
                 win_stride: Tuple[int, int] = (8, 8),
                 padding:    Tuple[int, int] = (16, 16),
                 scale:      float           = 1.05,
                 hit_threshold: float        = 0.0,
                 nms_thresh: float           = 0.45):
        self.win_stride    = win_stride
        self.padding       = padding
        self.scale         = scale
        self.hit_threshold = hit_threshold
        self.nms_thresh    = nms_thresh

        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect people in `frame` (BGR or grayscale).
        Returns:
            boxes   — (N, 4) array [x, y, w, h] after NMS
            weights — confidence score per box
        """
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        raw_boxes, raw_weights = self._hog.detectMultiScale(
            gray,
            hitThreshold=self.hit_threshold,
            winStride=self.win_stride,
            padding=self.padding,
            scale=self.scale,
        )

        if len(raw_boxes) == 0:
            return np.array([]), np.array([])

        raw_weights = raw_weights.flatten()
        boxes = non_max_suppression(np.array(raw_boxes),
                                    raw_weights,
                                    self.nms_thresh)
        # Recompute weights for kept boxes (keep highest in cluster)
        kept_weights = np.array([
            raw_weights[np.where(
                (raw_boxes[:, 0] == b[0]) & (raw_boxes[:, 1] == b[1])
            )[0][0]] for b in boxes
        ]) if len(boxes) > 0 else np.array([])

        return boxes, kept_weights

    def draw_detections(self, frame: np.ndarray,
                        boxes: np.ndarray,
                        weights: Optional[np.ndarray] = None,
                        colour: Tuple = (0, 200, 0),
                        thickness: int = 2) -> np.ndarray:
        """Render bounding boxes (and optional scores) on frame."""
        out = frame.copy()
        for i, (x, y, w, h) in enumerate(boxes):
            cv2.rectangle(out, (x, y), (x + w, y + h), colour, thickness)
            if weights is not None and len(weights) > i:
                label = f"{weights[i]:.2f}"
                cv2.putText(out, label, (x, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            colour, 1, cv2.LINE_AA)
        # Person count overlay
        cv2.putText(out, f"Count: {len(boxes)}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        return out


# ---------------------------------------------------------------------------
# Foreground-contour based counter (fast, no ML needed)
# ---------------------------------------------------------------------------

def count_from_mask(fg_mask: np.ndarray,
                    min_area: int = 800,
                    max_area: int = 50_000) -> Tuple[int, List]:
    """
    Estimate person count from a cleaned foreground mask by finding contours.

    Each connected blob that falls within [min_area, max_area] pixels is
    treated as one person.  Tiny blobs = noise; huge blobs = multiple merged
    people (counted as 2 if area > 2×min threshold).

    Returns:
        count   — estimated number of people
        contours — raw contour list for visualisation
    """
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    valid = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        if area > max_area:
            # Heuristic: very large blob = multiple people merged
            count += max(1, int(area / (2 * min_area)))
        else:
            count += 1
        valid.append(cnt)
    return count, valid


# ---------------------------------------------------------------------------
# Zone-based occupancy
# ---------------------------------------------------------------------------

def count_per_zone(boxes: np.ndarray,
                   zones: dict) -> dict:
    """
    Assign each detected bounding box to a named zone.

    zones: dict of {zone_name: (x1, y1, x2, y2)}
    Returns dict of {zone_name: count}

    A person is assigned to the zone whose centre contains the bottom-centre
    of the bounding box (feet position — more reliable than head for
    overhead/angled cameras).
    """
    zone_counts = {name: 0 for name in zones}

    for (x, y, w, h) in boxes:
        foot_x = x + w // 2
        foot_y = y + h         # bottom of bounding box

        for name, (zx1, zy1, zx2, zy2) in zones.items():
            if zx1 <= foot_x <= zx2 and zy1 <= foot_y <= zy2:
                zone_counts[name] += 1
                break

    return zone_counts


def occupancy_label(count: int,
                    thresholds: Tuple[int, int] = (5, 15)) -> str:
    """
    Map a raw person count to a human-readable occupancy label.
    thresholds: (moderate_threshold, crowded_threshold)
    """
    if count < thresholds[0]:
        return "Free"
    elif count < thresholds[1]:
        return "Moderate"
    else:
        return "Crowded"
