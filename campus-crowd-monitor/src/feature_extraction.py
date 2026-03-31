"""
feature_extraction.py
---------------------
Unit 3 — Feature Extraction & Image Segmentation

Covers:
  - Canny edge detection
  - Harris corner detector
  - Hough line transform
  - HOG (Histogram of Oriented Gradients) descriptor
  - SIFT keypoints (bonus — for report depth)
  - Scale-space: LOG / DOG edges
"""

import cv2
import numpy as np
from typing import Tuple, List


# ---------------------------------------------------------------------------
# 1. Canny edge detector  (Syllabus: Edges — Canny)
# ---------------------------------------------------------------------------

def canny_edges(gray: np.ndarray,
                low_thresh: int = 50,
                high_thresh: int = 150,
                aperture: int = 3) -> np.ndarray:
    """
    Canny edge detector — four stages:
      1. Gaussian smoothing  (already done in preprocessing, but Canny re-applies)
      2. Sobel gradient magnitude & direction
      3. Non-maximum suppression (thin edges to 1-pixel width)
      4. Hysteresis thresholding (strong edges kept; weak edges kept only if
         connected to strong ones)

    low_thresh / high_thresh define the hysteresis band.
    Ratio ~1:3 is a good starting point for person silhouettes.
    """
    return cv2.Canny(gray, low_thresh, high_thresh, apertureSize=aperture)


# ---------------------------------------------------------------------------
# 2. LOG / DOG edges  (Syllabus: LOG, DOG)
# ---------------------------------------------------------------------------

def log_edges(gray: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """
    Laplacian of Gaussian (LOG) edge detector.
    Convolves the image with a Gaussian (to suppress noise), then applies
    the Laplacian (second derivative) to find zero-crossings = edges.
    """
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
    log     = cv2.Laplacian(blurred, cv2.CV_64F)
    # Normalise to uint8 for display
    log_abs = np.uint8(np.clip(np.abs(log), 0, 255))
    return log_abs


def dog_edges(gray: np.ndarray,
              sigma1: float = 1.0,
              sigma2: float = 2.0) -> np.ndarray:
    """
    Difference of Gaussians (DOG) — approximation of LOG.
    DOG(x,y) = G(σ1)*I - G(σ2)*I
    Computationally cheaper; also the basis of SIFT scale-space.
    """
    g1 = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma1)
    g2 = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma2)
    dog = cv2.subtract(g1, g2)
    return dog


# ---------------------------------------------------------------------------
# 3. Harris corner detector  (Syllabus: Corners — Harris)
# ---------------------------------------------------------------------------

def harris_corners(gray: np.ndarray,
                   block_size: int = 2,
                   ksize: int = 3,
                   k: float = 0.04,
                   threshold_ratio: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    Harris corner detector.

    For each pixel computes the structure tensor M = sum over window of:
        [Ix²   IxIy]
        [IxIy  Iy² ]
    Corner response R = det(M) - k * trace(M)²

    High R → corner.  Low R → flat.  Large negative R → edge.

    Returns:
        corners_img  — image with detected corners marked (for visualisation)
        corner_map   — float32 response map (R values)
    """
    gray_f = np.float32(gray)
    corner_map = cv2.cornerHarris(gray_f, block_size, ksize, k)
    corner_map = cv2.dilate(corner_map, None)   # dilate for visibility

    # Mark strong corners on a copy of the grayscale image
    corners_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    threshold   = threshold_ratio * corner_map.max()
    corners_img[corner_map > threshold] = [0, 0, 255]   # red dots

    return corners_img, corner_map


# ---------------------------------------------------------------------------
# 4. Hough line transform  (Syllabus: Line detectors — Hough Transform)
# ---------------------------------------------------------------------------

def hough_lines(edges: np.ndarray,
                rho: float = 1,
                theta: float = np.pi / 180,
                threshold: int = 80,
                min_line_length: int = 60,
                max_line_gap: int = 10) -> np.ndarray:
    """
    Probabilistic Hough Transform to detect straight lines (e.g. table edges,
    corridor walls) in an edge image.

    Returns an array of shape (N, 1, 4) where each row is [x1, y1, x2, y2].
    Returns empty array if no lines found.
    """
    lines = cv2.HoughLinesP(edges,
                            rho=rho,
                            theta=theta,
                            threshold=threshold,
                            minLineLength=min_line_length,
                            maxLineGap=max_line_gap)
    return lines if lines is not None else np.array([])


def draw_hough_lines(frame: np.ndarray, lines: np.ndarray,
                     colour: Tuple = (0, 255, 0),
                     thickness: int = 2) -> np.ndarray:
    """Draw detected Hough lines onto a copy of frame."""
    out = frame.copy()
    if lines.size > 0:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(out, (x1, y1), (x2, y2), colour, thickness)
    return out


# ---------------------------------------------------------------------------
# 5. HOG descriptor  (Syllabus: HOG)
# ---------------------------------------------------------------------------

class HOGFeatureExtractor:
    """
    Histogram of Oriented Gradients (HOG) feature extractor.

    HOG divides the image into small cells (e.g. 8×8 px), computes a
    histogram of gradient orientations in each cell, then groups cells into
    overlapping blocks and L2-normalises them to achieve illumination and
    shadow invariance.

    The resulting feature vector is what HOG+SVM uses for classification.

    Parameters follow Dalal & Triggs (2005) — the original person-detection
    paper that introduced HOG.
    """

    def __init__(self,
                 win_size: Tuple[int, int]   = (64, 128),
                 block_size: Tuple[int, int] = (16, 16),
                 block_stride: Tuple[int, int] = (8, 8),
                 cell_size: Tuple[int, int]  = (8, 8),
                 nbins: int = 9):
        self.win_size     = win_size
        self.block_size   = block_size
        self.block_stride = block_stride
        self.cell_size    = cell_size
        self.nbins        = nbins

        self._hog = cv2.HOGDescriptor(win_size, block_size,
                                      block_stride, cell_size, nbins)

    def compute(self, gray_patch: np.ndarray) -> np.ndarray:
        """
        Compute HOG descriptor for a single grayscale image patch.
        The patch is resized to win_size before computation.
        Returns a 1-D feature vector.
        """
        patch = cv2.resize(gray_patch, self.win_size)
        descriptor = self._hog.compute(patch)
        return descriptor.flatten()

    def detect_multiscale(self, gray: np.ndarray,
                          win_stride: Tuple[int, int] = (8, 8),
                          padding:    Tuple[int, int] = (8, 8),
                          scale:      float           = 1.05
                          ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run OpenCV's pre-trained HOG person detector at multiple scales.
        This is the scale-space detection that loops over image pyramid levels.

        Returns:
            boxes   — (N, 4) array of [x, y, w, h] bounding boxes
            weights — confidence score for each box
        """
        # Use OpenCV's built-in people detector (trained HOG+SVM model)
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        boxes, weights = self._hog.detectMultiScale(
            gray,
            winStride=win_stride,
            padding=padding,
            scale=scale,
        )
        return (np.array(boxes) if len(boxes) > 0 else np.array([]),
                np.array(weights) if len(weights) > 0 else np.array([]))

    def visualise_descriptor(self, gray_patch: np.ndarray,
                              cell_size: int = 8) -> np.ndarray:
        """
        Render the HOG gradient arrows for a patch — useful for the report.
        Each cell shows the dominant gradient direction.
        """
        patch   = cv2.resize(gray_patch, self.win_size)
        h, w    = patch.shape
        vis     = np.zeros_like(patch, dtype=np.float32)

        gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=1)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

        for y in range(0, h, cell_size):
            for x in range(0, w, cell_size):
                cell_mag = mag[y:y+cell_size, x:x+cell_size]
                cell_ang = ang[y:y+cell_size, x:x+cell_size]
                dominant = cell_ang.flat[np.argmax(cell_mag)]
                cx = x + cell_size // 2
                cy = y + cell_size // 2
                length = 4
                dx = int(length * np.cos(np.deg2rad(dominant)))
                dy = int(length * np.sin(np.deg2rad(dominant)))
                cv2.arrowedLine(vis, (cx - dx, cy - dy),
                                (cx + dx, cy + dy), 255, 1,
                                tipLength=0.4)
        return np.uint8(vis)


# ---------------------------------------------------------------------------
# 6. SIFT keypoints  (Syllabus: SIFT — bonus for report)
# ---------------------------------------------------------------------------

def extract_sift(gray: np.ndarray,
                 n_features: int = 500
                 ) -> Tuple[list, np.ndarray]:
    """
    Extract SIFT (Scale-Invariant Feature Transform) keypoints and descriptors.
    SIFT builds a DOG scale-space, finds local extrema, refines their position,
    assigns a dominant orientation, then computes a 128-dim descriptor.

    Note: SIFT is included to demonstrate syllabus coverage.
    For the crowd monitor the HOG detector is the primary workhorse.
    """
    sift = cv2.SIFT_create(nfeatures=n_features)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors


def draw_keypoints(gray: np.ndarray, keypoints: list) -> np.ndarray:
    """Render SIFT keypoints with scale circles on a colour image."""
    return cv2.drawKeypoints(
        gray, keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
