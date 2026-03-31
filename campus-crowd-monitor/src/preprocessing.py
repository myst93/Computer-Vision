"""
preprocessing.py
----------------
Unit 1 — Digital Image Formation & Low-Level Processing

Covers:
  - Frame resizing & colour-space conversion
  - Gaussian blur (convolution-based noise removal)
  - CLAHE histogram equalisation (contrast enhancement)
  - MOG2 background subtraction (foreground mask)
  - Gaussian image pyramid (scale-space representation)
"""

import cv2
import numpy as np
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Step 1 — Resize & colour conversion
# ---------------------------------------------------------------------------

def resize_frame(frame: np.ndarray,
                 target_size: Tuple[int, int] = (640, 480)) -> np.ndarray:
    """
    Resize a BGR frame to target_size using bilinear interpolation.
    Bilinear interpolation (cv2.INTER_LINEAR) computes each output pixel
    as a weighted average of the four nearest input pixels — a good balance
    between speed and quality for video frames.
    """
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    """Convert BGR frame to single-channel grayscale."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Flip channel order for display in matplotlib / Streamlit."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Step 2 — Gaussian blur  (Syllabus: convolution & filtering)
# ---------------------------------------------------------------------------

def gaussian_blur(gray: np.ndarray,
                  kernel_size: Tuple[int, int] = (5, 5),
                  sigma: float = 1.0) -> np.ndarray:
    """
    Convolve the image with a 2-D Gaussian kernel to suppress high-frequency
    noise before edge / feature detection.

    The Gaussian kernel G(x,y) = exp(-(x²+y²)/(2σ²)) / (2πσ²).
    Larger σ → stronger smoothing but more detail loss.
    For canteen footage with moderate compression noise, σ=1.0 with a 5×5
    kernel is a safe default.
    """
    return cv2.GaussianBlur(gray, kernel_size, sigmaX=sigma, sigmaY=sigma)


# ---------------------------------------------------------------------------
# Step 3 — CLAHE histogram equalisation  (Syllabus: histogram processing)
# ---------------------------------------------------------------------------

def apply_clahe(gray: np.ndarray,
                clip_limit: float = 2.0,
                tile_grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Contrast-Limited Adaptive Histogram Equalisation (CLAHE).

    Unlike global histogram equalisation, CLAHE operates on small tiles so
    it boosts contrast locally without over-amplifying noise in already-bright
    regions (controlled by clip_limit).

    For canteen footage: uneven lighting from windows is the main problem.
    clipLimit=2.0 and tileGridSize=(8,8) are empirically good starting points.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(gray)


def apply_clahe_color(bgr: np.ndarray,
                      clip_limit: float = 2.0,
                      tile_grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE to the L channel of a LAB image (preserves colour balance).
    Better than applying CLAHE directly to each BGR channel.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq = clahe.apply(l_ch)
    lab_eq = cv2.merge([l_eq, a_ch, b_ch])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Step 4 — Background subtraction / MOG2  (Syllabus: Unit 4 motion analysis)
# ---------------------------------------------------------------------------

class BackgroundSubtractor:
    """
    Wrapper around OpenCV's MOG2 (Mixture of Gaussians v2) background model.

    MOG2 models each pixel's intensity history as a mixture of K Gaussians
    and classifies pixels that fall outside all Gaussians as foreground.
    The model adapts over time (controlled by `history`) so gradual lighting
    changes do not trigger false detections.

    Typical usage:
        bgsub = BackgroundSubtractor()
        for frame in video:
            fg_mask = bgsub.apply(frame)
            # fg_mask: 255 = foreground (person), 0 = background
    """

    def __init__(self,
                 history: int = 500,
                 var_threshold: float = 50.0,
                 detect_shadows: bool = True):
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )

    def apply(self, frame: np.ndarray,
              learning_rate: float = -1) -> np.ndarray:
        """
        Return binary foreground mask for `frame`.
        learning_rate=-1 lets OpenCV choose automatically (typically ~1/history).
        Shadows are marked as 127; set to 0 with a threshold to remove them.
        """
        mask = self._mog2.apply(frame, learningRate=learning_rate)
        # Remove shadow pixels (127) — keep only definite foreground (255)
        _, binary = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        return binary

    def clean_mask(self, mask: np.ndarray,
                   morph_kernel: int = 5) -> np.ndarray:
        """
        Morphological opening (erode then dilate) to remove small noise blobs,
        followed by closing to fill holes inside person silhouettes.
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        opened  = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        closed  = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        return closed


# ---------------------------------------------------------------------------
# Step 5 — Gaussian image pyramid  (Syllabus: Unit 3 scale-space analysis)
# ---------------------------------------------------------------------------

def build_gaussian_pyramid(image: np.ndarray,
                            levels: int = 4) -> list:
    """
    Build a Gaussian image pyramid with `levels` levels.

    At each level the image is blurred (to prevent aliasing) then downsampled
    by 2× using cv2.pyrDown().  The result is a list of images:
        pyramid[0] = original resolution
        pyramid[1] = half resolution
        pyramid[2] = quarter resolution
        ...

    This is the basis for multi-scale person detection: we run the HOG detector
    at each pyramid level so that distant (small) people are detectable at
    lower levels where they appear at 'standard' size.
    """
    pyramid = [image]
    for _ in range(levels - 1):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid


def build_laplacian_pyramid(gaussian_pyramid: list) -> list:
    """
    Derive the Laplacian pyramid from a Gaussian pyramid.

    Each Laplacian level L[i] = G[i] - upsample(G[i+1])
    and captures the band-pass detail lost between adjacent Gaussian levels.
    Useful for visualising which scale contains the most person-like structure.
    """
    laplacian = []
    for i in range(len(gaussian_pyramid) - 1):
        size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
        expanded = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=size)
        lap = cv2.subtract(gaussian_pyramid[i], expanded)
        laplacian.append(lap)
    laplacian.append(gaussian_pyramid[-1])   # top level kept as-is
    return laplacian


# ---------------------------------------------------------------------------
# Full pipeline  — chain all steps for one frame
# ---------------------------------------------------------------------------

def preprocess_frame(frame: np.ndarray,
                     bgsub: Optional[BackgroundSubtractor] = None,
                     target_size: Tuple[int, int] = (640, 480),
                     pyramid_levels: int = 3
                     ) -> dict:
    """
    Run the complete preprocessing pipeline on a single BGR frame.

    Returns a dict with all intermediate and final artefacts so the caller
    can choose what to pass downstream.
    """
    # 1. Resize
    resized = resize_frame(frame, target_size)

    # 2. Colour-enhance (CLAHE on colour frame)
    enhanced = apply_clahe_color(resized)

    # 3. Grayscale
    gray = to_grayscale(enhanced)

    # 4. Gaussian blur
    blurred = gaussian_blur(gray)

    # 5. Background mask (optional — skip if no bgsub passed)
    fg_mask = None
    if bgsub is not None:
        raw_mask = bgsub.apply(resized)
        fg_mask  = bgsub.clean_mask(raw_mask)

    # 6. Scale-space pyramid
    pyramid = build_gaussian_pyramid(blurred, levels=pyramid_levels)

    return {
        "resized":   resized,
        "enhanced":  enhanced,
        "gray":      gray,
        "blurred":   blurred,
        "fg_mask":   fg_mask,
        "pyramid":   pyramid,
    }
