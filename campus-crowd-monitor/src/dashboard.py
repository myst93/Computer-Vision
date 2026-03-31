"""
dashboard.py
------------
Campus Crowd Monitor — Streamlit Dashboard

Run with:
    streamlit run src/dashboard.py

Features:
  - Upload a video or use webcam
  - Live per-zone occupancy display
  - Heatmap overlay toggle
  - HOG detection bounding boxes toggle
  - Zone-level status panel
  - Frame-by-frame analysis
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np
import streamlit as st
from pathlib import Path
import tempfile
import time

from preprocessing import preprocess_frame, BackgroundSubtractor
from feature_extraction import HOGFeatureExtractor
from detector import HOGPersonDetector, count_per_zone, occupancy_label
from segmentation import draw_zone_overlay, DEFAULT_ZONES
from density_map import (make_density_map, overlay_heatmap,
                         TemporalDensityAverager, OccupancyAlertSystem)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Campus Crowd Monitor",
    page_icon="",
    layout="wide",
)

st.title(" Campus Crowd Monitor")
st.caption("Computer Vision project · Feature Extraction & Image Segmentation")

# ---------------------------------------------------------------------------
# Sidebar — controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    source = st.radio("Video source", ["Upload video", "Webcam (live)"])

    st.subheader("Detection")
    scale         = st.slider("HOG scale step", 1.01, 1.15, 1.05, 0.01)
    hit_thresh    = st.slider("Hit threshold", -1.0, 2.0, 0.0, 0.1)
    nms_thresh    = st.slider("NMS IoU threshold", 0.1, 0.9, 0.45, 0.05)

    st.subheader("Occupancy thresholds")
    mod_thresh    = st.number_input("Moderate at (persons)", 1, 50, 5)
    crowd_thresh  = st.number_input("Crowded at (persons)", 2, 100, 15)

    st.subheader("Display options")
    show_boxes    = st.checkbox("Show bounding boxes", value=True)
    show_heatmap  = st.checkbox("Show density heatmap", value=True)
    show_zones    = st.checkbox("Show zone overlay", value=True)
    show_edges    = st.checkbox("Show Canny edges (debug)", value=False)
    heatmap_alpha = st.slider("Heatmap opacity", 0.1, 0.9, 0.5, 0.05)
    hmap_vmax     = st.slider("Heatmap vmax (clamp)", 0.01, 1.0, 0.3, 0.01)

    st.subheader("Zones")
    st.caption("Edit DEFAULT_ZONES in segmentation.py to customise regions.")

# ---------------------------------------------------------------------------
# Initialise components
# ---------------------------------------------------------------------------
@st.cache_resource
def get_detector(scale, hit_thresh, nms_thresh):
    return HOGPersonDetector(scale=scale,
                             hit_threshold=hit_thresh,
                             nms_thresh=nms_thresh)

@st.cache_resource
def get_bgsub():
    return BackgroundSubtractor(history=500, var_threshold=50)

detector   = get_detector(scale, hit_thresh, nms_thresh)
bgsub      = get_bgsub()
averager   = TemporalDensityAverager(window=8)
alert_sys  = OccupancyAlertSystem(mod_thresh, crowd_thresh)

# ---------------------------------------------------------------------------
# Helper: process one frame → return display image + metrics
# ---------------------------------------------------------------------------
def process_frame(frame: np.ndarray) -> dict:
    prep    = preprocess_frame(frame, bgsub=bgsub)
    gray    = prep["gray"]
    blurred = prep["blurred"]
    enhanced = prep["enhanced"]

    # Detect people
    boxes, weights = detector.detect(enhanced)

    # Zone counts
    zone_counts = count_per_zone(boxes, DEFAULT_ZONES)
    total_count = len(boxes)

    # Density map
    h, w = frame.shape[:2]
    density      = make_density_map((h, w), boxes, sigma=35)
    smooth_density = averager.update(density)

    # Alert evaluation
    evaluation = alert_sys.evaluate(zone_counts)

    # Build display frame
    display = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

    if show_heatmap and total_count > 0:
        bgr_disp = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
        bgr_disp = overlay_heatmap(bgr_disp, smooth_density,
                                   alpha=heatmap_alpha, vmax=hmap_vmax)
        display  = cv2.cvtColor(bgr_disp, cv2.COLOR_BGR2RGB)

    if show_zones:
        bgr_disp = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
        bgr_disp = draw_zone_overlay(bgr_disp, zone_counts,
                                     thresholds=(mod_thresh, crowd_thresh))
        display  = cv2.cvtColor(bgr_disp, cv2.COLOR_BGR2RGB)

    if show_boxes and len(boxes) > 0:
        bgr_disp = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
        bgr_disp = detector.draw_detections(bgr_disp, boxes, weights)
        display  = cv2.cvtColor(bgr_disp, cv2.COLOR_BGR2RGB)

    # Canny debug
    edges_img = None
    if show_edges:
        edges = cv2.Canny(blurred, 50, 150)
        edges_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    return {
        "display":     display,
        "edges":       edges_img,
        "total":       total_count,
        "zone_counts": zone_counts,
        "evaluation":  evaluation,
        "density":     smooth_density,
    }


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
col_video, col_metrics = st.columns([3, 1])

with col_metrics:
    st.subheader("Occupancy status")
    total_placeholder = st.metric("Total people", "—")
    zone_placeholders = {name: st.empty() for name in DEFAULT_ZONES}
    alert_placeholder = st.empty()
    st.divider()
    st.caption("Zones defined in segmentation.py")

with col_video:
    frame_placeholder = st.empty()
    if show_edges:
        edge_placeholder = st.empty()

# ---------------------------------------------------------------------------
# Video source branching
# ---------------------------------------------------------------------------

if source == "Upload video":
    uploaded = st.file_uploader("Upload a video file",
                                type=["mp4", "avi", "mov", "mkv"])

    if uploaded is not None:
        # Save to temp file so OpenCV can read it
        suffix = Path(uploaded.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_delay = 1.0 / fps

        stop_btn = st.button("Stop")
        bgsub.reset() if hasattr(bgsub, 'reset') else None

        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret:
                break

            frame_resized = cv2.resize(frame, (640, 480))
            result = process_frame(frame_resized)

            frame_placeholder.image(result["display"],
                                    channels="RGB", use_container_width=True)
            if show_edges and result["edges"] is not None:
                edge_placeholder.image(result["edges"],
                                       caption="Canny edges",
                                       channels="RGB",
                                       use_container_width=True)

            # Update metrics
            total_placeholder = col_metrics.metric(
                "Total people", result["total"])

            for zone_name, ph in zone_placeholders.items():
                info = result["evaluation"].get(zone_name, {})
                level = info.get("level", "Free")
                count = info.get("count", 0)
                emoji = {"Free": "", "Moderate": "", "Crowded": ""}[level]
                ph.markdown(
                    f"**{zone_name}**  \n{emoji} {level} — {count} person(s)")

            time.sleep(frame_delay)

        cap.release()
        st.success("Video processing complete.")

else:
    st.info("Click 'Start webcam' to begin live monitoring.")
    start = st.button("Start webcam")

    if start:
        cap = cv2.VideoCapture(0)
        stop_btn = st.button("Stop webcam")

        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret:
                st.error("Cannot read from webcam.")
                break

            frame_resized = cv2.resize(frame, (640, 480))
            result = process_frame(frame_resized)

            frame_placeholder.image(result["display"],
                                    channels="RGB", use_container_width=True)

            for zone_name, ph in zone_placeholders.items():
                info = result["evaluation"].get(zone_name, {})
                level = info.get("level", "Free")
                count = info.get("count", 0)
                emoji = {"Free": "", "Moderate": "", "Crowded": ""}[level]
                ph.markdown(
                    f"**{zone_name}**  \n{emoji} {level} — {count} person(s)")

        cap.release()
