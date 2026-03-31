# Campus Crowd Monitor

> A real-time crowd density estimation system for campus canteens and libraries — built using classical Computer Vision techniques from the ground up.

---

## What this project does

Campus Crowd Monitor processes video footage from a fixed overhead or entry-facing camera to:

- **Detect people** using HOG (Histogram of Oriented Gradients) + SVM
- **Segment the room** into named zones (e.g. "Table Area", "Corridor") with colour-coded occupancy overlays
- **Generate a live density heatmap** using Gaussian kernel smoothing
- **Raise occupancy alerts** (Free / Moderate / Crowded) per zone
- **Visualise all intermediate CV steps** (edges, corners, pyramids, histograms) in an interactive Streamlit dashboard

All processing is **privacy-safe** — no faces are stored or identified. Only head/shoulder blobs are used.

---

## Syllabus coverage

| Module | Techniques used |
|--------|----------------|
| Unit 1 — Digital Image Formation | Resize, colour conversion, Gaussian blur (convolution), CLAHE histogram equalisation |
| Unit 3 — Feature Extraction | Canny, LOG, DOG edges · Harris corners · Hough lines · HOG descriptor · SIFT keypoints · Image pyramids (scale-space) |
| Unit 3 — Image Segmentation | Region growing · GrabCut (graph-cut) · Watershed · Mean-shift |
| Unit 4 — Pattern Analysis | HOG+SVM classifier · NMS · Background subtraction (MOG2) · Temporal density averaging |

---

## Project structure

```
campus-crowd-monitor/
├── data/
│   ├── sample_videos/        ← place your .mp4/.avi footage here
│   └── annotations/          ← manual counts for evaluation
├── src/
│   ├── preprocessing.py      ← Unit 1: Gaussian blur, CLAHE, MOG2, pyramids
│   ├── feature_extraction.py ← Unit 3: Canny, Harris, Hough, HOG, SIFT
│   ├── detector.py           ← Unit 3/4: HOG+SVM detection, NMS, zone counting
│   ├── segmentation.py       ← Unit 3: region growing, GrabCut, watershed, zones
│   ├── density_map.py        ← Unit 1/3: Gaussian KDE heatmap, alerts
│   └── dashboard.py          ← Streamlit web dashboard
├── models/
│   └── hog_svm.pkl           ← (auto-loaded from OpenCV; train your own here)
├── notebooks/
│   └── exploration.ipynb     ← step-by-step visualisation of every CV stage
├── report/
│   └── project_report.pdf
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/campus-crowd-monitor.git
cd campus-crowd-monitor
pip install -r requirements.txt
```

### 2. Add your footage

Place a `.mp4` or `.avi` file in `data/sample_videos/`. You can record 5–10 minutes of canteen/library footage on your phone (get permission from your institution first).

If you have no footage, the dashboard works with a webcam — see step 4.

### 3. Run the exploration notebook

```bash
jupyter notebook notebooks/exploration.ipynb
```

This walks through every preprocessing and feature-extraction step with visualisations — run it first to understand the pipeline and generate figures for your report.

### 4. Launch the dashboard

```bash
streamlit run src/dashboard.py
```

Open the URL shown in your terminal (usually `http://localhost:8501`).

- Upload a video **or** click "Start webcam"
- Toggle heatmap, bounding boxes, zone overlay, and edge debug view in the sidebar
- Adjust occupancy thresholds and detection sensitivity live

---

## How it works

### Pipeline (one frame)

```
Camera frame
    │
    ▼
resize_frame()          → 640×480, BGR → grayscale
    │
    ▼
gaussian_blur()         → 5×5 kernel, σ=1.0  (noise removal)
    │
    ▼
apply_clahe_color()     → CLAHE on LAB L-channel (contrast enhancement)
    │
    ├──→ BackgroundSubtractor.apply()  → foreground mask (MOG2)
    │
    ├──→ HOGPersonDetector.detect()   → bounding boxes + NMS
    │
    ├──→ make_density_map()           → Gaussian KDE heatmap
    │
    └──→ draw_zone_overlay()          → per-zone occupancy labels
```

### Occupancy zones

Zones are defined in `src/segmentation.py` as `DEFAULT_ZONES`:

```python
DEFAULT_ZONES = {
    "Zone A (tables 1-4)": (0,   0,   320, 240),
    "Zone B (tables 5-8)": (320, 0,   640, 240),
    "Zone C (corridor)":   (0,   240, 640, 480),
}
```

Edit these pixel coordinates to match your camera's field of view.

### Occupancy thresholds

| Label | Default condition |
|-------|------------------|
| Free | < 5 people |
| Moderate | 5–14 people |
| Crowded | ≥ 15 people |

Adjust via the sidebar sliders or directly in `src/density_map.py`.

---

## Evaluation

To evaluate detection accuracy:

1. Select 20–30 frames from your video
2. Manually count people in each frame (write to `data/annotations/counts.csv`)
3. Run the detector on those same frames
4. Compute Mean Absolute Error (MAE) — see the last cell of `exploration.ipynb`

---

## Customising zones

For a real deployment, calibrate zones to your camera:

1. Take a screenshot of your empty camera view
2. Open in any image editor and note pixel coordinates of table/zone boundaries
3. Update `DEFAULT_ZONES` in `src/segmentation.py`

---

## Tech stack

| Library | Version | Purpose |
|---------|---------|---------|
| OpenCV | ≥ 4.8 | All CV algorithms |
| NumPy | ≥ 1.24 | Array operations |
| Streamlit | ≥ 1.30 | Web dashboard |
| Matplotlib | ≥ 3.8 | Notebook visualisations |
| PyTorch | ≥ 2.1 | Optional deep learning backbone |

---

## License

MIT — free to use and adapt for academic purposes.
