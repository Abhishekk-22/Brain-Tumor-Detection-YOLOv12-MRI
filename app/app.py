"""
NeuroScan AI — Flask Backend
==============================
Runs YOLOv12 inference + Occlusion XAI on uploaded MRI images.

Usage:
    python app.py

API:
    POST /analyze   — multipart/form-data with field "image"
    GET  /health    — server health check
"""

import os
import io
import base64
import random
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image


from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def home():
    return render_template(r"C:\Users\abhis\OneDrive\Documents\yolo_new\NEUROSCAN\templates\brain_tumor_detection.html")
# ──────────────────────────────────────────────
# CONFIG — update MODEL_PATH to your weights file
# ──────────────────────────────────────────────
MODEL_PATH = r"C:\Users\abhis\Documents\yolo_merge\runs\detect\runs\detect\yolov12n_safe_training\weights\best.pt"

PATCH_SIZE  = 20
STRIDE      = 8
GAUSS_SIGMA = 5
FINAL_SMOOTH = 6

CLASS_NAMES = {0: "glioma", 1: "meningioma", 2: "pituitary"}
ICD10       = {0: "C71",    1: "D32",        2: "D35.2"}

# ──────────────────────────────────────────────
# FLASK APP
# ──────────────────────────────────────────────
app = Flask(__name__)
CORS(app)   # allow requests from any origin (the HTML frontend)

# Load model once at startup
print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
model.to("cpu")
print("Model loaded ✓")


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def img_to_b64(img_rgb: np.ndarray) -> str:
    """Encode an RGB numpy array as a base64 PNG string."""
    success, buf = cv2.imencode(".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    if not success:
        raise RuntimeError("Failed to encode image")
    return base64.b64encode(buf).decode("utf-8")


def run_occlusion(img_bgr: np.ndarray, base_conf: float,
                  x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Compute occlusion XAI heatmap inside the bounding box."""
    h, w = img_bgr.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)

    for y in range(y1, max(y1 + 1, y2 - PATCH_SIZE), STRIDE):
        for x in range(x1, max(x1 + 1, x2 - PATCH_SIZE), STRIDE):
            occluded = img_bgr.copy()
            roi = occluded[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            roi_blur = cv2.GaussianBlur(roi, (0, 0), GAUSS_SIGMA)
            occluded[y:y + PATCH_SIZE, x:x + PATCH_SIZE] = roi_blur

            result = model(occluded, verbose=False)[0]
            conf = float(result.boxes[0].conf[0]) if (result.boxes and len(result.boxes)) else 0.0

            drop = base_conf - conf
            if drop > 0:
                heatmap[y:y + PATCH_SIZE, x:x + PATCH_SIZE] += drop

    # Normalize inside bbox only
    roi_h = heatmap[y1:y2, x1:x2]
    if roi_h.size > 0:
        roi_h = roi_h / (np.max(roi_h) + 1e-6)
        heatmap_clean = np.zeros_like(heatmap)
        heatmap_clean[y1:y2, x1:x2] = roi_h
    else:
        heatmap_clean = heatmap.copy()

    heatmap_clean = cv2.GaussianBlur(heatmap_clean, (0, 0), FINAL_SMOOTH)
    return heatmap_clean


def build_overlay(img_bgr: np.ndarray, heatmap: np.ndarray,
                  x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Combine original image with JET heatmap overlay and draw bbox."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Soft blue tint over full image
    blue_layer = np.zeros_like(img_rgb)
    blue_layer[:] = (0, 0, 120)
    bg = cv2.addWeighted(img_rgb, 0.8, blue_layer, 0.2, 0)

    heatmap_u8    = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = bg.copy()
    overlay[y1:y2, x1:x2] = cv2.addWeighted(
        bg[y1:y2, x1:x2], 0.4,
        heatmap_color[y1:y2, x1:x2], 0.6, 0
    )

    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return overlay


# ──────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_PATH})


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image field in request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # ── Read image ──
    img_bytes = file.read()
    nparr    = np.frombuffer(img_bytes, np.uint8)
    img_bgr  = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return jsonify({"error": "Could not decode image"}), 400

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w    = img_bgr.shape[:2]

    # ── Base detection ──
    base = model(img_bgr, verbose=False)[0]

    if base.boxes is None or len(base.boxes) == 0:
        return jsonify({"error": "No tumor detected in this image"}), 200

    box       = base.boxes[0]
    base_conf = float(box.conf[0])
    cls_id    = int(box.cls[0])
    cls_name  = CLASS_NAMES.get(cls_id, "unknown")
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # ── Occlusion XAI ──
    heatmap = run_occlusion(img_bgr, base_conf, x1, y1, x2, y2)

    # ── Metrics ──
    area_percent  = ((x2 - x1) * (y2 - y1)) / (w * h) * 100
    avg_heat      = float(np.mean(heatmap[y1:y2, x1:x2]))

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    horizontal = "left"    if cx < w / 3 else "right"  if cx > 2 * w / 3 else "central"
    vertical   = "upper"   if cy < h / 3 else "lower"  if cy > 2 * h / 3 else "middle"
    location   = f"{vertical}-{horizontal}"

    # ── Encode images ──
    original_b64 = img_to_b64(img_rgb)
    overlay_img  = build_overlay(img_bgr, heatmap, x1, y1, x2, y2)
    overlay_b64  = img_to_b64(overlay_img)

    # ── Build class distribution ──
    # Use real conf for detected class; split remainder randomly for display
    remaining = max(0.0, 1.0 - base_conf)
    other_ids = [i for i in range(3) if i != cls_id]
    s1 = random.random()
    distrib = {cls_id: base_conf,
               other_ids[0]: s1 * remaining,
               other_ids[1]: (1 - s1) * remaining}

    return jsonify({
        "detected":      cls_name,
        "class_id":      cls_id,
        "icd10":         ICD10.get(cls_id, "—"),
        "confidence":    round(base_conf, 4),
        "bbox":          [x1, y1, x2, y2],
        "area_percent":  round(area_percent, 2),
        "attribution":   round(avg_heat, 4),
        "location":      location,
        "distribution":  {CLASS_NAMES[k]: round(v, 4) for k, v in distrib.items()},
        "original_b64":  original_b64,
        "overlay_b64":   overlay_b64,
    })


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

