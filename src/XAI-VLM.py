import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH = r"C:\Users\abhis\Documents\yolo_merge\runs\detect\runs\detect\yolov12n_safe_training\weights\best.pt"
IMAGE_PATH = r"C:\Users\abhis\Documents\yolo_merge\brain_merged_final_2\images\val\img_11.jpg"

PATCH_SIZE = 20
STRIDE = 8
GAUSS_SIGMA = 5
FINAL_SMOOTH = 6

# ============================================================

model = YOLO(MODEL_PATH)
model.to("cpu")

img_bgr = cv2.imread(IMAGE_PATH)
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
h, w, _ = img.shape

# ------------------------------------------------------------
# BASE PREDICTION
# ------------------------------------------------------------
base = model(IMAGE_PATH, verbose=False)[0]

if base.boxes is None or len(base.boxes) == 0:
    print("❌ No tumor detected")
    raise SystemExit

box = base.boxes[0]
base_conf = float(box.conf[0])
cls_id = int(box.cls[0])
cls_name = model.names[cls_id]

x1, y1, x2, y2 = map(int, box.xyxy[0])

# ------------------------------------------------------------
# OCCLUSION ONLY INSIDE BBOX
# ------------------------------------------------------------
heatmap = np.zeros((h, w), dtype=np.float32)

for y in range(y1, y2 - PATCH_SIZE, STRIDE):
    for x in range(x1, x2 - PATCH_SIZE, STRIDE):

        occluded = img_bgr.copy()

        roi = occluded[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
        roi_blur = cv2.GaussianBlur(roi, (0, 0), GAUSS_SIGMA)
        occluded[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = roi_blur

        result = model(occluded, verbose=False)[0]

        if result.boxes is None or len(result.boxes) == 0:
            conf = 0.0
        else:
            conf = float(result.boxes[0].conf[0])

        drop = base_conf - conf
        if drop > 0:
            heatmap[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += drop

# ------------------------------------------------------------
# NORMALIZE ONLY BBOX AREA
# ------------------------------------------------------------
roi = heatmap[y1:y2, x1:x2]

if roi.size > 0:
    roi = roi / (np.max(roi) + 1e-6)
    heatmap[y1:y2, x1:x2] = roi

heatmap = cv2.GaussianBlur(heatmap, (0, 0), FINAL_SMOOTH)

# Zero outside bbox strictly
mask = np.zeros_like(heatmap)
mask[y1:y2, x1:x2] = heatmap[y1:y2, x1:x2]
heatmap = mask

# ------------------------------------------------------------
# CREATE OVERLAY
# ------------------------------------------------------------
heatmap_uint8 = np.uint8(255 * heatmap)
heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

overlay = img.copy()
overlay[y1:y2, x1:x2] = cv2.addWeighted(
    img[y1:y2, x1:x2],
    0.6,
    heatmap_color[y1:y2, x1:x2],
    0.4,
    0
)

cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

# ------------------------------------------------------------
# DISPLAY
# ------------------------------------------------------------
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(overlay)
plt.title(f"{cls_name.upper()} | Confidence: {base_conf:.3f}")
plt.axis("off")

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# SHORT XAI INTERPRETATION
# ------------------------------------------------------------

area_percent = ((x2-x1)*(y2-y1)) / (w*h) * 100
avg_heat = np.mean(heatmap[y1:y2, x1:x2])

print("\n================ XAI INTERPRETATION ================\n")
print(f"The model predicts {cls_name.upper()} with confidence {base_conf:.3f}.")
print(f"The highlighted region inside the bounding box shows")
print(f"localized importance contributing to the classification.")
print(f"The tumor region covers approximately {area_percent:.2f}% of the image area.")
print(f"The average attribution strength inside the box is {avg_heat:.4f}.")
print("\nThis indicates that the prediction is primarily driven")
print("by features within the detected lesion area.")
print("\n====================================================\n")