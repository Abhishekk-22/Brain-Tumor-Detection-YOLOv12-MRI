
from ultralytics import YOLO

model = YOLO("yolo12n.pt")   # nano model

results = model.train(
    data="brain_merged_detect/data.yaml",
    imgsz=512,
    epochs=150,
    batch=4,
    optimizer="AdamW",
    lr0=0.001,
    device=0,          # use GPU (remove if CPU)
    workers=2,        # reduce workers for stability

    # -------- Light medical-safe augmentation --------
    hsv_h=0.02,
    hsv_s=0.2,
    hsv_v=0.2,
    scale=0.05,
    translate=0.03,
    mosaic=0.3,
    mixup=0.05,
    degrees=5,

    # -------- Training stability --------
    cos_lr=True,
    patience=30,
    label_smoothing=0.05,

       
    name="yolov12_merge",    # experiment name
    exist_ok=True    
)