# Brain Tumor Detection using YOLOv12n

This project focuses on detecting brain tumors from MRI images using the YOLOv12n object detection model. The system integrates advanced image preprocessing techniques to enhance tumor visibility and improve detection performance.

The project also generates interpretable visual outputs that resemble explainable AI (XAI) and vision-language model (VLM) style interpretations.

---

## Project Overview

Brain tumors are one of the most critical neurological conditions, and early detection is essential for effective treatment. This project applies deep learning-based object detection to identify tumor regions in MRI scans.

The workflow includes preprocessing MRI images, training a YOLOv12n model, and generating detection outputs that highlight tumor regions.

---

## Key Features

- Brain tumor detection using **YOLOv12n**
- MRI preprocessing to enhance tumor visibility
- Integration of two medical imaging datasets
- Visual outputs resembling **explainable AI interpretations**
- Automated tumor localization using bounding boxes

---

## Datasets Used

### BraTS 2020 Dataset
The Brain Tumor Segmentation (BraTS) dataset provides multimodal MRI scans including tumor annotations.

Dataset link:
https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation

### Figshare Brain Tumor Dataset
This dataset contains MRI images labeled for different types of brain tumors.

Dataset link:
https://www.kaggle.com/datasets/ashkhagan/figshare-brain-tumor-dataset

---

## Image Preprocessing Techniques

To improve tumor visibility and model performance, the following preprocessing techniques were applied:

### Log Transformation
Enhances low-intensity regions in MRI scans and improves contrast for subtle tumor regions.

### Histogram Equalization
Redistributes intensity values to improve global contrast in MRI images.

### CLAHE (Contrast Limited Adaptive Histogram Equalization)
Enhances local contrast and prevents over-amplification of noise.

These preprocessing steps significantly improve the clarity of tumor regions before detection.

---

## Model Architecture

The detection model used in this project is:

YOLOv12n (nano version)

Reasons for choosing YOLOv12n:
- Lightweight architecture
- Fast inference
- Efficient object detection performance

The model was trained to detect tumor regions and output bounding boxes around abnormal regions in MRI scans.

---

## Project Workflow

1. Load MRI datasets (BraTS2020 and Figshare)
2. Apply preprocessing techniques:
   - Log Transform
   - Histogram Equalization
   - CLAHE
3. Convert annotations to YOLO format
4. Train YOLOv12n detection model
5. Perform tumor detection on MRI images
6. Generate detection visualizations
7. Produce interpretable visual outputs similar to XAI/VLM explanations

---

## Technologies Used

- Python
- PyTorch
- Ultralytics YOLO
- OpenCV
- NumPy
- Matplotlib
- Scikit-image

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/brain-tumor-detection-yolov12.git
cd brain-tumor-detection-yolov12
