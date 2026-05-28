# 😷 Face Mask Detection System
### Real-Time Deep Learning with MobileNetV2 | Python · OpenCV · TensorFlow · Keras

---

## 📌 Project Overview

A production-ready **Face Mask Detection** system that uses **Transfer Learning** on top of
**MobileNetV2** to classify whether a detected face is wearing a mask or not.  
The system works on:
- 📷 **Static images**
- 🎥 **Live webcam / video streams**

---

## 🏗️ Project Structure

```
face_mask_detection/
│
├── train_mask_detector.py      ← Train MobileNetV2 model
├── detect_mask_image.py        ← Run detection on a static image
├── detect_mask_video.py        ← Run real-time webcam detection
│
├── src/
│   ├── config.py               ← Central configuration (paths, hyper-params)
│   ├── model.py                ← MobileNetV2 architecture builder
│   └── inference.py            ← Shared detection / annotation pipeline
│
├── utils/
│   ├── dataset_utils.py        ← Dataset counting, verification, resizing
│   └── evaluate_model.py       ← Confusion matrix, ROC curve, full report
│
├── dataset/
│   ├── with_mask/              ← Training images (mask on)
│   └── without_mask/           ← Training images (no mask)
│
├── model/
│   ├── mask_detector.model     ← Saved Keras model (generated after training)
│   └── label_binarizer.pkl     ← Serialised LabelBinarizer
│
├── haarcascade/
│   └── haarcascade_frontalface_default.xml   ← OpenCV face detector
│
├── output/
│   ├── training_plot.png       ← Loss / accuracy curves
│   ├── confusion_matrix.png    ← Evaluation confusion matrix
│   ├── roc_curve.png           ← ROC curve + AUC
│   └── classification_report.txt
│
└── requirements.txt
```

---

## ⚙️ Technologies Used

| Tool | Role |
|------|------|
| **Python 3.8+** | Core language |
| **TensorFlow / Keras** | Deep learning framework |
| **MobileNetV2** | Lightweight CNN backbone (ImageNet pre-trained) |
| **OpenCV** | Image / video I/O, face detection (Haar cascade) |
| **scikit-learn** | Train/test split, metrics, label encoding |
| **Matplotlib** | Training curves, evaluation plots |
| **imutils** | Convenience utilities for image processing |

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Haar Cascade
Download from OpenCV's official GitHub and place it in `haarcascade/`:
```
https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
```

### 3. Prepare your dataset
Place images in:
```
dataset/
    with_mask/       ← images of people wearing masks
    without_mask/    ← images of people without masks
```

**Verify & check dataset:**
```bash
python utils/dataset_utils.py --dataset dataset --action count
python utils/dataset_utils.py --dataset dataset --action verify
```

### 4. Train the model
```bash
python train_mask_detector.py \
    --dataset dataset \
    --model   model/mask_detector.model \
    --plot    output/training_plot.png
```

### 5. Detect on a static image
```bash
python detect_mask_image.py \
    --image path/to/image.jpg \
    --model model/mask_detector.model
```

### 6. Real-time webcam detection
```bash
python detect_mask_video.py \
    --model model/mask_detector.model
```
> Press **`q`** to quit the live stream.

### 7. Full model evaluation
```bash
python utils/evaluate_model.py \
    --dataset dataset \
    --model   model/mask_detector.model \
    --output  output
```

---

## 🧠 Model Architecture

```
Input (224×224×3)
       │
 MobileNetV2 backbone  ← frozen during initial training
       │
 AveragePooling2D (7×7)
       │
 Flatten
       │
 Dense(128, ReLU)
       │
 Dropout(0.5)
       │
 Dense(2, Softmax)     ← [with_mask, without_mask]
```

**Why MobileNetV2?**
- 3.4M parameters — fast enough for real-time inference on CPU
- Pre-trained on ImageNet → excellent low-level feature extraction
- Inverted residuals + linear bottlenecks → efficient and accurate

---

## 📊 Training Configuration

| Parameter | Value |
|-----------|-------|
| Input size | 224 × 224 |
| Learning rate | 1e-4 |
| Optimiser | Adam |
| Loss | Binary Cross-Entropy |
| Epochs | 20 |
| Batch size | 32 |
| Train / Test split | 80 / 20 |
| Data augmentation | rotation, zoom, shift, flip |

---

## 📈 Output Artifacts

After training and evaluation the `output/` directory contains:

- **`training_plot.png`** – loss & accuracy curves over epochs
- **`confusion_matrix.png`** – true positives vs false positives
- **`roc_curve.png`** – AUC-ROC performance curve
- **`classification_report.txt`** – precision, recall, F1-score per class

---

## 💡 Tips & Troubleshooting

| Issue | Fix |
|-------|-----|
| Low accuracy | Add more diverse images; try increasing EPOCHS |
| Slow inference | Use `--confidence 0.6` to skip borderline faces |
| Camera not found | Change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` |
| ImportError | Re-run `pip install -r requirements.txt` |
| `None` face detections | Improve lighting; decrease `minNeighbors` in cascade |

---

## 📄 License

MIT – free to use, modify, and distribute for academic and commercial projects.

---

## 🙌 Acknowledgements

- [MobileNetV2 paper](https://arxiv.org/abs/1801.04381) – Sandler et al., 2018  
- [OpenCV Haar cascades](https://github.com/opencv/opencv)  
- Dataset inspiration: [Prajna Bhandary's dataset](https://github.com/prajnasb/observations)
