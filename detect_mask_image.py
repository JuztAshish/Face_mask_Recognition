# ============================================================
#  Face Mask Detection - Static Image Inference
#  Tech : Python, OpenCV, TensorFlow/Keras, MobileNetV2
# ============================================================

import numpy as np
import argparse
import cv2
import os
import pickle

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# ─────────────────── Argument Parser ───────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image",   required=True,  help="Path to input image")
ap.add_argument("-f", "--face",    type=str,       default="haarcascade/haarcascade_frontalface_default.xml",
                help="Path to face detector model")
ap.add_argument("-m", "--model",   type=str,       default="model/mask_detector.model",
                help="Path to trained mask detector model")
ap.add_argument("-c", "--confidence", type=float,  default=0.5,
                help="Minimum probability to filter weak detections")
args = vars(ap.parse_args())

# ─────────────────── Load Models ───────────────────────────────
print("[INFO] Loading face detector ...")
faceNet  = cv2.CascadeClassifier(args["face"])

print("[INFO] Loading mask detector ...")
maskNet  = load_model(args["model"])

with open("model/label_binarizer.pkl", "rb") as f:
    lb = pickle.load(f)

# ─────────────────── Load & Pre-process Image ──────────────────
image = cv2.imread(args["image"])
orig  = image.copy()
(h, w) = image.shape[:2]

gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = faceNet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

faces_arr  = []
locs       = []
preds      = []

for (x, y, fw, fh) in faces:
    face = image[y:y+fh, x:x+fw]
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    face = img_to_array(face)
    face = preprocess_input(face)
    faces_arr.append(face)
    locs.append((x, y, x+fw, y+fh))

if len(faces_arr) > 0:
    faces_arr = np.array(faces_arr, dtype="float32")
    preds     = maskNet.predict(faces_arr, batch_size=32)

# ─────────────────── Draw Results ──────────────────────────────
for (box, pred) in zip(locs, preds):
    (startX, startY, endX, endY) = box
    (withoutMask, withMask)      = pred          # order depends on lb.classes_

    label = "Mask"     if withMask > withoutMask else "No Mask"
    color = (0, 255, 0) if label == "Mask"       else (0, 0, 255)
    conf  = max(withMask, withoutMask) * 100

    label_text = f"{label}: {conf:.2f}%"
    cv2.putText(image, label_text, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

os.makedirs("output", exist_ok=True)
output_path = "output/result_image.jpg"
cv2.imwrite(output_path, image)
cv2.imshow("Face Mask Detection", image)
cv2.waitKey(0)
print(f"[INFO] Result saved → {output_path}")
