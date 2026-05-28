# ============================================================
#  Face Mask Detection - Real-Time Video Stream
#  Tech : Python, OpenCV, TensorFlow/Keras, MobileNetV2
# ============================================================

import numpy as np
import argparse
import cv2
import pickle
import time

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# ─────────────────── Argument Parser ───────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face",    type=str, default="haarcascade/haarcascade_frontalface_default.xml",
                help="Path to Haar-cascade face detector")
ap.add_argument("-m", "--model",   type=str, default="model/mask_detector.model",
                help="Path to trained mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="Minimum probability filter")
args = vars(ap.parse_args())

# ─────────────────── Helper ────────────────────────────────────
def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces     = faceNet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    locs      = []
    faces_arr = []

    for (x, y, fw, fh) in faces:
        face = frame[y:y+fh, x:x+fw]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        faces_arr.append(face)
        locs.append((x, y, x+fw, y+fh))

    preds = []
    if len(faces_arr) > 0:
        faces_arr = np.array(faces_arr, dtype="float32")
        preds     = maskNet.predict(faces_arr, batch_size=32)

    return (locs, preds)

# ─────────────────── Load Models ───────────────────────────────
print("[INFO] Loading face detector ...")
faceNet = cv2.CascadeClassifier(args["face"])

print("[INFO] Loading mask detector ...")
maskNet = load_model(args["model"])

with open("model/label_binarizer.pkl", "rb") as f:
    lb = pickle.load(f)

# ─────────────────── Start Video Stream ────────────────────────
print("[INFO] Starting video stream ...")
vs = cv2.VideoCapture(0)
time.sleep(2.0)

fps_start = time.time()
frame_count = 0

while True:
    ret, frame = vs.read()
    if not ret:
        break

    frame = cv2.resize(frame, (800, 600))
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (withoutMask, withMask)       = pred

        label = "Mask"      if withMask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask"        else (0, 0, 255)
        conf  = max(withMask, withoutMask) * 100

        cv2.putText(frame, f"{label}: {conf:.1f}%",
                    (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # FPS overlay
    frame_count += 1
    elapsed = time.time() - fps_start
    fps = frame_count / elapsed if elapsed > 0 else 0
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Overall status banner
    if locs:
        no_mask_count = sum(
            1 for pred in preds if pred[0] > pred[1]   # withoutMask > withMask
        )
        banner       = "ALERT: No Mask Detected!" if no_mask_count > 0 else "All Clear  - Masks On!"
        banner_color = (0, 0, 255)                  if no_mask_count > 0 else (0, 255, 0)
        cv2.putText(frame, banner, (10, 560),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, banner_color, 2)

    cv2.imshow("Face Mask Detection - Live", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

vs.release()
cv2.destroyAllWindows()
print("[INFO] Stream closed.")
