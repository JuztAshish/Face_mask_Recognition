# ============================================================
#  src/inference.py  –  Shared detection pipeline
# ============================================================

import numpy as np
import cv2

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array


def detect_faces(frame, faceNet, min_size=(60, 60)):
    """
    Detect faces using a Haar-cascade classifier.

    Returns
    -------
    list of (x, y, x2, y2) bounding boxes
    """
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = faceNet.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=min_size
    )
    boxes = []
    for (x, y, w, h) in rects:
        boxes.append((x, y, x + w, y + h))
    return boxes


def preprocess_faces(frame, boxes, target=(224, 224)):
    """
    Crop and pre-process each detected face for MobileNetV2 input.

    Returns
    -------
    np.ndarray  shape (N, 224, 224, 3)  or empty array
    """
    faces = []
    for (x1, y1, x2, y2) in boxes:
        face = frame[y1:y2, x1:x2]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, target)
        face = img_to_array(face)
        face = preprocess_input(face)
        faces.append(face)

    if faces:
        return np.array(faces, dtype="float32")
    return np.empty((0, *target, 3), dtype="float32")


def predict_masks(faces_arr, maskNet):
    """
    Run the mask detector on pre-processed face crops.

    Returns
    -------
    np.ndarray of shape (N, 2)  – (without_mask_prob, with_mask_prob)
    """
    if faces_arr.shape[0] == 0:
        return np.empty((0, 2), dtype="float32")
    return maskNet.predict(faces_arr, batch_size=32)


def annotate_frame(frame, boxes, preds, conf_thresh=0.5):
    """
    Draw bounding boxes and labels on `frame` in-place.

    Returns annotated frame.
    """
    for (box, pred) in zip(boxes, preds):
        (startX, startY, endX, endY) = box
        (withoutMask, withMask)       = pred

        label = "Mask"      if withMask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask"        else (0, 0, 255)
        conf  = max(withMask, withoutMask) * 100

        cv2.putText(frame, f"{label}: {conf:.1f}%",
                    (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    return frame
