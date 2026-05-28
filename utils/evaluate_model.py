# ============================================================
#  utils/evaluate_model.py
#  Full evaluation : confusion matrix, ROC, classification report
# ============================================================

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, pickle, argparse

from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_curve, auc, ConfusionMatrixDisplay)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from imutils import paths

# ─────────────────── Args ──────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True)
ap.add_argument("-m", "--model",   default="model/mask_detector.model")
ap.add_argument("-o", "--output",  default="output")
args = vars(ap.parse_args())

os.makedirs(args["output"], exist_ok=True)

# ─────────────────── Load Data ─────────────────────────────────
print("[INFO] Loading images ...")
imagePaths = list(paths.list_images(args["dataset"]))
data, labels = [], []

with open("model/label_binarizer.pkl", "rb") as f:
    lb = pickle.load(f)

for ip in imagePaths:
    label = ip.split(os.path.sep)[-2]
    img   = load_img(ip, target_size=(224, 224))
    img   = preprocess_input(img_to_array(img))
    data.append(img)
    labels.append(label)

data   = np.array(data,   dtype="float32")
labels = np.array(labels)

# ─────────────────── Load Model ────────────────────────────────
print("[INFO] Loading model ...")
model  = load_model(args["model"])

# ─────────────────── Predict ───────────────────────────────────
print("[INFO] Predicting ...")
probs     = model.predict(data, batch_size=32)
predIdxs  = np.argmax(probs, axis=1)
trueIdxs  = np.array([np.where(lb.classes_ == l)[0][0] for l in labels])

# ─────────────────── Classification Report ─────────────────────
report = classification_report(trueIdxs, predIdxs, target_names=lb.classes_)
print(report)
with open(os.path.join(args["output"], "classification_report.txt"), "w") as f:
    f.write(report)

# ─────────────────── Confusion Matrix ──────────────────────────
cm  = confusion_matrix(trueIdxs, predIdxs)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lb.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(os.path.join(args["output"], "confusion_matrix.png"))
plt.close()
print("[INFO] Confusion matrix saved.")

# ─────────────────── ROC Curve ─────────────────────────────────
fpr, tpr, _ = roc_curve(trueIdxs, probs[:, 1])
roc_auc     = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=2,
         label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.savefig(os.path.join(args["output"], "roc_curve.png"))
plt.close()
print(f"[INFO] ROC AUC = {roc_auc:.4f}  |  plot saved.")
