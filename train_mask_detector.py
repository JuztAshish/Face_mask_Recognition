# ============================================================
#  Face Mask Detection - Model Training
#  Author  : Your Name
#  Tech    : Python, TensorFlow, Keras, MobileNetV2, OpenCV
# ============================================================

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import argparse
import pickle

# ─────────────────────── Argument Parser ───────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset",  required=True,  help="Path to input dataset")
ap.add_argument("-p", "--plot",     type=str,       default="output/training_plot.png", help="Path to output loss/accuracy plot")
ap.add_argument("-m", "--model",    type=str,       default="model/mask_detector.model", help="Path to output face mask detector model")
args = vars(ap.parse_args())

# ─────────────────────── Hyper-parameters ──────────────────────
INIT_LR   = 1e-4
EPOCHS    = 20
BATCH_SIZE = 32
IMG_SIZE  = (224, 224)

print("[INFO] Loading images ...")
imagePaths = list(paths.list_images(args["dataset"]))
data   = []
labels = []

for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]          # folder name = class label
    image = load_img(imagePath, target_size=IMG_SIZE)
    image = img_to_array(image)
    image = preprocess_input(image)
    data.append(image)
    labels.append(label)

data   = np.array(data,   dtype="float32")
labels = np.array(labels)

# One-hot encode labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Train / Test split
(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.20, stratify=labels, random_state=42
)

# ─────────────────── Data Augmentation ─────────────────────────
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

# ─────────────────── Build Model ───────────────────────────────
print("[INFO] Building model ...")
baseModel = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_tensor=Input(shape=(224, 224, 3))
)

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2,   activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze base layers
for layer in baseModel.layers:
    layer.trainable = False

# ─────────────────── Compile & Train ───────────────────────────
print("[INFO] Compiling model ...")
from tensorflow.keras.optimizers.legacy import Adam

opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] Training head ...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
    steps_per_epoch=len(trainX) // BATCH_SIZE,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BATCH_SIZE,
    epochs=EPOCHS
)

# ─────────────────── Evaluate ──────────────────────────────────
print("[INFO] Evaluating network ...")
predIdxs = model.predict(testX, batch_size=BATCH_SIZE)
predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(
    testY.argmax(axis=1),
    predIdxs,
    target_names=lb.classes_
))

# ─────────────────── Save Artifacts ────────────────────────────
os.makedirs("model",  exist_ok=True)
os.makedirs("output", exist_ok=True)

print(f"[INFO] Saving model  → {args['model']}")
model.save(args["model"], save_format="h5")

print("[INFO] Saving label binarizer ...")
with open("model/label_binarizer.pkl", "wb") as f:
    pickle.dump(lb, f)

# ─────────────────── Plot Training Curve ───────────────────────
N = len(H.history["loss"])
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"],          label="train_loss")
if "val_loss" in H.history:
    plt.plot(np.arange(0, len(H.history["val_loss"])),
             H.history["val_loss"],
             label="val_loss")

if "val_accuracy" in H.history:
    plt.plot(np.arange(0, len(H.history["val_accuracy"])),
             H.history["val_accuracy"],
             label="val_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"],  label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss / Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
print(f"[INFO] Training plot saved → {args['plot']}")
