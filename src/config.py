# ============================================================
#  src/config.py  –  Central project configuration
# ============================================================

import os

# ── Paths ────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH    = os.path.join(BASE_DIR, "dataset")
MODEL_PATH      = os.path.join(BASE_DIR, "model", "mask_detector.model")
LB_PATH         = os.path.join(BASE_DIR, "model", "label_binarizer.pkl")
HAARCASCADE     = os.path.join(BASE_DIR, "haarcascade",
                               "haarcascade_frontalface_default.xml")
OUTPUT_DIR      = os.path.join(BASE_DIR, "output")
PLOT_PATH       = os.path.join(OUTPUT_DIR, "training_plot.png")

# ── Training hyper-parameters ───────────────────────────────
INIT_LR         = 1e-4
EPOCHS          = 20
BATCH_SIZE      = 32
IMG_SIZE        = (224, 224)
TEST_SPLIT      = 0.20
RANDOM_SEED     = 42

# ── Inference settings ──────────────────────────────────────
CONFIDENCE_THRESH = 0.50          # minimum confidence to accept a detection
FRAME_WIDTH       = 800
FRAME_HEIGHT      = 600

# ── Labels & colors (BGR) ───────────────────────────────────
CLASSES = {
    "Mask"    : (0, 255, 0),      # green
    "No Mask" : (0, 0, 255),      # red
}
