# ============================================================
#  src/model.py  –  MobileNetV2-based mask detector
# ============================================================

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (AveragePooling2D, Dropout,
                                      Flatten, Dense, Input)
from tensorflow.keras.models import Model


def build_mask_detector(img_size=(224, 224, 3), num_classes=2):
    """
    Construct a fine-tuned MobileNetV2 classifier for mask detection.

    Args:
        img_size    : Input tensor shape  (H, W, C)
        num_classes : Number of output classes (default 2 : mask / no-mask)

    Returns:
        keras.Model ready for compilation
    """
    # ── Base model (ImageNet weights, no top) ────────────────
    baseModel = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_tensor=Input(shape=img_size)
    )

    # ── Custom classification head ───────────────────────────
    head = baseModel.output
    head = AveragePooling2D(pool_size=(7, 7))(head)
    head = Flatten(name="flatten")(head)
    head = Dense(128, activation="relu")(head)
    head = Dropout(0.5)(head)
    head = Dense(num_classes, activation="softmax")(head)

    model = Model(inputs=baseModel.input, outputs=head)

    # ── Freeze base layers (transfer learning) ───────────────
    for layer in baseModel.layers:
        layer.trainable = False

    return model


def unfreeze_top_layers(model, n_layers=20):
    """
    Unfreeze the last `n_layers` of the base model for fine-tuning.
    Call after the head has converged.
    """
    for layer in model.layers[-n_layers:]:
        layer.trainable = True
    return model


if __name__ == "__main__":
    m = build_mask_detector()
    m.summary()
