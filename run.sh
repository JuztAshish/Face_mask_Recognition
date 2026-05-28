#!/usr/bin/env bash
# ============================================================
#  run.sh – convenience script for common project commands
# ============================================================

set -e

ACTION=${1:-"help"}

case "$ACTION" in

  install)
    echo "[INFO] Installing dependencies ..."
    pip install -r requirements.txt
    ;;

  download-cascade)
    echo "[INFO] Downloading Haar cascade ..."
    wget -q "https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml" \
         -O haarcascade/haarcascade_frontalface_default.xml
    echo "[OK]  Saved to haarcascade/"
    ;;

  dataset-check)
    python utils/dataset_utils.py --dataset dataset --action count
    python utils/dataset_utils.py --dataset dataset --action verify
    ;;

  train)
    python train_mask_detector.py \
        --dataset dataset \
        --model   model/mask_detector.model \
        --plot    output/training_plot.png
    ;;

  detect-image)
    IMAGE=${2:-"test.jpg"}
    python detect_mask_image.py \
        --image "$IMAGE" \
        --model model/mask_detector.model
    ;;

  detect-video)
    python detect_mask_video.py \
        --model model/mask_detector.model
    ;;

  evaluate)
    python utils/evaluate_model.py \
        --dataset dataset \
        --model   model/mask_detector.model \
        --output  output
    ;;

  help|*)
    echo ""
    echo "Usage: bash run.sh <action> [args]"
    echo ""
    echo "  install           Install Python dependencies"
    echo "  download-cascade  Download Haar cascade XML"
    echo "  dataset-check     Count & verify dataset images"
    echo "  train             Train the mask detector"
    echo "  detect-image      Detect mask in a static image"
    echo "  detect-video      Real-time webcam detection"
    echo "  evaluate          Full evaluation (CM, ROC, report)"
    echo ""
    ;;
esac
