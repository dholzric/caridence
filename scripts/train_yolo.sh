#!/usr/bin/env bash
set -e
export PATH="$HOME/.local/bin:$PATH"
cd ~/caridence
source .venv/bin/activate
yolo detect train model=yolo11l.pt data=data/kaggle/cardd_yolo/data.yaml \
  epochs=100 imgsz=1024 batch=8 device=0 \
  project=outputs/yolo name=cardd patience=30 exist_ok=True
echo "YOLO_TRAIN_DONE"
