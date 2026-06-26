# Caridence detection results — the recall breakthrough

**Eval:** CarDD test split — 374 images, 785 damage instances. Same set across all models.

## The problem we solved

The fine-tuned VLM (Qwen2.5-VL-7B) topped out at **48.5% instance recall**. A diagnostic
showed why: the VLM *under-enumerates* — it predicts ~1.5 boxes/image when there are ~2.1,
naming the obvious damage and stopping. It knew damage was present (~78% presence recall)
but wouldn't localize every instance. That is exactly the job a dedicated object detector
does and a VLM does not.

## The fix: YOLOv11-L detector (1024px) + test-time augmentation

Fine-tuned on CarDD (2,816 train images), 100 epochs, on a single RTX 3090.

| Model | Instance recall (IoU 0.3) | Presence recall | Precision | Latency |
|---|---|---|---|---|
| **Detector + TTA (high-recall, conf 0.05)** | **89.2%** | **97.8%** | 33.5% | ~120 ms |
| Detector + TTA (balanced, conf 0.25) | 81.0% | 94.1% | 59.6% | ~120 ms |
| Caridence-7B VLM (fine-tuned) | 48.5% | 78% | 67.3% | 6,167 ms |
| Qwen2.5-VL base (zero-shot) | 0.1% | 0% | 100% | 4,612 ms |

At conf 0.01 the detector reaches **92.7% instance recall**; at IoU 0.1, **95–98%**.

### Per-class instance recall (detector + TTA, conf 0.05, IoU 0.3)

| Class | Recall |
|---|---|
| glass shatter | 0.97 |
| lamp broken | 0.93 |
| tire flat | 0.91 |
| scratch | 0.90 |
| dent | 0.88 |
| crack | 0.80 |

Every class ≥ 0.80 — including crack, the VLM's worst (0.21).

## Recommended product architecture (hybrid)

Run the **detector in high-recall mode** (conf ~0.05) to catch ~98% of damage at the
vehicle level, then use the **VLM to verify each candidate crop** (and add severity +
description). This keeps the detector's recall while restoring precision — the right
design for an inspection tool, where missing damage in a dispute is the costly error.

## Reproduce

```bash
# data: kaggle datasets download gabrielfcarvalho/cardd-with-yolo-annotations-images-labels
bash scripts/train_yolo.sh                 # YOLOv11-L @1024, 100 epochs
python scripts/yolo_eval.py <best.pt> --aug   # conf/IoU sweep + per-class + presence
```

Weights: `whitelinux:~/caridence/runs/detect/outputs/yolo/cardd/weights/best.pt`.
