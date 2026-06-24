# caridence/metrics.py
from __future__ import annotations
from pydantic import BaseModel
from caridence.schema import BBox, Detection
from caridence.data.types import LabeledImage


class DetectionScore(BaseModel):
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int
    images_all_caught: float
    fp_per_image: float


def iou(a: BBox, b: BBox) -> float:
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h
    ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a.w * a.h + b.w * b.h - inter
    return inter / union


def _match_image(gt: list[Detection], pred: list[Detection], iou_thr: float) -> tuple[int, int, int]:
    """Greedy match within an image. Returns (tp, fp, fn)."""
    candidates = []
    for pi, p in enumerate(pred):
        for gi, g in enumerate(gt):
            if p.damage_type == g.damage_type:
                ov = iou(p.bbox, g.bbox)
                if ov >= iou_thr:
                    candidates.append((ov, pi, gi))
    candidates.sort(reverse=True)
    used_p, used_g = set(), set()
    tp = 0
    for ov, pi, gi in candidates:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi); used_g.add(gi); tp += 1
    fp = len(pred) - len(used_p)
    fn = len(gt) - len(used_g)
    return tp, fp, fn


def score_predictions(gt_images: list[LabeledImage],
                      predictions: list[list[Detection]],
                      iou_thr: float = 0.3) -> DetectionScore:
    assert len(gt_images) == len(predictions), "gt/pred length mismatch"
    TP = FP = FN = 0
    all_caught = 0
    for img, pred in zip(gt_images, predictions):
        tp, fp, fn = _match_image(img.detections, pred, iou_thr)
        TP += tp; FP += fp; FN += fn
        if fn == 0:
            all_caught += 1
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    n = len(gt_images) or 1
    return DetectionScore(
        precision=precision, recall=recall, f1=f1,
        true_positives=TP, false_positives=FP, false_negatives=FN,
        images_all_caught=all_caught / n, fp_per_image=FP / n,
    )
