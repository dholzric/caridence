# caridence/analyzer/detector.py
"""High-recall YOLO detector backend.

Wraps an ultralytics YOLO model (fine-tuned on CarDD) behind the VLMBackend
protocol. Returns one Detection per detected box, with panel and severity
derived deterministically from the box geometry (the detector localizes; we
don't ask it to also guess panel/severity).
"""
from __future__ import annotations
import os
from caridence.schema import Frame, Detection, DamageType, BBox
from caridence.data.panels import infer_panel
from caridence.data.severity import severity_from_area

# YOLO class index -> DamageType, matching the CarDD data.yaml class order.
DEFAULT_CLASSES = [
    DamageType.DENT, DamageType.SCRATCH, DamageType.CRACK,
    DamageType.GLASS_SHATTER, DamageType.LAMP_BROKEN, DamageType.TIRE_FLAT,
]


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


class DetectorBackend:
    """Detector running in high-recall mode (low conf). Pair with a verifier
    (see HybridBackend) to restore precision."""

    def __init__(self, weights: str | None = None, model=None,
                 conf: float = 0.05, imgsz: int = 1024, augment: bool = True,
                 classes: list[DamageType] | None = None):
        self.conf = conf
        self.imgsz = imgsz
        self.augment = augment
        self.classes = classes or DEFAULT_CLASSES
        if model is not None:
            self.model = model
        else:
            from ultralytics import YOLO
            self.model = YOLO(weights or os.environ.get(
                "CARIDENCE_DETECTOR_WEIGHTS", "best.pt"))

    def detect(self, frame: Frame) -> list[Detection]:
        result = self.model.predict(
            frame.path, conf=self.conf, iou=0.6, imgsz=self.imgsz,
            augment=self.augment, verbose=False)[0]
        dets: list[Detection] = []
        for box in result.boxes:
            cls = int(box.cls)
            if cls < 0 or cls >= len(self.classes):
                continue
            conf = float(box.conf)
            row = box.xywhn[0]
            cx, cy, w, h = row.tolist() if hasattr(row, "tolist") else list(row)
            bbox = BBox(
                x=_clamp(cx - w / 2), y=_clamp(cy - h / 2),
                w=_clamp(w, 1e-6, 1.0), h=_clamp(h, 1e-6, 1.0))
            dets.append(Detection(
                damage_type=self.classes[cls],
                panel=infer_panel(bbox),
                severity=severity_from_area(bbox.w * bbox.h),
                bbox=bbox,
                confidence=conf,
            ))
        return dets
