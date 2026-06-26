# caridence/redact.py
"""Privacy redaction for inspection frames.

Detects license plates (and optionally other sensitive regions) with a YOLO
model and obscures them, so plates never reach the report, cited frames, or the
demo. Redaction writes to a separate output path — it never mutates originals.
"""
from __future__ import annotations
import os
import cv2
from caridence.schema import Frame


def _xyxy(box):
    row = box.xyxy[0]
    return row.tolist() if hasattr(row, "tolist") else list(row)


class PlateRedactor:
    """Detect license plates and obscure them.

    method: 'blur' (gaussian), 'pixelate' (mosaic), or 'box' (solid fill).
    """

    def __init__(self, weights: str | None = None, model=None,
                 conf: float = 0.35, method: str = "blur", pad: int = 4,
                 min_aspect: float = 1.6, max_aspect: float = 8.0):
        self.conf = conf
        self.method = method
        self.pad = pad
        # License plates are wide rectangles; this rejects square false
        # positives (wheels/badges) that a plate detector can fire on.
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        if model is not None:
            self.model = model
        else:
            from ultralytics import YOLO
            self.model = YOLO(weights or os.environ.get(
                "CARIDENCE_PLATE_WEIGHTS", "plates.pt"))

    def _regions(self, image_path: str, w: int, h: int):
        res = self.model.predict(image_path, conf=self.conf, verbose=False)[0]
        out = []
        for b in res.boxes:
            rx1, ry1, rx2, ry2 = _xyxy(b)
            bw, bh = rx2 - rx1, ry2 - ry1
            if bw <= 0 or bh <= 0:
                continue
            aspect = bw / bh
            if aspect < self.min_aspect or aspect > self.max_aspect:
                continue  # not plate-shaped -> skip (avoid blurring wheels/damage)
            x1 = max(0, int(rx1) - self.pad); y1 = max(0, int(ry1) - self.pad)
            x2 = min(w, int(rx2) + self.pad); y2 = min(h, int(ry2) + self.pad)
            if x2 > x1 and y2 > y1:
                out.append((x1, y1, x2, y2))
        return out

    def _obscure(self, roi):
        if self.method == "box":
            roi[:] = 0
            return roi
        if self.method == "pixelate":
            h, w = roi.shape[:2]
            small = cv2.resize(roi, (max(1, w // 12), max(1, h // 12)),
                               interpolation=cv2.INTER_LINEAR)
            return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        k = max(15, (min(roi.shape[:2]) // 2) * 2 + 1)  # odd kernel
        return cv2.GaussianBlur(roi, (k, k), 0)

    def redact(self, image_path: str, out_path: str) -> str:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"cannot read image: {image_path}")
        h, w = img.shape[:2]
        for (x1, y1, x2, y2) in self._regions(image_path, w, h):
            img[y1:y2, x1:x2] = self._obscure(img[y1:y2, x1:x2])
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        cv2.imwrite(out_path, img)
        return out_path


def apply_redaction(frames: list[Frame], redactor: PlateRedactor,
                    out_dir: str) -> list[Frame]:
    """Redact each frame into out_dir, returning frames that point at the
    redacted copies. Originals are never modified."""
    os.makedirs(out_dir, exist_ok=True)
    out: list[Frame] = []
    for f in frames:
        dst = os.path.join(out_dir, os.path.basename(f.path))
        redactor.redact(f.path, dst)
        out.append(f.model_copy(update={"path": dst}))
    return out
