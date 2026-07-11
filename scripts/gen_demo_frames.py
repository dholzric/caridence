"""Generate clean, honest demo stills with cardd_v3 for the submission video.

For the van (dent) and chevy (scratches) walkarounds: sample frames, run the
v3 detector at serving config (conf 0.4, body classes, TTA), annotate frames
that have detections, and save them. Also emit real plate-redaction before/
after pairs from our own footage.

Run: .venv/Scripts/python scripts/gen_demo_frames.py
Output: scratchpad demo_frames/ (annotated) and redact/ (before/after).
"""
import os
from pathlib import Path

import cv2
from ultralytics import YOLO

OUT = Path(os.environ.get("DEMO_OUT", "demo_frames"))
(OUT / "van").mkdir(parents=True, exist_ok=True)
(OUT / "chevy").mkdir(parents=True, exist_ok=True)
(OUT / "redact").mkdir(parents=True, exist_ok=True)

NAMES = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"]
COLOR = (60, 60, 235)  # BGR red

det = YOLO("weights/cardd_v3.pt")
plate = YOLO("weights/plates_best.pt")

TARGETS = {
    "van": "demo_videos/20260626_143914.mp4",
    "chevy": "demo_videos/20260626_143759.mp4",
}


def severity(area_frac):
    if area_frac > 0.04:
        return "severe"
    if area_frac > 0.012:
        return "moderate"
    return "minor"


def annotate(img, boxes):
    H, W = img.shape[:2]
    out = img.copy()
    for cls, conf, (x1, y1, x2, y2) in boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), COLOR, 4)
        area = ((x2 - x1) * (y2 - y1)) / (W * H)
        label = f"{NAMES[cls]}  {severity(area)}  {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
        cv2.rectangle(out, (x1, y1 - th - 14), (x1 + tw + 12, y1), COLOR, -1)
        cv2.putText(out, label, (x1 + 6, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
    return out


def sample(video, step_fps=3):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60
    step = max(1, int(fps / step_fps))
    i = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i % step == 0:
            yield i, img
        i += 1
    cap.release()


def main():
    for tag, video in TARGETS.items():
        kept = 0
        for fi, img in sample(video):
            r = det.predict(img, conf=0.4, iou=0.6, imgsz=1024, augment=True,
                            classes=[0, 1, 2], verbose=False)[0]
            boxes = []
            for b in r.boxes:
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                boxes.append((int(b.cls), float(b.conf), (x1, y1, x2, y2)))
            if not boxes:
                continue
            out = annotate(img, boxes)
            H, W = out.shape[:2]
            out = cv2.resize(out, (1080, int(H * 1080 / W)))
            cv2.imwrite(str(OUT / tag / f"{tag}_{fi:04d}.jpg"), out)
            kept += 1
        print(f"{tag}: {kept} annotated frames", flush=True)

    # Real plate redaction before/after (one good frame per vehicle).
    for tag, video in TARGETS.items():
        for fi, img in sample(video, step_fps=1):
            pr = plate.predict(img, conf=0.35, imgsz=1280, verbose=False)[0]
            if not len(pr.boxes):
                continue
            before = img.copy()
            after = img.copy()
            for b in pr.boxes:
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                roi = after[y1:y2, x1:x2]
                if roi.size:
                    k = max(15, ((x2 - x1) // 6) | 1)
                    after[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
            H, W = img.shape[:2]
            nb = cv2.resize(before, (1080, int(H * 1080 / W)))
            na = cv2.resize(after, (1080, int(H * 1080 / W)))
            cv2.imwrite(str(OUT / "redact" / f"{tag}_before.jpg"), nb)
            cv2.imwrite(str(OUT / "redact" / f"{tag}_after.jpg"), na)
            print(f"{tag}: redaction pair at frame {fi}", flush=True)
            break


if __name__ == "__main__":
    main()
