"""Teacher-label the van walkaround for cardd_v3 real-domain training.

The first realdomain build (local Qwen 7B teacher) confirmed 0 van boxes even
though the detector sees the slide-door dent (conf ~0.66) — the teacher was the
bottleneck. This rerun uses kimi-k2p6 on Fireworks (reasoning_effort=low),
which judged our validation crops correctly, and samples denser (4 fps).

Run on the training box from ~/caridence:
    FIREWORKS_API_KEY=... .../python build_realdomain_van.py
Writes rd2_van_* images/labels into data/kaggle/cardd_yolo/train.
"""
import base64
import os
import re

import cv2
from openai import OpenAI
from ultralytics import YOLO

NAMES = ["dent", "scratch", "crack"]
VIDEO = "demo_videos/20260626_143914.mp4"  # Toyota Sienna: dent on passenger slide door
I = "data/kaggle/cardd_yolo/train/images"
L = "data/kaggle/cardd_yolo/train/labels"
FPS_SAMPLE = 4          # frames per second to scan (first run used 2)
DET_CONF = 0.30         # candidate threshold (first run used 0.45)
AUTO_KEEP = 0.82        # skip the teacher above this confidence
MODEL = "accounts/fireworks/models/kimi-k2p6"

client = OpenAI(base_url="https://api.fireworks.ai/inference/v1",
                api_key=os.environ["FIREWORKS_API_KEY"])


def verify(crop_bgr, damage: str) -> bool:
    ok, buf = cv2.imencode(".jpg", crop_bgr)
    b64 = base64.b64encode(buf.tobytes()).decode()
    q = (f"Does this close-up show real {damage} damage on the car body or "
         f"bumper? Intact lights, emblems, trim, reflections, shadows, dirt = "
         f"not damage. End your answer with exactly one word: yes or no.")
    try:
        resp = client.chat.completions.create(
            model=MODEL, max_tokens=2048, temperature=0.0,
            extra_body={"reasoning_effort": "low"},
            messages=[{"role": "user", "content": [
                {"type": "text", "text": q},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]}])
        text = (resp.choices[0].message.content or "").lower()
    except Exception as e:
        print(f"  teacher error ({e}); rejecting candidate", flush=True)
        return False  # teacher labels must be precise: reject on failure
    verdicts = re.findall(r"\b(yes|no)\b", text)
    return bool(verdicts) and verdicts[-1] == "yes"


def ahash(img):
    g = cv2.cvtColor(cv2.resize(img, (8, 8)), cv2.COLOR_BGR2GRAY)
    a = g.mean()
    v = 0
    for b in (g > a).flatten():
        v = (v << 1) | int(b)
    return v


def ham(x, y):
    return bin(x ^ y).count("1")


def main():
    assert os.path.isdir(I) and os.path.isdir(L), "dataset dirs missing"
    det = YOLO("runs/detect/outputs/yolo/cardd/weights/best.pt")
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60
    step = max(1, int(fps / FPS_SAMPLE))
    frames, i = [], 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames.append((i // step, img))
        i += 1
    cap.release()
    print(f"scanning {len(frames)} frames", flush=True)

    hashes, imgs_out, boxes_out, asked, kept_by_teacher = [], 0, 0, 0, 0
    for fi, img in frames:
        r = det.predict(img, conf=DET_CONF, iou=0.6, imgsz=1024, augment=True,
                        classes=[0, 1, 2], verbose=False)[0]
        labs = []
        for b in r.boxes:
            conf = float(b.conf)
            x1, y1, x2, y2 = [int(x) for x in b.xyxy[0].tolist()]
            crop = img[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue
            if conf >= AUTO_KEEP:
                keep = True
            else:
                asked += 1
                keep = verify(crop, NAMES[int(b.cls)])
                kept_by_teacher += int(keep)
            if keep:
                cx, cy, w, h = b.xywhn[0].tolist()
                labs.append(f"{int(b.cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        if not labs:
            continue
        h = ahash(img)
        if any(ham(h, p) < 6 for p in hashes):
            continue
        hashes.append(h)
        H, W = img.shape[:2]
        out = cv2.resize(img, (1280, int(H * 1280 / W)))
        fn = f"rd2_van_{fi:04d}"
        assert cv2.imwrite(f"{I}/{fn}.jpg", out)
        with open(f"{L}/{fn}.txt", "w") as f:
            f.write("\n".join(labs))
        imgs_out += 1
        boxes_out += len(labs)
        print(f"  kept frame {fi}: {len(labs)} boxes", flush=True)

    print(f"VAN_DONE imgs={imgs_out} boxes={boxes_out} "
          f"teacher_asked={asked} teacher_kept={kept_by_teacher}", flush=True)


if __name__ == "__main__":
    main()
