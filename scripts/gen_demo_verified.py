"""Generate honest hybrid demo stills: detector proposes, VLM verifies.

Raw cardd_v3 detections include false positives (badges, handles, reflections).
The product filters them with a VLM verifier. This renders, for the van and
chevy walkarounds:
  * verified-only annotated frames (what the report shows), and
  * for a couple of frames, a candidates-vs-verified pair (yellow = all
    detector boxes, red = VLM-confirmed) — the precision beat for scene 3.

Run: FIREWORKS_API_KEY=... .venv/Scripts/python scripts/gen_demo_verified.py
"""
import base64
import os
import re
from pathlib import Path

import cv2
from openai import OpenAI
from ultralytics import YOLO

OUT = Path(os.environ.get("DEMO_OUT", "demo_frames_v"))
for sub in ("van", "chevy", "compare"):
    (OUT / sub).mkdir(parents=True, exist_ok=True)

NAMES = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"]
RED = (60, 60, 235)
YELLOW = (40, 200, 240)
MODEL = "accounts/fireworks/models/kimi-k2p6"
MAX_PER = 6  # verified frames to keep per vehicle

det = YOLO("weights/cardd_v3.pt")
client = OpenAI(base_url="https://api.fireworks.ai/inference/v1",
                api_key=os.environ["FIREWORKS_API_KEY"])

TARGETS = {"van": "demo_videos/20260626_143914.mp4",
           "chevy": "demo_videos/20260626_143759.mp4"}


def verify(crop_bgr, damage):
    ok, buf = cv2.imencode(".jpg", crop_bgr)
    b64 = base64.b64encode(buf.tobytes()).decode()
    q = (f"Does this close-up show real {damage} damage on the car body or "
         f"bumper? Intact lights, badges, emblems, door handles, trim, "
         f"reflections, shadows and dirt are NOT damage. End with one word: "
         f"yes or no.")
    try:
        r = client.chat.completions.create(
            model=MODEL, max_tokens=2048, temperature=0.0,
            extra_body={"reasoning_effort": "low"},
            messages=[{"role": "user", "content": [
                {"type": "text", "text": q},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}])
        v = re.findall(r"\b(yes|no)\b", (r.choices[0].message.content or "").lower())
        return bool(v) and v[-1] == "yes"
    except Exception as e:
        print("  verify error:", e)
        return False


def severity(a):
    return "severe" if a > 0.04 else "moderate" if a > 0.012 else "minor"


def draw(img, cls, conf, box, color):
    x1, y1, x2, y2 = box
    H, W = img.shape[:2]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
    lab = f"{NAMES[cls]}  {severity((x2-x1)*(y2-y1)/(W*H))}  {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
    cv2.rectangle(img, (x1, y1-th-14), (x1+tw+12, y1), color, -1)
    cv2.putText(img, lab, (x1+6, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 3)


def frames(video, fps_s):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60
    step = max(1, int(fps/fps_s))
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
        made_compare = False
        for fi, img in frames(video, 1):
            r = det.predict(img, conf=0.4, iou=0.6, imgsz=1024, augment=True,
                            classes=[0, 1, 2], verbose=False)[0]
            cands = []
            for b in r.boxes:
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                cands.append((int(b.cls), float(b.conf), (x1, y1, x2, y2)))
            if not cands:
                continue
            verified = []
            for cls, conf, (x1, y1, x2, y2) in cands:
                crop = img[max(0, y1):y2, max(0, x1):x2]
                if crop.size and (conf >= 0.85 or verify(crop, NAMES[cls])):
                    verified.append((cls, conf, (x1, y1, x2, y2)))
            # Candidates-vs-verified compare: a frame where verify removed >=1 box.
            if not made_compare and len(verified) < len(cands) and verified:
                ca = img.copy()
                for c in cands:
                    draw(ca, *c, YELLOW)
                ve = img.copy()
                for c in verified:
                    draw(ve, *c, RED)
                H, W = img.shape[:2]
                sz = (1080, int(H*1080/W))
                cv2.imwrite(str(OUT/"compare"/f"{tag}_candidates.jpg"), cv2.resize(ca, sz))
                cv2.imwrite(str(OUT/"compare"/f"{tag}_verified.jpg"), cv2.resize(ve, sz))
                made_compare = True
                print(f"{tag}: compare frame {fi} ({len(cands)}->{len(verified)})", flush=True)
            if not verified:
                continue
            out = img.copy()
            for c in verified:
                draw(out, *c, RED)
            H, W = out.shape[:2]
            cv2.imwrite(str(OUT/tag/f"{tag}_{fi:04d}.jpg"),
                        cv2.resize(out, (1080, int(H*1080/W))))
            kept += 1
            if kept >= MAX_PER:
                break
        print(f"{tag}: {kept} verified frames", flush=True)


if __name__ == "__main__":
    main()
