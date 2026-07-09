"""Benchmark the Caridence detector on AMD ROCm hardware.

Measures single-frame inference latency and throughput for the fine-tuned
cardd_v3 YOLO detector at our serving config (imgsz=1024). Reports both
augment=True (test-time augmentation, our high-recall mode) and augment=False.

Run on an AMD ROCm box:
    /opt/venv/bin/python scripts/bench_amd.py weights/cardd_v3.pt

Prints a compact, quotable summary: GPU name, ROCm/torch versions, mean/p50
latency, and frames-per-second. Input is a synthetic phone-resolution frame,
so numbers reflect compute cost, not disk/codec.
"""
import statistics
import sys
import time

import numpy as np
import torch
from ultralytics import YOLO

WEIGHTS = sys.argv[1] if len(sys.argv) > 1 else "weights/cardd_v3.pt"
IMGSZ = 1024
WARMUP = 5
ITERS = 40
# Portrait phone frame (what a walkaround actually produces).
FRAME = np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)


def bench(model, augment):
    for _ in range(WARMUP):
        model.predict(FRAME, imgsz=IMGSZ, augment=augment, conf=0.4,
                      classes=[0, 1, 2], verbose=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        model.predict(FRAME, imgsz=IMGSZ, augment=augment, conf=0.4,
                      classes=[0, 1, 2], verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def summarize(label, times):
    mean = statistics.mean(times)
    p50 = statistics.median(times)
    p90 = sorted(times)[int(0.9 * len(times)) - 1]
    print(f"  {label:16s} mean={mean:7.1f}ms  p50={p50:7.1f}ms  "
          f"p90={p90:7.1f}ms  fps={1000.0/mean:5.1f}")


def main():
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("=" * 64)
    print("Caridence detector benchmark — AMD ROCm")
    print(f"  device : {dev}")
    print(f"  torch  : {torch.__version__}  hip={torch.version.hip}")
    print(f"  weights: {WEIGHTS}  imgsz={IMGSZ}  iters={ITERS}")
    print("=" * 64)
    model = YOLO(WEIGHTS)
    summarize("augment=False", bench(model, augment=False))
    summarize("augment=True", bench(model, augment=True))
    print("=" * 64)
    print("augment=False is production single-pass; augment=True is the "
          "high-recall TTA mode used for the walkaround sweep.")


if __name__ == "__main__":
    main()
