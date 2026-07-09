# Caridence on AMD ROCm — detector benchmark

The Caridence damage detector was served and benchmarked on **AMD ROCm**
hardware via the AMD Developer Hackathon notebook environment.

## Environment

| | |
|---|---|
| GPU | AMD Radeon **gfx1100** (RDNA3) |
| Stack | ROCm **7.2**, PyTorch **2.9.1** (ROCm/HIP build), Ultralytics 8.4 |
| Model | `weights/cardd_v3.pt` — YOLOv11-L fine-tuned on CarDD + real-domain data |
| Input | 1080×1920 phone frame, `imgsz=1024`, 40 timed iters after 5 warmup |

## Results

| Mode | Mean | p50 | p90 | Throughput |
|---|---|---|---|---|
| `augment=False` (production single-pass) | **15.1 ms** | 15.1 ms | 15.2 ms | **66 FPS** |
| `augment=True` (high-recall TTA sweep) | **38.5 ms** | 38.6 ms | 38.6 ms | **26 FPS** |

Single-pass detection runs comfortably in real time on AMD hardware; the
high-recall test-time-augmentation mode used for the walkaround sweep still
clears 25 FPS, so a 60-second phone video is analyzed faster than it was shot.

## Reproduce

```bash
git clone https://github.com/dholzric/caridence && cd caridence
/opt/venv/bin/pip install ultralytics          # ROCm torch already present
/opt/venv/bin/python scripts/bench_amd.py weights/cardd_v3.pt
```

## Notes

- The same code path targets AMD Instinct (MI300X) unchanged — `Dockerfile.rocm`
  builds the training/serving image on `rocm/pytorch`; PyTorch exposes the HIP
  backend through the CUDA API, so no source changes are needed across AMD parts.
- This ROCm PyTorch build ships without a HIP-compiled `torchvision::nms`
  kernel, so the final non-max-suppression step is routed to CPU
  (`scripts/bench_amd.py`); the convolutional inference runs on the GPU. NMS is
  a few hundred boxes, so its latency contribution is negligible.
