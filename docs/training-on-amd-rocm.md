# Training & serving Caridence on AMD GPUs (ROCm)

Caridence is built on open models (YOLOv11, Qwen2.5-VL) and standard PyTorch, so
**the same training and serving code runs on AMD GPUs via ROCm** — MI300X in the
AMD Developer Cloud, or Radeon workstation cards locally. PyTorch's ROCm build
exposes AMD's HIP backend through the familiar `torch.cuda` API, so scripts that
call `.cuda()`, `device_map="auto"`, etc. work unchanged.

> For this hackathon build we trained on local NVIDIA GPUs (the benchmark numbers
> in `data/bench.json` come from that hardware). This document is the AMD path:
> everything here is ROCm-native and has no NVIDIA dependency.

## 1. Environment

### Option A — Docker (recommended)

```bash
docker build -f Dockerfile.rocm -t caridence-rocm .

# verify the GPU is visible inside the container
docker run --rm -it --device=/dev/kfd --device=/dev/dri --group-add video \
  --security-opt seccomp=unconfined caridence-rocm
# -> "ROCm/HIP available: True"  +  the AMD device name (e.g. AMD Instinct MI300X)
```

### Option B — pip on a ROCm host

```bash
python -m venv .venv && source .venv/bin/activate
# ROCm build of PyTorch (match your ROCm version; 6.2 shown)
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
pip install -e . "transformers>=4.49" peft accelerate qwen-vl-utils pillow ultralytics
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Device notes

- **MI300X** is `gfx942` and is supported out of the box by recent ROCm PyTorch.
- **Radeon** consumer cards may need a gfx override, e.g. for RDNA3:
  `export HSA_OVERRIDE_GFX_VERSION=11.0.0`
- Select GPUs with `HIP_VISIBLE_DEVICES=0,1,...` (the ROCm analog of
  `CUDA_VISIBLE_DEVICES`; the scripts also accept `CUDA_VISIBLE_DEVICES`, which
  ROCm honors).

## 2. Detector training (YOLOv11) on ROCm

Ultralytics runs on the ROCm PyTorch build with no changes:

```bash
yolo detect train model=yolo11l.pt data=data/kaggle/cardd_yolo/data.yaml \
  epochs=100 imgsz=1024 batch=16 device=0
# (scripts/train_yolo.sh wraps this)
```

## 3. VLM LoRA fine-tune (Qwen2.5-VL) on ROCm

`train/train_lora.py` is device-agnostic. For full-precision / bf16 fine-tuning
on MI300X (192 GB) you can drop the `--qlora` flag entirely:

```bash
python train/train_lora.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --train data/prepared/train.jsonl \
  --output outputs/caridence-7b --steps 600
```

**QLoRA (4-bit) on ROCm:** the `--qlora` path uses `bitsandbytes`. Use the
multi-backend ROCm build of bitsandbytes (gfx942) per AMD's current install
guide; on MI300X's large memory, bf16 LoRA is usually preferable to 4-bit anyway,
so `--qlora` is optional on AMD.

## 4. Serving on MI300X (vLLM-ROCm)

The fine-tuned/merged checkpoint serves through vLLM's ROCm build behind the same
OpenAI-compatible API the app already speaks (`CARIDENCE_BACKEND=qwen`,
`CARIDENCE_API_BASE=...`). `scripts/serve_vllm.sh` is unchanged; run it inside a
`rocm/vllm` container:

```bash
python merge_adapter.py --adapter outputs/caridence-7b --out outputs/caridence-7b-merged
bash scripts/serve_vllm.sh outputs/caridence-7b-merged 8000
```

## 5. Why this is portable by design

Nothing in the pipeline is NVIDIA-specific:

| Stage | Library | AMD support |
|---|---|---|
| Detector train/infer | ultralytics + PyTorch | ROCm PyTorch |
| VLM LoRA fine-tune | transformers + peft | ROCm PyTorch (+ bitsandbytes-ROCm for 4-bit) |
| Serving | vLLM | vLLM-ROCm |
| Inference app | FastAPI + OpenCV | CPU / any GPU |

Point the scripts at an AMD device and they run. The Fireworks AI integration
additionally executes inference on AMD-hardware-hosted models through the
Fireworks API.
