# Caridence

**Phone walkaround → cited vehicle-damage report in 60 seconds.**

Caridence turns a short phone walkaround of any vehicle into an objective,
structured condition report where **every damage finding cites the exact
timestamped frame it was seen in** — graded by severity with a repair-cost
estimate. It produces shareable, defensible proof of condition for rental
returns, peer-to-peer car sharing, dealer trade-ins, and insurance
first-notice-of-loss.

Built for the **AMD Developer Hackathon: ACT II** (Track 3 — Unicorn).

## Why it's interesting

The engine is a **fine-tuned YOLOv11 damage detector** (high recall) paired with
a **fine-tuned Qwen2.5-VL** verifier (precision) — a hybrid that finds damage and
confirms each finding, then auto-redacts license plates. A built-in benchmark
dashboard reports measured numbers (97.8% presence recall on the CarDD test set),
not marketing.

**AMD:** all models are open and run on **ROCm** — the training and serving code
is AMD-native (MI300X / Radeon); see [`docs/training-on-amd-rocm.md`](docs/training-on-amd-rocm.md)
and [`Dockerfile.rocm`](Dockerfile.rocm). Natural-language inspection reports run
on AMD-hardware models via the **Fireworks AI** API. (This build was trained on
local GPUs; the code is portable to AMD unchanged.)

- **No fragile video model.** Ingestion is `ffmpeg`/OpenCV frame sampling +
  per-frame analysis, then aggregation. Robust, and every finding maps to a
  citable frame.
- **Video is optional.** Accepts a walkaround video *or* a folder of photos.
- **Privacy by default.** License plates (and faces) are detected and redacted
  before anything reaches the report.
- **Swappable backend.** `mock` (no GPU), `detector`/`hybrid`, or `qwen` against
  any vLLM OpenAI-compatible endpoint (NVIDIA or AMD MI300) — one env var.

## Architecture

```
video/photos
  → ingest      (timestamped, blur-filtered, deduped frames)
  → analyzer    (per-frame VLM detections: damage_type, panel, severity, bbox)
  → aggregator  (unique findings + clearest "cited frame" + panel)
  → estimator   (severity + repair-cost range + condition score)
  → report      (JSON + HTML with cited-frame bbox overlay)
  → web         (FastAPI upload UI + benchmark dashboard)
```

Training & benchmark (the moat): `CarDD → enrich (panel/severity) → instruction
JSONL → LoRA fine-tune Qwen2.5-VL → merge → serve on MI300 → benchmark vs
frontier → data/bench.json` (rendered by the dashboard).

## Quickstart (mock backend, no GPU)

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"
.venv/Scripts/pytest                                   # 60 tests
.venv/Scripts/uvicorn app.server:app --port 8000       # open http://127.0.0.1:8000/
```

Upload a folder of car photos (or a walkaround video) and get a cited damage
report. `/dashboard` renders the benchmark numbers from `data/bench.json`.

### Docker (recommended — real models, no GPU required)

```bash
docker build -t caridence .
docker run --rm -p 8000:8000 caridence
```

The image ships the fine-tuned CarDD damage detector and the license-plate
redactor (`weights/`) and serves the `detector` backend on CPU — upload a
walkaround video at http://localhost:8000 and get a cited damage report.

To enable hybrid mode (detector recall + VLM verification of every candidate),
point the verifier at any OpenAI-compatible vision endpoint — vLLM on
ROCm/MI300X, or Fireworks AI:

```bash
docker run --rm -p 8000:8000 \
  -e CARIDENCE_BACKEND=hybrid \
  -e CARIDENCE_API_BASE=https://api.fireworks.ai/inference/v1 \
  -e CARIDENCE_API_KEY=$FIREWORKS_API_KEY \
  -e CARIDENCE_VERIFY_MODEL=accounts/fireworks/models/kimi-k2p6 \
  -e CARIDENCE_VERIFY_REASONING_EFFORT=low \
  -e CARIDENCE_VERIFY_MAX_TOKENS=2048 \
  caridence
```

`CARIDENCE_VERIFY_REASONING_EFFORT=low` keeps reasoning models decisive;
the verifier reads the final yes/no of the reply, so plain VLMs
(e.g. our fine-tuned Qwen served by vLLM) work with the same flags.

### Real model backend (bare metal)

```bash
export CARIDENCE_BACKEND=qwen
export CARIDENCE_API_BASE=http://<vllm-host>:8000/v1
export CARIDENCE_MODEL=caridence-7b
```

## Training & benchmark

```bash
# 1. Prepare data from a CarDD download
python train/prepare_data.py --ann data/cardd/annotations/train.json \
    --images data/cardd/images/train --out data/prepared --val-frac 0.1

# 2. LoRA fine-tune (QLoRA on 24GB GPUs; full precision on MI300)
python train/train_lora.py --train data/prepared/train.jsonl \
    --output outputs/caridence-7b --steps 600         # add --qlora on 3090s
python train/merge_adapter.py --adapter outputs/caridence-7b \
    --out outputs/caridence-7b-merged

# 3. Serve + benchmark → fills data/bench.json
bash scripts/serve_vllm.sh outputs/caridence-7b-merged 8000
python -m caridence.bench.run_bench
```

## Project layout

| Path | Responsibility |
|---|---|
| `caridence/ingest.py` | video/photos → frames |
| `caridence/analyzer/` | VLM backends (`mock`, `qwen_http`) + parser |
| `caridence/aggregator.py` | detections → unique cited findings |
| `caridence/estimator.py`, `costs.py` | severity + repair-cost + condition score |
| `caridence/report.py` | JSON + HTML report |
| `caridence/pipeline.py` | end-to-end orchestration |
| `app/server.py` | FastAPI upload UI + dashboard |
| `caridence/data/`, `caridence/metrics.py`, `caridence/bench/` | training data + eval + benchmark |
| `train/`, `scripts/` | fine-tune, merge, serve |

## License

MIT.
