#!/usr/bin/env bash
# Serve the merged Caridence model via vLLM (OpenAI-compatible).
# Local 3090s: smaller model / lower mem. MI300: full model, high throughput.
set -euo pipefail
MODEL_PATH="${1:-outputs/caridence-7b-merged}"
PORT="${2:-8000}"
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name caridence-7b \
  --port "$PORT" \
  --max-model-len 4096 \
  --limit-mm-per-prompt image=1
