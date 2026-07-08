# Caridence serving image — real detector backend by default.
#
# Build:  docker build -t caridence .
# Run:    docker run --rm -p 8000:8000 caridence
# Then open http://localhost:8000 and upload a walkaround video.
#
# Backends (CARIDENCE_BACKEND): detector (default, CPU-friendly) | hybrid |
# qwen | mock. `hybrid` adds VLM verification of each candidate — point
# CARIDENCE_API_BASE / CARIDENCE_API_KEY / CARIDENCE_VERIFY_MODEL at any
# OpenAI-compatible VLM endpoint (vLLM on ROCm/MI300X, or Fireworks AI):
#   docker run --rm -p 8000:8000 \
#     -e CARIDENCE_BACKEND=hybrid \
#     -e CARIDENCE_API_BASE=https://api.fireworks.ai/inference/v1 \
#     -e CARIDENCE_API_KEY=$FIREWORKS_API_KEY \
#     -e CARIDENCE_VERIFY_MODEL=<vision-model-id> \
#     caridence
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml ./
# CPU torch wheels keep the image lean; the same code runs on ROCm/CUDA hosts.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -e . ultralytics
COPY caridence ./caridence
COPY app ./app
COPY data ./data
COPY weights ./weights
ENV CARIDENCE_BACKEND=detector \
    CARIDENCE_DETECTOR_WEIGHTS=/app/weights/cardd_v3.pt \
    CARIDENCE_PLATE_WEIGHTS=/app/weights/plates_best.pt \
    CARIDENCE_REDACT=1
EXPOSE 8000
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
