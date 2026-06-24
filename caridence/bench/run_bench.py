# caridence/bench/run_bench.py
from __future__ import annotations
import time
import json
import os
from pathlib import Path
from caridence.schema import Frame, Detection
from caridence.data.types import LabeledImage
from caridence.data.eval_set import load_eval_set
from caridence.metrics import score_predictions


def run_benchmark(eval_images: list[LabeledImage], backends, iou_thr: float = 0.3, frames_per_inspection: int = 10) -> list[dict]:
    """backends: list of (name, backend, cost_fn). cost_fn(n_frames)->usd."""
    rows: list[dict] = []
    for name, backend, cost_fn in backends:
        preds: list[list[Detection]] = []
        t0 = time.perf_counter()
        for img in eval_images:
            frame = Frame(index=0, timestamp=0.0, path=img.image_path)
            preds.append(backend.detect(frame))
        elapsed = time.perf_counter() - t0
        score = score_predictions(eval_images, preds, iou_thr=iou_thr)
        n = len(eval_images) or 1
        rows.append({
            "model": name,
            "f1": round(score.f1, 4),
            "precision": round(score.precision, 4),
            "recall": round(score.recall, 4),
            "fp_per_image": round(score.fp_per_image, 4),
            "cost_per_inspection_usd": round(cost_fn(frames_per_inspection), 6),
            "latency_ms": int(1000 * elapsed / n),
        })
    return rows


def write_bench_json(rows: list[dict], path: Path, generated: str = "local") -> None:
    Path(path).write_text(json.dumps({"generated": generated, "rows": rows}, indent=2))


def main() -> None:  # pragma: no cover - wiring only, exercised manually on MI300
    from caridence.analyzer.qwen_http import QwenHTTPBackend
    from caridence.bench.cost import inspection_cost_api, inspection_cost_selfhosted

    eval_path = Path(os.environ.get("CARIDENCE_EVAL", "data/eval_set.jsonl"))
    images = load_eval_set(eval_path)
    ft = QwenHTTPBackend(
        api_base=os.environ.get("CARIDENCE_FT_API_BASE", "http://127.0.0.1:8000/v1"),
        model=os.environ.get("CARIDENCE_FT_MODEL", "caridence-7b"),
    )
    gpt = QwenHTTPBackend(
        api_base="https://api.openai.com/v1",
        api_key=os.environ["OPENAI_API_KEY"],
        model="gpt-4o",
    )
    backends = [
        ("Caridence-7B (MI300)", ft,
         lambda n: inspection_cost_selfhosted(gpu_hourly_usd=2.0, frames_per_sec=5.0, n_frames=n)),
        ("GPT-4o", gpt,
         lambda n: inspection_cost_api("gpt-4o", n, in_tokens_per_frame=1200, out_tokens_per_frame=120)),
    ]
    rows = run_benchmark(images, backends)
    write_bench_json(rows, Path("data/bench.json"), generated="mi300")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
