"""Local benchmark: fine-tuned Caridence vs base Qwen2.5-VL on the CarDD eval set.

Runs both models via transformers on a single GPU (no vLLM serving needed),
scores detections against ground truth, and writes data/bench.json for the
dashboard. Reuses the same prompt, parser, metrics, and cost model as the rest
of the pipeline so the numbers are apples-to-apples.

Usage (on the GPU box, after training):
    python scripts/bench_local.py --adapter outputs/caridence-3b
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import torch
from transformers import (
    AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig,
)
from peft import PeftModel
from PIL import Image

from caridence.data.eval_set import load_eval_set
from caridence.analyzer.base import parse_detections
from caridence.analyzer.qwen_http import build_prompt
from caridence.metrics import score_predictions
from caridence.bench.run_bench import write_bench_json
from caridence.bench.cost import inspection_cost_selfhosted


def load_model(base: str, adapter: str | None):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base, quantization_config=bnb, torch_dtype=torch.bfloat16, device_map="auto")
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.train(False)  # inference mode
    return model


def run_model(model, processor, eval_images, max_new_tokens: int = 512):
    prompt = build_prompt()
    preds = []
    t0 = time.perf_counter()
    for li in eval_images:
        image = Image.open(li.image_path).convert("RGB")
        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        content = processor.decode(gen, skip_special_tokens=True)
        preds.append(parse_detections(content))
    return preds, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--adapter", default="outputs/caridence-3b")
    ap.add_argument("--eval-set", dest="eval_set", default="data/prepared_test/eval_set.jsonl")
    ap.add_argument("--out", default="data/bench.json")
    ap.add_argument("--limit", type=int, default=0, help="0 = all eval images")
    ap.add_argument("--ft-label", default="Caridence-3B (fine-tuned, AMD-ready)")
    ap.add_argument("--base-label", default="Qwen2.5-VL-3B base (zero-shot)")
    ap.add_argument("--skip-base", action="store_true", help="benchmark only the fine-tuned model")
    args = ap.parse_args()

    eval_images = load_eval_set(args.eval_set)
    if args.limit:
        eval_images = eval_images[: args.limit]
    n = len(eval_images)
    processor = AutoProcessor.from_pretrained(args.base)

    variants = [(args.ft_label, args.adapter)]
    if not args.skip_base:
        variants.append((args.base_label, None))
    rows = []
    for name, adapter in variants:
        print(f"[bench] {name} ...", flush=True)
        model = load_model(args.base, adapter)
        preds, elapsed = run_model(model, processor, eval_images)
        score = score_predictions(eval_images, preds, iou_thr=0.3)
        rows.append({
            "model": name,
            "f1": round(score.f1, 4),
            "precision": round(score.precision, 4),
            "recall": round(score.recall, 4),
            "fp_per_image": round(score.fp_per_image, 4),
            "cost_per_inspection_usd": round(
                inspection_cost_selfhosted(gpu_hourly_usd=2.0, frames_per_sec=5.0, n_frames=10), 6),
            "latency_ms": int(1000 * elapsed / max(1, n)),
        })
        print(f"[bench] {name} -> {rows[-1]}", flush=True)
        del model
        torch.cuda.empty_cache()

    write_bench_json(rows, Path(args.out), generated="3090-local")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
