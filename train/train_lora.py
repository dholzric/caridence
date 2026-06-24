"""LoRA fine-tune Qwen2.5-VL on Caridence instruction JSONL.

Each JSONL line: {"image": path, "prompt": str, "response": str}.
Prompt tokens are masked; loss is computed only on the response span.
Mirrors the proven QwenSight LoRA recipe (attention projections only).
"""
from __future__ import annotations
import json
import argparse
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor, Qwen2_5_VLForConditionalGeneration,
    TrainingArguments, Trainer,
)
from peft import LoraConfig, get_peft_model
from PIL import Image


class CaridenceDataset(Dataset):
    def __init__(self, jsonl: Path, processor, max_len: int = 1024):
        self.rows = [json.loads(l) for l in Path(jsonl).read_text().splitlines() if l.strip()]
        self.processor = processor
        self.max_len = max_len

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        image = Image.open(row["image"]).convert("RGB")
        user_msg = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": row["prompt"]}]}]
        prompt_text = self.processor.apply_chat_template(user_msg, add_generation_prompt=True, tokenize=False)
        full_text = prompt_text + row["response"]
        proc = self.processor(text=[full_text], images=[image], return_tensors="pt", padding=True)
        prompt_only = self.processor(text=[prompt_text], images=[image], return_tensors="pt", padding=True)
        input_ids = proc["input_ids"][0][: self.max_len]
        labels = input_ids.clone()
        n_prompt = prompt_only["input_ids"].shape[1]
        labels[:n_prompt] = -100  # mask prompt tokens
        item = {k: v[0] for k, v in proc.items()}
        item["input_ids"] = input_ids
        item["labels"] = labels
        return item


def collate(batch):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        if k in ("input_ids", "labels", "attention_mask"):
            out[k] = torch.nn.utils.rnn.pad_sequence(
                [b[k] for b in batch], batch_first=True,
                padding_value=(-100 if k == "labels" else 0))
        else:
            out[k] = torch.stack([b[k] for b in batch]) if batch[0][k].dim() > 0 else torch.tensor([b[k] for b in batch])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--train", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--qlora", action="store_true", help="4-bit base (for 24GB 3090s)")
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(args.model)
    load_kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto")
    if args.qlora:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, **load_kwargs)
    model.gradient_checkpointing_enable()

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = CaridenceDataset(args.train, processor)
    if args.limit:
        ds.rows = ds.rows[: args.limit]

    targs = TrainingArguments(
        output_dir=args.output, per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr,
        max_steps=args.steps, logging_steps=10, save_steps=args.steps,
        bf16=True, report_to=[], remove_unused_columns=False)
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collate)
    trainer.train()
    model.save_pretrained(args.output)
    processor.save_pretrained(args.output)
    print(f"Saved adapter to {args.output}")


if __name__ == "__main__":
    main()
