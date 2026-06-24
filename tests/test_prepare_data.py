# tests/test_prepare_data.py
import json
from pathlib import Path
from train.prepare_data import prepare


def test_prepare_writes_three_files(cardd_mini, tmp_path):
    out = tmp_path / "out"
    counts = prepare(cardd_mini["ann"], cardd_mini["images"], out, val_frac=0.0)
    assert (out / "train.jsonl").exists()
    assert (out / "val.jsonl").exists()
    assert (out / "eval_set.jsonl").exists()
    # val_frac=0 -> all images in train
    train_lines = (out / "train.jsonl").read_text().strip().splitlines()
    assert len(train_lines) == counts["train"] == 1
    ex = json.loads(train_lines[0])
    assert ex["image"].endswith("car1.jpg")
    assert json.loads(ex["response"])  # valid JSON target
