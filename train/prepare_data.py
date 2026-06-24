# train/prepare_data.py
from __future__ import annotations
import json
import argparse
import hashlib
from pathlib import Path
from caridence.data.cardd import parse_cardd_coco
from caridence.data.convert import enrich, to_instruction_example
from caridence.data.eval_set import save_eval_set


def _is_val(image_path: str, val_frac: float) -> bool:
    if val_frac <= 0:
        return False
    h = int(hashlib.sha1(image_path.encode()).hexdigest(), 16) % 1000
    return h < int(val_frac * 1000)


def prepare(cardd_ann: Path, images_dir: Path, out_dir: Path, val_frac: float = 0.1) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cardd_images = parse_cardd_coco(cardd_ann, images_dir)
    labeled = [enrich(ci) for ci in cardd_images]

    train, val = [], []
    for lab in labeled:
        (val if _is_val(lab.image_path, val_frac) else train).append(lab)

    with (out_dir / "train.jsonl").open("w", encoding="utf-8") as fh:
        for lab in train:
            fh.write(json.dumps(to_instruction_example(lab)) + "\n")
    with (out_dir / "val.jsonl").open("w", encoding="utf-8") as fh:
        for lab in val:
            fh.write(json.dumps(to_instruction_example(lab)) + "\n")
    # eval set is the val split's GT (or train if no val) for benchmarking
    save_eval_set(val or train, out_dir / "eval_set.jsonl")
    return {"train": len(train), "val": len(val)}


def main() -> None:  # pragma: no cover - CLI wiring
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()
    counts = prepare(Path(args.ann), Path(args.images), Path(args.out), args.val_frac)
    print(counts)


if __name__ == "__main__":  # pragma: no cover
    main()
