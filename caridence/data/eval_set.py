# caridence/data/eval_set.py
from __future__ import annotations
from pathlib import Path
from caridence.data.types import LabeledImage


def save_eval_set(items: list[LabeledImage], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(it.model_dump_json() + "\n")


def load_eval_set(path: Path) -> list[LabeledImage]:
    out: list[LabeledImage] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(LabeledImage.model_validate_json(line))
    return out
