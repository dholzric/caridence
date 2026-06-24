# caridence/data/cardd.py
from __future__ import annotations
import json
from pathlib import Path
from caridence.schema import DamageType, BBox
from caridence.data.types import CarddImage, CarddBox

_NAME_MAP = {
    "dent": DamageType.DENT,
    "scratch": DamageType.SCRATCH,
    "crack": DamageType.CRACK,
    "glass shatter": DamageType.GLASS_SHATTER,
    "glass_shatter": DamageType.GLASS_SHATTER,
    "lamp broken": DamageType.LAMP_BROKEN,
    "lamp_broken": DamageType.LAMP_BROKEN,
    "tire flat": DamageType.TIRE_FLAT,
    "tire_flat": DamageType.TIRE_FLAT,
}


def category_to_damage_type(name: str) -> DamageType:
    key = name.strip().lower()
    if key not in _NAME_MAP:
        raise KeyError(f"Unknown CarDD category: {name!r}")
    return _NAME_MAP[key]


def parse_cardd_coco(ann_path: Path, images_dir: Path) -> list[CarddImage]:
    ann_path, images_dir = Path(ann_path), Path(images_dir)
    coco = json.loads(ann_path.read_text())
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    imgs = {i["id"]: i for i in coco["images"]}
    by_image: dict[int, list[CarddBox]] = {iid: [] for iid in imgs}
    for a in coco["annotations"]:
        meta = imgs[a["image_id"]]
        W, H = meta["width"], meta["height"]
        x, y, w, h = a["bbox"]
        try:
            dt = category_to_damage_type(cats[a["category_id"]])
        except KeyError:
            continue
        bbox = BBox(
            x=max(0.0, min(1.0, x / W)),
            y=max(0.0, min(1.0, y / H)),
            w=max(1e-6, min(1.0, w / W)),
            h=max(1e-6, min(1.0, h / H)),
        )
        by_image[a["image_id"]].append(CarddBox(damage_type=dt, bbox=bbox))
    out: list[CarddImage] = []
    for iid, meta in imgs.items():
        out.append(CarddImage(
            image_path=str(images_dir / meta["file_name"]),
            width=meta["width"], height=meta["height"],
            boxes=by_image[iid],
        ))
    return out
