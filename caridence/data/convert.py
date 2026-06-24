# caridence/data/convert.py
from __future__ import annotations
import json
from caridence.schema import Detection, BBox
from caridence.data.types import CarddImage, LabeledImage
from caridence.data.panels import infer_panel
from caridence.data.severity import severity_from_area
from caridence.analyzer.qwen_http import build_prompt


def enrich(img: CarddImage) -> LabeledImage:
    dets: list[Detection] = []
    for b in img.boxes:
        area = b.bbox.w * b.bbox.h
        dets.append(Detection(
            damage_type=b.damage_type,
            panel=infer_panel(b.bbox),
            severity=severity_from_area(area),
            bbox=b.bbox,
            confidence=1.0,
        ))
    return LabeledImage(image_path=img.image_path, width=img.width,
                        height=img.height, detections=dets)


def to_target_json(detections: list[Detection]) -> str:
    arr = [{
        "damage_type": d.damage_type.value,
        "panel": d.panel,
        "severity": d.severity.value,
        "bbox": [round(d.bbox.x, 4), round(d.bbox.y, 4), round(d.bbox.w, 4), round(d.bbox.h, 4)],
    } for d in detections]
    return json.dumps(arr, separators=(",", ":"))


def to_instruction_example(labeled: LabeledImage) -> dict:
    return {
        "image": labeled.image_path,
        "prompt": build_prompt(),
        "response": to_target_json(labeled.detections),
    }
