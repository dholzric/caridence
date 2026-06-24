# caridence/data/types.py
from __future__ import annotations
from pydantic import BaseModel
from caridence.schema import DamageType, BBox, Detection


class CarddBox(BaseModel):
    damage_type: DamageType
    bbox: BBox  # normalized 0..1


class CarddImage(BaseModel):
    image_path: str
    width: int
    height: int
    boxes: list[CarddBox]


class LabeledImage(BaseModel):
    image_path: str
    width: int
    height: int
    detections: list[Detection]  # GT detections (panel + severity filled in)
