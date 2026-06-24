# caridence/schema.py
from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field


class DamageType(str, Enum):
    DENT = "dent"
    SCRATCH = "scratch"
    CRACK = "crack"
    GLASS_SHATTER = "glass_shatter"
    LAMP_BROKEN = "lamp_broken"
    TIRE_FLAT = "tire_flat"


class Severity(str, Enum):
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"


class BBox(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    w: float = Field(gt=0.0, le=1.0)
    h: float = Field(gt=0.0, le=1.0)


class Frame(BaseModel):
    index: int
    timestamp: float  # seconds from start
    path: str


class Detection(BaseModel):
    damage_type: DamageType
    panel: str
    severity: Severity
    bbox: BBox
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class FrameDetections(BaseModel):
    frame: Frame
    detections: list[Detection]


class Finding(BaseModel):
    id: str
    damage_type: DamageType
    panel: str
    severity: Severity
    cited_frame: Frame
    bbox: BBox
    confidence: float
    occurrences: int
    cost_low: float | None = None
    cost_high: float | None = None


class InspectionReport(BaseModel):
    vehicle_label: str | None = None
    findings: list[Finding]
    condition_score: int = Field(ge=0, le=100)
    total_cost_low: float
    total_cost_high: float
    frame_count: int
