# caridence/analyzer/base.py
from __future__ import annotations
import json
import re
from typing import Protocol
from caridence.schema import Frame, Detection, FrameDetections, BBox

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class VLMBackend(Protocol):
    def detect(self, frame: Frame) -> list[Detection]:
        ...


def _coerce_bbox(value) -> BBox:
    if isinstance(value, dict):
        return BBox(**value)
    x, y, w, h = value
    return BBox(x=x, y=y, w=w, h=h)


def parse_detections(raw: str) -> list[Detection]:
    """Parse model output into Detections. Never raises — bad output -> []."""
    text = raw.strip()
    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        items = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []
    out: list[Detection] = []
    for it in items:
        try:
            out.append(Detection(
                damage_type=it["damage_type"],
                panel=it["panel"],
                severity=it["severity"],
                bbox=_coerce_bbox(it["bbox"]),
                confidence=float(it.get("confidence", 1.0)),
            ))
        except Exception:
            continue
    return out


def analyze_frames(frames: list[Frame], backend: VLMBackend) -> list[FrameDetections]:
    return [FrameDetections(frame=f, detections=backend.detect(f)) for f in frames]
