# caridence/analyzer/mock.py
from __future__ import annotations
from caridence.schema import Frame, Detection, DamageType, Severity, BBox

# A canned "four corners" scenario keyed by frame index, so the mock produces
# a realistic multi-finding report deterministically (used for UI/demo dev).
_SCRIPT: dict[int, list[Detection]] = {
    0: [Detection(damage_type=DamageType.SCRATCH, panel="front_driver_corner",
                  severity=Severity.MINOR, bbox=BBox(x=0.1, y=0.5, w=0.2, h=0.15), confidence=0.86)],
    1: [Detection(damage_type=DamageType.SCRATCH, panel="front_driver_corner",
                  severity=Severity.MINOR, bbox=BBox(x=0.12, y=0.52, w=0.2, h=0.15), confidence=0.9)],
    2: [Detection(damage_type=DamageType.DENT, panel="rear_driver_corner",
                  severity=Severity.MODERATE, bbox=BBox(x=0.6, y=0.55, w=0.18, h=0.2), confidence=0.88)],
    3: [Detection(damage_type=DamageType.DENT, panel="rear_passenger_corner",
                  severity=Severity.MINOR, bbox=BBox(x=0.55, y=0.5, w=0.16, h=0.18), confidence=0.81)],
    4: [Detection(damage_type=DamageType.SCRATCH, panel="front_passenger_corner",
                  severity=Severity.MINOR, bbox=BBox(x=0.2, y=0.48, w=0.15, h=0.12), confidence=0.79)],
}


class MockBackend:
    """Deterministic canned backend. No GPU/endpoint required."""

    def detect(self, frame: Frame) -> list[Detection]:
        return [d.model_copy(deep=True) for d in _SCRIPT.get(frame.index, [])]
