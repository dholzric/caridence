# tests/test_schema.py
from caridence.schema import (
    DamageType, Severity, BBox, Detection, Frame, FrameDetections,
    Finding, InspectionReport,
)


def test_bbox_rejects_out_of_range():
    import pytest
    with pytest.raises(Exception):
        BBox(x=1.5, y=0.0, w=0.1, h=0.1)


def test_detection_roundtrip():
    d = Detection(
        damage_type=DamageType.DENT, panel="front_driver_door",
        severity=Severity.MODERATE, bbox=BBox(x=0.1, y=0.2, w=0.3, h=0.4),
        confidence=0.9,
    )
    assert d.damage_type == "dent"
    assert Detection.model_validate(d.model_dump()) == d


def test_inspection_report_minimal():
    f = Frame(index=0, timestamp=0.0, path="frames/0.jpg")
    finding = Finding(
        id="f1", damage_type=DamageType.SCRATCH, panel="rear_bumper",
        severity=Severity.MINOR, cited_frame=f, bbox=BBox(x=0, y=0, w=0.1, h=0.1),
        confidence=0.8, occurrences=2, cost_low=80.0, cost_high=150.0,
    )
    rep = InspectionReport(
        vehicle_label="Chevy", findings=[finding], condition_score=82,
        total_cost_low=80.0, total_cost_high=150.0, frame_count=30,
    )
    assert rep.findings[0].panel == "rear_bumper"
    assert rep.total_cost_high == 150.0
