# tests/test_aggregator.py
from caridence.aggregator import aggregate, _severity_rank
from caridence.schema import (
    Frame, Detection, FrameDetections, DamageType, Severity, BBox,
)


def _fd(idx, dets):
    return FrameDetections(frame=Frame(index=idx, timestamp=float(idx), path=f"x/{idx}.jpg"),
                           detections=dets)


def test_severity_rank_orders():
    assert _severity_rank(Severity.SEVERE) > _severity_rank(Severity.MINOR)


def test_aggregate_merges_same_panel_and_type():
    d_low = Detection(damage_type=DamageType.SCRATCH, panel="front_driver_corner",
                      severity=Severity.MINOR, bbox=BBox(x=0.1, y=0.1, w=0.1, h=0.1), confidence=0.6)
    d_high = Detection(damage_type=DamageType.SCRATCH, panel="front_driver_corner",
                       severity=Severity.MODERATE, bbox=BBox(x=0.1, y=0.1, w=0.2, h=0.2), confidence=0.95)
    findings = aggregate([_fd(0, [d_low]), _fd(1, [d_high])])
    assert len(findings) == 1
    f = findings[0]
    assert f.occurrences == 2
    assert f.severity == Severity.MODERATE          # max severity
    assert f.cited_frame.index == 1                  # highest confidence
    assert f.confidence == 0.95


def test_aggregate_separates_distinct_panels():
    a = Detection(damage_type=DamageType.DENT, panel="rear_driver_corner",
                  severity=Severity.MODERATE, bbox=BBox(x=0.1, y=0.1, w=0.1, h=0.1), confidence=0.9)
    b = Detection(damage_type=DamageType.DENT, panel="rear_passenger_corner",
                  severity=Severity.MINOR, bbox=BBox(x=0.5, y=0.1, w=0.1, h=0.1), confidence=0.8)
    findings = aggregate([_fd(0, [a, b])])
    assert len(findings) == 2
    assert {f.panel for f in findings} == {"rear_driver_corner", "rear_passenger_corner"}


def test_aggregate_empty():
    assert aggregate([]) == []
