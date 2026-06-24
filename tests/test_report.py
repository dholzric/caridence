# tests/test_report.py
from caridence.report import render_report_html, report_to_dict
from caridence.estimator import build_report
from caridence.schema import Finding, Frame, BBox, DamageType, Severity


def _report():
    f = Finding(
        id="rear_driver__dent", damage_type=DamageType.DENT, panel="rear_driver_corner",
        severity=Severity.MODERATE, cited_frame=Frame(index=2, timestamp=2.0, path="frames/2.jpg"),
        bbox=BBox(x=0.6, y=0.55, w=0.18, h=0.2), confidence=0.88, occurrences=3,
        cost_low=270.0, cost_high=1080.0,
    )
    return build_report([f], frame_count=30, vehicle_label="Chevy")


def test_report_to_dict_roundtrips():
    d = report_to_dict(_report())
    assert d["vehicle_label"] == "Chevy"
    assert d["findings"][0]["panel"] == "rear_driver_corner"


def test_render_html_contains_key_facts():
    html = render_report_html(_report())
    assert "Chevy" in html
    assert "rear_driver_corner" in html or "rear driver corner" in html.lower()
    assert "Condition" in html
    assert "270" in html  # cost appears
    assert "<html" in html.lower()
