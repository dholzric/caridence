# tests/test_estimator.py
from caridence.costs import COST_TABLE, lookup_cost
from caridence.estimator import estimate, condition_score, build_report
from caridence.schema import (
    Finding, Frame, BBox, DamageType, Severity, InspectionReport,
)


def _finding(dtype, sev, conf=0.9, idx=0):
    return Finding(
        id=f"{dtype.value}", damage_type=dtype, panel="p", severity=sev,
        cited_frame=Frame(index=idx, timestamp=float(idx), path=f"{idx}.jpg"),
        bbox=BBox(x=0.1, y=0.1, w=0.1, h=0.1), confidence=conf, occurrences=1,
    )


def test_cost_table_covers_all_combinations():
    for dt in DamageType:
        for sv in Severity:
            assert (dt, sv) in COST_TABLE


def test_lookup_cost_monotonic_in_severity():
    minor = lookup_cost(DamageType.DENT, Severity.MINOR)
    severe = lookup_cost(DamageType.DENT, Severity.SEVERE)
    assert severe[0] > minor[0] and severe[1] > minor[1]


def test_estimate_attaches_costs():
    findings = estimate([_finding(DamageType.SCRATCH, Severity.MINOR)])
    assert findings[0].cost_low is not None
    assert findings[0].cost_high >= findings[0].cost_low


def test_condition_score_decreases_with_damage():
    clean = condition_score([])
    damaged = condition_score([_finding(DamageType.DENT, Severity.SEVERE)])
    assert clean == 100
    assert damaged < clean


def test_build_report_totals():
    findings = estimate([
        _finding(DamageType.SCRATCH, Severity.MINOR),
        _finding(DamageType.DENT, Severity.MODERATE, idx=2),
    ])
    rep = build_report(findings, frame_count=30, vehicle_label="Chevy")
    assert isinstance(rep, InspectionReport)
    assert rep.total_cost_low == sum(f.cost_low for f in findings)
    assert rep.total_cost_high == sum(f.cost_high for f in findings)
    assert rep.frame_count == 30
    assert 0 <= rep.condition_score <= 100
