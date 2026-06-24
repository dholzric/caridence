# caridence/estimator.py
from __future__ import annotations
from caridence.schema import Finding, Severity, InspectionReport
from caridence.costs import lookup_cost

_PENALTY = {Severity.MINOR: 4, Severity.MODERATE: 10, Severity.SEVERE: 20}


def estimate(findings: list[Finding]) -> list[Finding]:
    out: list[Finding] = []
    for f in findings:
        low, high = lookup_cost(f.damage_type, f.severity)
        out.append(f.model_copy(update={"cost_low": low, "cost_high": high}))
    return out


def condition_score(findings: list[Finding]) -> int:
    score = 100 - sum(_PENALTY[f.severity] for f in findings)
    return max(0, min(100, score))


def build_report(findings: list[Finding], frame_count: int,
                 vehicle_label: str | None = None) -> InspectionReport:
    priced = [f for f in findings if f.cost_low is not None]
    return InspectionReport(
        vehicle_label=vehicle_label,
        findings=findings,
        condition_score=condition_score(findings),
        total_cost_low=round(sum(f.cost_low for f in priced), 2),
        total_cost_high=round(sum(f.cost_high for f in priced), 2),
        frame_count=frame_count,
    )
