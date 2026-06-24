# caridence/data/severity.py
from __future__ import annotations
from caridence.schema import Severity

# Damage-area ratio thresholds (bbox area / image area). Estimate only.
MINOR_MAX = 0.02
MODERATE_MAX = 0.06


def severity_from_area(area_ratio: float) -> Severity:
    if area_ratio < MINOR_MAX:
        return Severity.MINOR
    if area_ratio < MODERATE_MAX:
        return Severity.MODERATE
    return Severity.SEVERE
