# caridence/costs.py
from __future__ import annotations
from caridence.schema import DamageType, Severity

# Rough US body-shop ranges (USD). Estimates only — surfaced as ranges in the UI.
_BASE: dict[DamageType, tuple[float, float]] = {
    DamageType.SCRATCH: (80.0, 200.0),
    DamageType.DENT: (150.0, 600.0),
    DamageType.CRACK: (120.0, 400.0),
    DamageType.GLASS_SHATTER: (250.0, 700.0),
    DamageType.LAMP_BROKEN: (90.0, 450.0),
    DamageType.TIRE_FLAT: (20.0, 250.0),
}
_MULT: dict[Severity, float] = {
    Severity.MINOR: 1.0,
    Severity.MODERATE: 1.8,
    Severity.SEVERE: 3.0,
}

COST_TABLE: dict[tuple[DamageType, Severity], tuple[float, float]] = {
    (dt, sv): (round(lo * _MULT[sv], 2), round(hi * _MULT[sv], 2))
    for dt, (lo, hi) in _BASE.items()
    for sv in Severity
}


def lookup_cost(damage_type: DamageType, severity: Severity) -> tuple[float, float]:
    return COST_TABLE[(damage_type, severity)]
