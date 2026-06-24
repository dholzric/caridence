# tests/test_severity.py
from caridence.data.severity import severity_from_area
from caridence.schema import Severity


def test_small_is_minor():
    assert severity_from_area(0.005) == Severity.MINOR


def test_medium_is_moderate():
    assert severity_from_area(0.04) == Severity.MODERATE


def test_large_is_severe():
    assert severity_from_area(0.2) == Severity.SEVERE


def test_monotonic_boundaries():
    order = [severity_from_area(a) for a in (0.001, 0.03, 0.5)]
    assert order == [Severity.MINOR, Severity.MODERATE, Severity.SEVERE]
