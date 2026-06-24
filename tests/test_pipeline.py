# tests/test_pipeline.py
from caridence.pipeline import run_inspection
from caridence.analyzer.mock import MockBackend
from caridence.schema import InspectionReport


def test_run_inspection_end_to_end_photos(frames_dir):
    report = run_inspection(frames_dir, backend=MockBackend(), vehicle_label="TestCar", fps=2.0)
    assert isinstance(report, InspectionReport)
    assert report.frame_count >= 1
    # Mock script emits damage on low indices -> at least one finding expected
    assert len(report.findings) >= 1
    assert all(f.cost_low is not None for f in report.findings)


def test_run_inspection_handles_clean_vehicle(frames_dir):
    class CleanBackend:
        def detect(self, frame):
            return []
    report = run_inspection(frames_dir, backend=CleanBackend())
    assert report.condition_score == 100
    assert report.findings == []
    assert report.total_cost_low == 0
