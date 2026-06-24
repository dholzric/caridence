# tests/test_analyzer_mock.py
from caridence.analyzer import analyze_frames
from caridence.analyzer.mock import MockBackend
from caridence.analyzer.base import parse_detections
from caridence.schema import Frame, DamageType


def test_parse_detections_valid_json():
    raw = '[{"damage_type":"dent","panel":"front_driver_door","severity":"moderate","bbox":[0.1,0.2,0.3,0.4],"confidence":0.9}]'
    dets = parse_detections(raw)
    assert len(dets) == 1
    assert dets[0].damage_type == DamageType.DENT
    assert dets[0].bbox.w == 0.3


def test_parse_detections_tolerates_garbage_and_fences():
    raw = "```json\n[]\n```"
    assert parse_detections(raw) == []
    assert parse_detections("not json at all") == []


def test_mock_backend_is_deterministic():
    b = MockBackend()
    f = Frame(index=0, timestamp=0.0, path="x/img_0.jpg")
    assert b.detect(f) == b.detect(f)


def test_analyze_frames_returns_framedetections():
    b = MockBackend()
    frames = [Frame(index=i, timestamp=float(i), path=f"x/img_{i}.jpg") for i in range(4)]
    results = analyze_frames(frames, b)
    assert len(results) == 4
    assert all(r.frame.index == frames[i].index for i, r in enumerate(results))
