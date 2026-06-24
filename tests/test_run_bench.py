# tests/test_run_bench.py
import json
from caridence.bench.run_bench import run_benchmark, write_bench_json
from caridence.data.types import LabeledImage
from caridence.schema import Detection, BBox, DamageType, Severity, Frame


def _img(dt, path):
    return LabeledImage(image_path=path, width=100, height=80,
        detections=[Detection(damage_type=dt, panel="left_upper",
            severity=Severity.MINOR, bbox=BBox(x=0.1, y=0.1, w=0.2, h=0.2))])


class PerfectBackend:
    def __init__(self, images): self._by_path = {im.image_path: im.detections for im in images}
    def detect(self, frame: Frame): return list(self._by_path.get(frame.path, []))


class BlindBackend:
    def detect(self, frame: Frame): return []


def test_run_benchmark_scores_backends():
    images = [_img(DamageType.DENT, "a.jpg"), _img(DamageType.SCRATCH, "b.jpg")]
    backends = [
        ("Caridence-7B (MI300)", PerfectBackend(images), lambda n: 0.0007),
        ("GPT-4o", BlindBackend(), lambda n: 0.02),
    ]
    rows = run_benchmark(images, backends)
    perfect = next(r for r in rows if "MI300" in r["model"])
    blind = next(r for r in rows if r["model"] == "GPT-4o")
    assert perfect["f1"] == 1.0
    assert blind["f1"] == 0.0
    assert perfect["cost_per_inspection_usd"] == 0.0007
    assert "latency_ms" in perfect


def test_write_bench_json(tmp_path):
    rows = [{"model": "m", "f1": 1.0, "precision": 1.0, "recall": 1.0,
             "cost_per_inspection_usd": 0.001, "latency_ms": 5}]
    out = tmp_path / "bench.json"
    write_bench_json(rows, out)
    data = json.loads(out.read_text())
    assert data["rows"][0]["model"] == "m"
    assert "generated" in data
