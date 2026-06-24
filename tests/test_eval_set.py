# tests/test_eval_set.py
from caridence.data.eval_set import save_eval_set, load_eval_set
from caridence.data.types import LabeledImage
from caridence.schema import Detection, BBox, DamageType, Severity


def _labeled():
    return LabeledImage(
        image_path="x/car1.jpg", width=100, height=80,
        detections=[Detection(damage_type=DamageType.DENT, panel="left_upper",
                              severity=Severity.SEVERE, bbox=BBox(x=0.1, y=0.1, w=0.3, h=0.3))],
    )


def test_save_and_load_roundtrip(tmp_path):
    p = tmp_path / "eval.jsonl"
    save_eval_set([_labeled(), _labeled()], p)
    loaded = load_eval_set(p)
    assert len(loaded) == 2
    assert loaded[0].detections[0].damage_type == DamageType.DENT
    assert loaded[0] == _labeled()
