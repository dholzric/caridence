# tests/test_convert.py
import json
from caridence.data.convert import enrich, to_target_json, to_instruction_example
from caridence.data.types import CarddImage, CarddBox, LabeledImage
from caridence.schema import DamageType, BBox, Severity


def _cardd_img():
    return CarddImage(
        image_path="x/car1.jpg", width=100, height=80,
        boxes=[
            CarddBox(damage_type=DamageType.DENT, bbox=BBox(x=0.05, y=0.1, w=0.3, h=0.3)),
            CarddBox(damage_type=DamageType.SCRATCH, bbox=BBox(x=0.8, y=0.7, w=0.05, h=0.05)),
        ],
    )


def test_enrich_fills_panel_and_severity():
    labeled = enrich(_cardd_img())
    assert isinstance(labeled, LabeledImage)
    dent = labeled.detections[0]
    assert dent.panel == "left_upper"
    assert dent.severity == Severity.SEVERE        # area 0.09 > 0.06
    scratch = labeled.detections[1]
    assert scratch.panel == "right_lower"
    assert scratch.severity == Severity.MINOR      # area 0.0025 < 0.02


def test_to_target_json_is_parseable_and_compact():
    labeled = enrich(_cardd_img())
    s = to_target_json(labeled.detections)
    arr = json.loads(s)
    assert isinstance(arr, list) and len(arr) == 2
    assert set(arr[0].keys()) == {"damage_type", "panel", "severity", "bbox", "confidence"}
    assert arr[0]["bbox"] == [0.05, 0.1, 0.3, 0.3]


def test_instruction_example_shape():
    labeled = enrich(_cardd_img())
    ex = to_instruction_example(labeled)
    assert ex["image"] == "x/car1.jpg"
    assert "damage" in ex["prompt"].lower()
    assert json.loads(ex["response"])  # response is valid JSON
