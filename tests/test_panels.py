# tests/test_panels.py
from caridence.data.panels import infer_panel
from caridence.schema import BBox


def test_left_upper():
    assert infer_panel(BBox(x=0.05, y=0.1, w=0.1, h=0.1)) == "left_upper"


def test_right_lower():
    assert infer_panel(BBox(x=0.8, y=0.7, w=0.1, h=0.1)) == "right_lower"


def test_center_lower():
    assert infer_panel(BBox(x=0.45, y=0.6, w=0.1, h=0.1)) == "center_lower"
