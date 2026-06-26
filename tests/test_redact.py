# tests/test_redact.py
import os
import numpy as np
import cv2
from caridence.redact import PlateRedactor, apply_redaction
from caridence.schema import Frame


class _FakeBox:
    def __init__(self, xyxy): self.xyxy = [xyxy]
class _FakeResult:
    def __init__(self, boxes): self.boxes = boxes
class _FakeYOLO:
    def __init__(self, boxes): self._b = boxes
    def predict(self, source, **kw): return [_FakeResult(self._b)]


def _img(tmp_path, name="car.jpg"):
    p = tmp_path / name
    a = np.zeros((100, 200, 3), dtype=np.uint8)
    a[40:60, 80:140] = 255  # bright "plate" patch
    cv2.imwrite(str(p), a)
    return p


def test_redactor_obscures_plate_region(tmp_path):
    p = _img(tmp_path)
    r = PlateRedactor(model=_FakeYOLO([_FakeBox([80, 40, 140, 60])]), method="box")
    out = tmp_path / "red.jpg"
    r.redact(str(p), str(out))
    orig, red = cv2.imread(str(p)), cv2.imread(str(out))
    # plate region changed, rest of image untouched
    assert not np.array_equal(orig[40:60, 80:140], red[40:60, 80:140])
    assert np.array_equal(orig[0:10, 0:10], red[0:10, 0:10])


def test_redactor_blur_method(tmp_path):
    p = _img(tmp_path)
    r = PlateRedactor(model=_FakeYOLO([_FakeBox([80, 40, 140, 60])]), method="blur")
    out = tmp_path / "red.jpg"
    r.redact(str(p), str(out))
    assert not np.array_equal(cv2.imread(str(p))[40:60, 80:140],
                              cv2.imread(str(out))[40:60, 80:140])


def test_redactor_no_detection_still_writes(tmp_path):
    p = _img(tmp_path)
    r = PlateRedactor(model=_FakeYOLO([]))
    out = tmp_path / "red.jpg"
    r.redact(str(p), str(out))
    assert os.path.exists(str(out))


def test_apply_redaction_updates_frame_paths(tmp_path):
    p = _img(tmp_path)
    frames = [Frame(index=0, timestamp=0.0, path=str(p))]
    r = PlateRedactor(model=_FakeYOLO([_FakeBox([80, 40, 140, 60])]), method="blur")
    out = apply_redaction(frames, r, str(tmp_path / "redacted"))
    assert out[0].path != str(p)
    assert out[0].path.endswith("car.jpg")
    assert os.path.exists(out[0].path)
