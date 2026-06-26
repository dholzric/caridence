# tests/test_detector.py
from caridence.analyzer.detector import DetectorBackend, DEFAULT_CLASSES
from caridence.schema import Frame, DamageType, Severity


class _FakeBox:
    def __init__(self, cls, conf, xywhn):
        self.cls = cls
        self.conf = conf
        self.xywhn = [xywhn]  # mimics tensor[0].tolist() access


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeYOLO:
    def __init__(self, boxes):
        self._boxes = boxes
        self.last_kwargs = None

    def predict(self, source, **kwargs):
        self.last_kwargs = {"source": source, **kwargs}
        return [_FakeResult(self._boxes)]


def test_default_classes_order():
    assert DEFAULT_CLASSES[0] == DamageType.DENT
    assert DEFAULT_CLASSES[3] == DamageType.GLASS_SHATTER
    assert DEFAULT_CLASSES[5] == DamageType.TIRE_FLAT


def test_detector_maps_boxes_to_detections():
    boxes = [_FakeBox(0, 0.9, [0.5, 0.5, 0.2, 0.2]),
             _FakeBox(3, 0.7, [0.1, 0.1, 0.05, 0.05])]
    b = DetectorBackend(model=_FakeYOLO(boxes), conf=0.05)
    dets = b.detect(Frame(index=0, timestamp=0.0, path="x.jpg"))
    assert len(dets) == 2
    assert dets[0].damage_type == DamageType.DENT
    assert dets[1].damage_type == DamageType.GLASS_SHATTER
    # xywhn center 0.5,0.5 wh 0.2 -> top-left 0.4,0.4
    assert abs(dets[0].bbox.x - 0.4) < 1e-6
    assert abs(dets[0].bbox.y - 0.4) < 1e-6
    assert abs(dets[0].bbox.w - 0.2) < 1e-6
    assert dets[0].confidence == 0.9
    # panel + severity are populated deterministically
    assert dets[0].panel
    assert isinstance(dets[0].severity, Severity)


def test_detector_passes_inference_kwargs():
    fake = _FakeYOLO([])
    b = DetectorBackend(model=fake, conf=0.05, imgsz=1024, augment=True)
    b.detect(Frame(index=0, timestamp=0.0, path="img.jpg"))
    assert fake.last_kwargs["source"] == "img.jpg"
    assert fake.last_kwargs["conf"] == 0.05
    assert fake.last_kwargs["imgsz"] == 1024
    assert fake.last_kwargs["augment"] is True


def test_detector_empty_result():
    b = DetectorBackend(model=_FakeYOLO([]))
    assert b.detect(Frame(index=0, timestamp=0.0, path="x.jpg")) == []
