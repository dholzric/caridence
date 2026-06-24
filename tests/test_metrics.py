# tests/test_metrics.py
from caridence.metrics import iou, score_predictions, DetectionScore
from caridence.data.types import LabeledImage
from caridence.schema import Detection, BBox, DamageType, Severity


def _det(dt, x, y, w=0.1, h=0.1):
    return Detection(damage_type=dt, panel="p", severity=Severity.MINOR,
                     bbox=BBox(x=x, y=y, w=w, h=h))


def _img(dets):
    return LabeledImage(image_path="i.jpg", width=100, height=80, detections=dets)


def test_iou_identical_is_one():
    b = BBox(x=0.1, y=0.1, w=0.2, h=0.2)
    assert abs(iou(b, b) - 1.0) < 1e-9


def test_iou_disjoint_is_zero():
    assert iou(BBox(x=0, y=0, w=0.1, h=0.1), BBox(x=0.5, y=0.5, w=0.1, h=0.1)) == 0.0


def test_perfect_predictions_f1_one():
    gt = [_img([_det(DamageType.DENT, 0.1, 0.1)])]
    pred = [[_det(DamageType.DENT, 0.1, 0.1)]]
    s = score_predictions(gt, pred, iou_thr=0.3)
    assert isinstance(s, DetectionScore)
    assert s.f1 == 1.0 and s.precision == 1.0 and s.recall == 1.0


def test_wrong_type_counts_as_fp_and_fn():
    gt = [_img([_det(DamageType.DENT, 0.1, 0.1)])]
    pred = [[_det(DamageType.SCRATCH, 0.1, 0.1)]]
    s = score_predictions(gt, pred, iou_thr=0.3)
    assert s.recall == 0.0
    assert s.false_positives == 1


def test_missed_damage_lowers_recall():
    gt = [_img([_det(DamageType.DENT, 0.1, 0.1), _det(DamageType.SCRATCH, 0.7, 0.7)])]
    pred = [[_det(DamageType.DENT, 0.1, 0.1)]]
    s = score_predictions(gt, pred, iou_thr=0.3)
    assert s.recall == 0.5
    assert s.images_all_caught == 0.0
