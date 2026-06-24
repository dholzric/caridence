# tests/test_cardd.py
from caridence.data.cardd import parse_cardd_coco
from caridence.data.types import CarddImage
from caridence.schema import DamageType


def test_parse_returns_one_image_two_boxes(cardd_mini):
    images = parse_cardd_coco(cardd_mini["ann"], cardd_mini["images"])
    assert len(images) == 1
    img = images[0]
    assert isinstance(img, CarddImage)
    assert img.width == 100 and img.height == 80
    assert img.image_path.endswith("car1.jpg")
    assert len(img.boxes) == 2


def test_parse_normalizes_bbox_and_maps_types(cardd_mini):
    img = parse_cardd_coco(cardd_mini["ann"], cardd_mini["images"])[0]
    dent = next(b for b in img.boxes if b.damage_type == DamageType.DENT)
    # COCO bbox [10,8,30,24] on 100x80 -> normalized
    assert abs(dent.bbox.x - 0.10) < 1e-6
    assert abs(dent.bbox.y - 0.10) < 1e-6
    assert abs(dent.bbox.w - 0.30) < 1e-6
    assert abs(dent.bbox.h - 0.30) < 1e-6


def test_parse_maps_multiword_category(cardd_mini):
    # category name "glass shatter" must map to DamageType.GLASS_SHATTER if present
    from caridence.data.cardd import category_to_damage_type
    assert category_to_damage_type("glass shatter") == DamageType.GLASS_SHATTER
    assert category_to_damage_type("lamp broken") == DamageType.LAMP_BROKEN
    assert category_to_damage_type("tire flat") == DamageType.TIRE_FLAT
