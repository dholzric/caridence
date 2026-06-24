# caridence/data/panels.py
from __future__ import annotations
from caridence.schema import BBox


def infer_panel(bbox: BBox) -> str:
    """Coarse position region from the bbox center. Training-label heuristic only."""
    cx = bbox.x + bbox.w / 2
    cy = bbox.y + bbox.h / 2
    h = "left" if cx < 1 / 3 else ("center" if cx < 2 / 3 else "right")
    v = "upper" if cy < 0.5 else "lower"
    return f"{h}_{v}"
