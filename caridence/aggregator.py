# caridence/aggregator.py
from __future__ import annotations
import re
from caridence.schema import (
    FrameDetections, Finding, Severity, DamageType, Frame, BBox, Detection,
)

_RANK = {Severity.MINOR: 1, Severity.MODERATE: 2, Severity.SEVERE: 3}


def _severity_rank(s: Severity) -> int:
    return _RANK[s]


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _area(b: BBox) -> float:
    return b.w * b.h


def aggregate(frame_dets: list[FrameDetections]) -> list[Finding]:
    groups: dict[tuple[str, DamageType], list[tuple[Frame, Detection]]] = {}
    for fd in frame_dets:
        for det in fd.detections:
            groups.setdefault((det.panel, det.damage_type), []).append((fd.frame, det))

    findings: list[Finding] = []
    for (panel, dtype), occ in groups.items():
        best_frame, best_det = max(occ, key=lambda fd: (fd[1].confidence, _area(fd[1].bbox)))
        max_sev = max((d.severity for _, d in occ), key=_severity_rank)
        findings.append(Finding(
            id=f"{_slug(panel)}__{dtype.value}",
            damage_type=dtype,
            panel=panel,
            severity=max_sev,
            cited_frame=best_frame,
            bbox=best_det.bbox,
            confidence=best_det.confidence,
            occurrences=len(occ),
        ))
    findings.sort(key=lambda f: (-_severity_rank(f.severity), -f.confidence))
    # De-duplicate ids: if two findings slug to the same id, append a counter
    seen: dict[str, int] = {}
    deduped: list[Finding] = []
    for f in findings:
        base_id = f.id
        if base_id in seen:
            seen[base_id] += 1
            f = f.model_copy(update={"id": f"{base_id}__{seen[base_id]}"})
        else:
            seen[base_id] = 0
        deduped.append(f)
    return deduped
