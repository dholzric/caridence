# caridence/pipeline.py
from __future__ import annotations
from pathlib import Path
from caridence.ingest import ingest_source
from caridence.analyzer import analyze_frames, VLMBackend
from caridence.aggregator import aggregate
from caridence.estimator import estimate, build_report
from caridence.schema import InspectionReport


def run_inspection(source: Path, backend: VLMBackend,
                   vehicle_label: str | None = None, fps: float = 2.0) -> InspectionReport:
    frames = ingest_source(source, fps=fps)
    frame_dets = analyze_frames(frames, backend)
    findings = aggregate(frame_dets)
    findings = estimate(findings)
    return build_report(findings, frame_count=len(frames), vehicle_label=vehicle_label)
