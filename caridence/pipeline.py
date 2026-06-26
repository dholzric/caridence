# caridence/pipeline.py
from __future__ import annotations
import tempfile
from pathlib import Path
from caridence.ingest import ingest_source
from caridence.analyzer import analyze_frames, VLMBackend
from caridence.aggregator import aggregate
from caridence.estimator import estimate, build_report
from caridence.redact import apply_redaction
from caridence.schema import InspectionReport


def run_inspection(source: Path, backend: VLMBackend,
                   vehicle_label: str | None = None, fps: float = 2.0,
                   redactor=None) -> InspectionReport:
    frames = ingest_source(source, fps=fps)
    if redactor is not None:
        # Obscure plates/faces before anything sees or stores the frames.
        frames = apply_redaction(frames, redactor,
                                 tempfile.mkdtemp(prefix="caridence_redact_"))
    frame_dets = analyze_frames(frames, backend)
    findings = aggregate(frame_dets)
    findings = estimate(findings)
    return build_report(findings, frame_count=len(frames), vehicle_label=vehicle_label)
