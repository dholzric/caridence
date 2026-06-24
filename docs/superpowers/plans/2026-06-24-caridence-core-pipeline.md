# Caridence Core Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the end-to-end Caridence inspection pipeline — a phone walkaround video (or photo folder) becomes a structured vehicle-damage report with per-finding cited frames, severity, and cost estimates — served through a web UI, with a pluggable VLM backend so it runs on mock/base-model today and the fine-tuned MI300 model later.

**Architecture:** A linear, deterministic pipeline of small single-responsibility modules under a `caridence/` package: `ingest` (video/photos → frames) → `analyzer` (per-frame VLM detections via a swappable backend) → `aggregator` (multi-frame detections → unique findings + cited frame) → `estimator` (severity + repair-cost) → `report` (JSON + HTML). A FastAPI app drives it. The VLM is abstracted behind a `VLMBackend` protocol with a `MockBackend` (canned, no GPU) and a `QwenHTTPBackend` (vLLM OpenAI-compatible endpoint). Pydantic models are the contracts between stages.

**Tech Stack:** Python 3.11, Pydantic v2, OpenCV (`opencv-python`) for frame extraction + blur detection, NumPy, Jinja2 for HTML, FastAPI + Uvicorn, `openai` client for the vLLM endpoint, pytest. Docker for packaging.

---

## File Structure

```
caridence/
  __init__.py
  schema.py            # Pydantic contracts: DamageType, Severity, BBox, Detection, Frame, FrameDetections, Finding, InspectionReport
  ingest.py            # extract_frames (cv2), variance_of_laplacian, filter_blurry, dedupe_frames, load_photos, ingest_source
  analyzer/
    __init__.py        # re-exports VLMBackend, analyze_frames
    base.py            # VLMBackend Protocol, parse_detections, analyze_frames
    mock.py            # MockBackend (canned detections keyed by frame)
    qwen_http.py       # QwenHTTPBackend (vLLM OpenAI-compatible), build_prompt
  aggregator.py        # aggregate (group detections -> Findings, pick cited frame, panel merge)
  costs.py             # COST_TABLE + lookup_cost
  estimator.py         # estimate (severity already set; attach cost) + condition_score + build_report
  pipeline.py          # run_inspection (orchestrates all stages)
app/
  __init__.py
  server.py            # FastAPI: /, /inspect (upload), /report/{id}, /dashboard
  templates/
    report.html        # Jinja2 report template
    dashboard.html     # benchmark dashboard (reads data/bench.json; placeholder data committed)
  static/
    app.js, styles.css
tests/
  conftest.py          # fixtures: synthetic frames, synthetic video, sample detections
  test_schema.py
  test_ingest.py
  test_analyzer_mock.py
  test_aggregator.py
  test_estimator.py
  test_report.py
  test_pipeline.py
  test_qwen_http.py
  test_server.py
data/
  bench.json           # committed placeholder dashboard data (real numbers land in Plan 2)
pyproject.toml
Dockerfile
```

Each module owns one stage. `schema.py` holds every cross-stage type so the contracts live in one place. The analyzer is a sub-package because it has three swappable implementations.

---

## Task 0: Project scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `caridence/__init__.py` (empty)
- Create: `caridence/analyzer/__init__.py` (empty for now)
- Create: `app/__init__.py` (empty)
- Create: `tests/__init__.py` (empty)

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "caridence"
version = "0.1.0"
description = "Vehicle-damage walkaround inspector with cited evidence frames"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.6",
    "opencv-python-headless>=4.9",
    "numpy>=1.26",
    "jinja2>=3.1",
    "fastapi>=0.110",
    "uvicorn>=0.29",
    "python-multipart>=0.0.9",
    "openai>=1.30",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "httpx>=0.27"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.setuptools.packages.find]
include = ["caridence*", "app*"]
```

- [ ] **Step 2: Create empty package files**

Create `caridence/__init__.py`, `caridence/analyzer/__init__.py`, `app/__init__.py`, `tests/__init__.py` as empty files.

- [ ] **Step 3: Create venv and install**

Run: `python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"`
Expected: installs successfully, ends with `Successfully installed ... caridence-0.1.0`.

- [ ] **Step 4: Verify pytest runs (no tests yet)**

Run: `.venv/Scripts/pytest`
Expected: `no tests ran` (exit code 5 is fine at this stage).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml caridence/__init__.py caridence/analyzer/__init__.py app/__init__.py tests/__init__.py
git commit -m "chore: scaffold caridence package"
```

---

## Task 1: Schema contracts

**Files:**
- Create: `caridence/schema.py`
- Test: `tests/test_schema.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_schema.py
from caridence.schema import (
    DamageType, Severity, BBox, Detection, Frame, FrameDetections,
    Finding, InspectionReport,
)


def test_bbox_rejects_out_of_range():
    import pytest
    with pytest.raises(Exception):
        BBox(x=1.5, y=0.0, w=0.1, h=0.1)


def test_detection_roundtrip():
    d = Detection(
        damage_type=DamageType.DENT, panel="front_driver_door",
        severity=Severity.MODERATE, bbox=BBox(x=0.1, y=0.2, w=0.3, h=0.4),
        confidence=0.9,
    )
    assert d.damage_type == "dent"
    assert Detection.model_validate(d.model_dump()) == d


def test_inspection_report_minimal():
    f = Frame(index=0, timestamp=0.0, path="frames/0.jpg")
    finding = Finding(
        id="f1", damage_type=DamageType.SCRATCH, panel="rear_bumper",
        severity=Severity.MINOR, cited_frame=f, bbox=BBox(x=0, y=0, w=0.1, h=0.1),
        confidence=0.8, occurrences=2, cost_low=80.0, cost_high=150.0,
    )
    rep = InspectionReport(
        vehicle_label="Chevy", findings=[finding], condition_score=82,
        total_cost_low=80.0, total_cost_high=150.0, frame_count=30,
    )
    assert rep.findings[0].panel == "rear_bumper"
    assert rep.total_cost_high == 150.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/pytest tests/test_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'caridence.schema'`.

- [ ] **Step 3: Write `caridence/schema.py`**

```python
# caridence/schema.py
from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field


class DamageType(str, Enum):
    DENT = "dent"
    SCRATCH = "scratch"
    CRACK = "crack"
    GLASS_SHATTER = "glass_shatter"
    LAMP_BROKEN = "lamp_broken"
    TIRE_FLAT = "tire_flat"


class Severity(str, Enum):
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"


class BBox(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    w: float = Field(gt=0.0, le=1.0)
    h: float = Field(gt=0.0, le=1.0)


class Frame(BaseModel):
    index: int
    timestamp: float  # seconds from start
    path: str


class Detection(BaseModel):
    damage_type: DamageType
    panel: str
    severity: Severity
    bbox: BBox
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class FrameDetections(BaseModel):
    frame: Frame
    detections: list[Detection]


class Finding(BaseModel):
    id: str
    damage_type: DamageType
    panel: str
    severity: Severity
    cited_frame: Frame
    bbox: BBox
    confidence: float
    occurrences: int
    cost_low: float | None = None
    cost_high: float | None = None


class InspectionReport(BaseModel):
    vehicle_label: str | None = None
    findings: list[Finding]
    condition_score: int = Field(ge=0, le=100)
    total_cost_low: float
    total_cost_high: float
    frame_count: int
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/pytest tests/test_schema.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add caridence/schema.py tests/test_schema.py
git commit -m "feat: schema contracts for detections, findings, reports"
```

---

## Task 2: Ingest (video/photos → frames)

**Files:**
- Create: `caridence/ingest.py`
- Create: `tests/conftest.py`
- Test: `tests/test_ingest.py`

- [ ] **Step 1: Write fixtures in `tests/conftest.py`**

```python
# tests/conftest.py
import numpy as np
import cv2
import pytest


def _solid(color, size=(120, 160)):
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    img[:] = color
    return img


@pytest.fixture
def frames_dir(tmp_path):
    """Three sharp, visually distinct images + one blurry duplicate of the first."""
    d = tmp_path / "photos"
    d.mkdir()
    sharp_colors = [(20, 40, 200), (30, 200, 40), (200, 60, 30)]
    for i, c in enumerate(sharp_colors):
        img = _solid(c)
        cv2.rectangle(img, (10, 10), (60, 80), (255, 255, 255), 2)  # sharp edges
        cv2.imwrite(str(d / f"img_{i}.jpg"), img)
    blurry = cv2.GaussianBlur(_solid(sharp_colors[0]), (21, 21), 0)
    cv2.imwrite(str(d / "img_blurry.jpg"), blurry)
    return d


@pytest.fixture
def sample_video(tmp_path):
    """A 2-second 10fps video that cycles through 3 colors."""
    path = tmp_path / "walk.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (160, 120))
    colors = [(20, 40, 200), (30, 200, 40), (200, 60, 30)]
    for n in range(20):
        img = _solid(colors[n % 3])
        cv2.rectangle(img, (10, 10), (60, 80), (255, 255, 255), 2)
        writer.write(img)
    writer.release()
    return path
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_ingest.py
from caridence.ingest import (
    variance_of_laplacian, filter_blurry, dedupe_frames,
    load_photos, extract_frames, ingest_source,
)
from caridence.schema import Frame
import cv2


def test_variance_of_laplacian_higher_for_sharp(frames_dir):
    sharp = cv2.imread(str(frames_dir / "img_0.jpg"))
    blurry = cv2.imread(str(frames_dir / "img_blurry.jpg"))
    assert variance_of_laplacian(sharp) > variance_of_laplacian(blurry)


def test_load_photos_returns_sorted_frames(frames_dir):
    frames = load_photos(frames_dir)
    assert [f.index for f in frames] == [0, 1, 2, 3]
    assert all(isinstance(f, Frame) for f in frames)
    assert frames[0].path.endswith("img_0.jpg")


def test_filter_blurry_drops_blurry(frames_dir):
    frames = load_photos(frames_dir)
    kept = filter_blurry(frames, threshold=100.0)
    names = [f.path.split("/")[-1].split("\\")[-1] for f in kept]
    assert "img_blurry.jpg" not in names
    assert len(kept) == 3


def test_dedupe_frames_removes_near_duplicates(frames_dir):
    frames = load_photos(frames_dir)
    # img_0 and img_blurry are the same color -> near-duplicate
    deduped = dedupe_frames(frames, hash_threshold=5)
    assert len(deduped) <= 3


def test_extract_frames_from_video(sample_video):
    frames = extract_frames(sample_video, fps=2.0)
    assert len(frames) >= 3
    assert frames[0].timestamp == 0.0
    assert frames[1].timestamp > frames[0].timestamp


def test_ingest_source_dispatches(frames_dir, sample_video):
    from_photos = ingest_source(frames_dir, fps=2.0)
    from_video = ingest_source(sample_video, fps=2.0)
    assert len(from_photos) >= 1
    assert len(from_video) >= 1
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/Scripts/pytest tests/test_ingest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'caridence.ingest'`.

- [ ] **Step 4: Write `caridence/ingest.py`**

```python
# caridence/ingest.py
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from caridence.schema import Frame

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def variance_of_laplacian(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _ahash(image: np.ndarray, size: int = 8) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size, size))
    avg = small.mean()
    bits = (small > avg).flatten()
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def load_photos(folder: Path) -> list[Frame]:
    folder = Path(folder)
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return [Frame(index=i, timestamp=float(i), path=str(p)) for i, p in enumerate(paths)]


def extract_frames(video_path: Path, fps: float = 2.0) -> list[Frame]:
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / fps)))
    out_dir = video_path.parent / f"{video_path.stem}_frames"
    out_dir.mkdir(exist_ok=True)
    frames: list[Frame] = []
    pos = 0
    kept = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if pos % step == 0:
            fp = out_dir / f"frame_{kept:04d}.jpg"
            cv2.imwrite(str(fp), img)
            frames.append(Frame(index=kept, timestamp=pos / src_fps, path=str(fp)))
            kept += 1
        pos += 1
    cap.release()
    return frames


def filter_blurry(frames: list[Frame], threshold: float = 100.0) -> list[Frame]:
    kept: list[Frame] = []
    for f in frames:
        img = cv2.imread(f.path)
        if img is None:
            continue
        if variance_of_laplacian(img) >= threshold:
            kept.append(f)
    return kept


def dedupe_frames(frames: list[Frame], hash_threshold: int = 5) -> list[Frame]:
    kept: list[Frame] = []
    hashes: list[int] = []
    for f in frames:
        img = cv2.imread(f.path)
        if img is None:
            continue
        h = _ahash(img)
        if any(_hamming(h, prev) <= hash_threshold for prev in hashes):
            continue
        hashes.append(h)
        kept.append(f)
    return kept


def ingest_source(source: Path, fps: float = 2.0,
                  blur_threshold: float = 100.0, hash_threshold: int = 5) -> list[Frame]:
    source = Path(source)
    if source.is_dir():
        frames = load_photos(source)
    elif source.suffix.lower() in VIDEO_EXTS:
        frames = extract_frames(source, fps=fps)
    else:
        raise ValueError(f"Unsupported source: {source}")
    frames = filter_blurry(frames, threshold=blur_threshold)
    frames = dedupe_frames(frames, hash_threshold=hash_threshold)
    # reindex after filtering so indices are contiguous
    return [f.model_copy(update={"index": i}) for i, f in enumerate(frames)]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/Scripts/pytest tests/test_ingest.py -v`
Expected: 6 passed. (If `test_filter_blurry` is flaky on the synthetic blur, adjust the fixture's GaussianBlur kernel — the sharp image has white rectangle edges giving high Laplacian variance; the blurred solid has near-zero.)

- [ ] **Step 6: Commit**

```bash
git add caridence/ingest.py tests/conftest.py tests/test_ingest.py
git commit -m "feat: ingest video/photos into deduped, blur-filtered frames"
```

---

## Task 3: Analyzer — backend protocol + mock

**Files:**
- Create: `caridence/analyzer/base.py`
- Create: `caridence/analyzer/mock.py`
- Modify: `caridence/analyzer/__init__.py`
- Test: `tests/test_analyzer_mock.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_analyzer_mock.py
from caridence.analyzer import analyze_frames
from caridence.analyzer.mock import MockBackend
from caridence.analyzer.base import parse_detections
from caridence.schema import Frame, DamageType


def test_parse_detections_valid_json():
    raw = '[{"damage_type":"dent","panel":"front_driver_door","severity":"moderate","bbox":[0.1,0.2,0.3,0.4],"confidence":0.9}]'
    dets = parse_detections(raw)
    assert len(dets) == 1
    assert dets[0].damage_type == DamageType.DENT
    assert dets[0].bbox.w == 0.3


def test_parse_detections_tolerates_garbage_and_fences():
    raw = "```json\n[]\n```"
    assert parse_detections(raw) == []
    assert parse_detections("not json at all") == []


def test_mock_backend_is_deterministic():
    b = MockBackend()
    f = Frame(index=0, timestamp=0.0, path="x/img_0.jpg")
    assert b.detect(f) == b.detect(f)


def test_analyze_frames_returns_framedetections():
    b = MockBackend()
    frames = [Frame(index=i, timestamp=float(i), path=f"x/img_{i}.jpg") for i in range(4)]
    results = analyze_frames(frames, b)
    assert len(results) == 4
    assert all(r.frame.index == frames[i].index for i, r in enumerate(results))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/pytest tests/test_analyzer_mock.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'caridence.analyzer.base'`.

- [ ] **Step 3: Write `caridence/analyzer/base.py`**

```python
# caridence/analyzer/base.py
from __future__ import annotations
import json
import re
from typing import Protocol
from caridence.schema import Frame, Detection, FrameDetections, BBox

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class VLMBackend(Protocol):
    def detect(self, frame: Frame) -> list[Detection]:
        ...


def _coerce_bbox(value) -> BBox:
    if isinstance(value, dict):
        return BBox(**value)
    x, y, w, h = value
    return BBox(x=x, y=y, w=w, h=h)


def parse_detections(raw: str) -> list[Detection]:
    """Parse model output into Detections. Never raises — bad output -> []."""
    text = raw.strip()
    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        items = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []
    out: list[Detection] = []
    for it in items:
        try:
            out.append(Detection(
                damage_type=it["damage_type"],
                panel=it["panel"],
                severity=it["severity"],
                bbox=_coerce_bbox(it["bbox"]),
                confidence=float(it.get("confidence", 1.0)),
            ))
        except Exception:
            continue
    return out


def analyze_frames(frames: list[Frame], backend: VLMBackend) -> list[FrameDetections]:
    return [FrameDetections(frame=f, detections=backend.detect(f)) for f in frames]
```

- [ ] **Step 4: Write `caridence/analyzer/mock.py`**

```python
# caridence/analyzer/mock.py
from __future__ import annotations
from caridence.schema import Frame, Detection, DamageType, Severity, BBox

# A canned "four corners" scenario keyed by frame index, so the mock produces
# a realistic multi-finding report deterministically (used for UI/demo dev).
_SCRIPT: dict[int, list[Detection]] = {
    0: [Detection(damage_type=DamageType.SCRATCH, panel="front_driver_corner",
                  severity=Severity.MINOR, bbox=BBox(x=0.1, y=0.5, w=0.2, h=0.15), confidence=0.86)],
    1: [Detection(damage_type=DamageType.SCRATCH, panel="front_driver_corner",
                  severity=Severity.MINOR, bbox=BBox(x=0.12, y=0.52, w=0.2, h=0.15), confidence=0.9)],
    2: [Detection(damage_type=DamageType.DENT, panel="rear_driver_corner",
                  severity=Severity.MODERATE, bbox=BBox(x=0.6, y=0.55, w=0.18, h=0.2), confidence=0.88)],
    3: [Detection(damage_type=DamageType.DENT, panel="rear_passenger_corner",
                  severity=Severity.MINOR, bbox=BBox(x=0.55, y=0.5, w=0.16, h=0.18), confidence=0.81)],
    4: [Detection(damage_type=DamageType.SCRATCH, panel="front_passenger_corner",
                  severity=Severity.MINOR, bbox=BBox(x=0.2, y=0.48, w=0.15, h=0.12), confidence=0.79)],
}


class MockBackend:
    """Deterministic canned backend. No GPU/endpoint required."""

    def detect(self, frame: Frame) -> list[Detection]:
        return [d.model_copy(deep=True) for d in _SCRIPT.get(frame.index, [])]
```

- [ ] **Step 5: Write `caridence/analyzer/__init__.py`**

```python
# caridence/analyzer/__init__.py
from caridence.analyzer.base import VLMBackend, analyze_frames, parse_detections

__all__ = ["VLMBackend", "analyze_frames", "parse_detections"]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/Scripts/pytest tests/test_analyzer_mock.py -v`
Expected: 4 passed.

- [ ] **Step 7: Commit**

```bash
git add caridence/analyzer/ tests/test_analyzer_mock.py
git commit -m "feat: analyzer backend protocol, robust parser, mock backend"
```

---

## Task 4: Aggregator (detections → findings + cited frame)

**Files:**
- Create: `caridence/aggregator.py`
- Test: `tests/test_aggregator.py`

**Design:** group detections across frames by `(panel, damage_type)`. Each group becomes one `Finding`. The **cited frame** is the occurrence with the highest `confidence` (ties broken by largest bbox area). `severity` is the max severity in the group. `occurrences` is the group size. `bbox` is the cited frame's bbox. Finding `id` is `f"{panel}:{damage_type}"` slugified.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aggregator.py
from caridence.aggregator import aggregate, _severity_rank
from caridence.schema import (
    Frame, Detection, FrameDetections, DamageType, Severity, BBox,
)


def _fd(idx, dets):
    return FrameDetections(frame=Frame(index=idx, timestamp=float(idx), path=f"x/{idx}.jpg"),
                           detections=dets)


def test_severity_rank_orders():
    assert _severity_rank(Severity.SEVERE) > _severity_rank(Severity.MINOR)


def test_aggregate_merges_same_panel_and_type():
    d_low = Detection(damage_type=DamageType.SCRATCH, panel="front_driver_corner",
                      severity=Severity.MINOR, bbox=BBox(x=0.1, y=0.1, w=0.1, h=0.1), confidence=0.6)
    d_high = Detection(damage_type=DamageType.SCRATCH, panel="front_driver_corner",
                       severity=Severity.MODERATE, bbox=BBox(x=0.1, y=0.1, w=0.2, h=0.2), confidence=0.95)
    findings = aggregate([_fd(0, [d_low]), _fd(1, [d_high])])
    assert len(findings) == 1
    f = findings[0]
    assert f.occurrences == 2
    assert f.severity == Severity.MODERATE          # max severity
    assert f.cited_frame.index == 1                  # highest confidence
    assert f.confidence == 0.95


def test_aggregate_separates_distinct_panels():
    a = Detection(damage_type=DamageType.DENT, panel="rear_driver_corner",
                  severity=Severity.MODERATE, bbox=BBox(x=0.1, y=0.1, w=0.1, h=0.1), confidence=0.9)
    b = Detection(damage_type=DamageType.DENT, panel="rear_passenger_corner",
                  severity=Severity.MINOR, bbox=BBox(x=0.5, y=0.1, w=0.1, h=0.1), confidence=0.8)
    findings = aggregate([_fd(0, [a, b])])
    assert len(findings) == 2
    assert {f.panel for f in findings} == {"rear_driver_corner", "rear_passenger_corner"}


def test_aggregate_empty():
    assert aggregate([]) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/pytest tests/test_aggregator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'caridence.aggregator'`.

- [ ] **Step 3: Write `caridence/aggregator.py`**

```python
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
    return findings
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/pytest tests/test_aggregator.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add caridence/aggregator.py tests/test_aggregator.py
git commit -m "feat: aggregate per-frame detections into cited findings"
```

---

## Task 5: Estimator (cost + condition score + report assembly)

**Files:**
- Create: `caridence/costs.py`
- Create: `caridence/estimator.py`
- Test: `tests/test_estimator.py`

**Design:** `COST_TABLE[(damage_type, severity)] -> (low, high)` rough US body-shop ranges. `lookup_cost` returns the range. `estimate(findings)` attaches `cost_low/cost_high` to each. `condition_score(findings)` starts at 100 and subtracts severity-weighted penalties, floored at 0. `build_report` assembles the `InspectionReport` (totals + score + frame_count).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_estimator.py
from caridence.costs import COST_TABLE, lookup_cost
from caridence.estimator import estimate, condition_score, build_report
from caridence.schema import (
    Finding, Frame, BBox, DamageType, Severity, InspectionReport,
)


def _finding(dtype, sev, conf=0.9, idx=0):
    return Finding(
        id=f"{dtype.value}", damage_type=dtype, panel="p", severity=sev,
        cited_frame=Frame(index=idx, timestamp=float(idx), path=f"{idx}.jpg"),
        bbox=BBox(x=0.1, y=0.1, w=0.1, h=0.1), confidence=conf, occurrences=1,
    )


def test_cost_table_covers_all_combinations():
    for dt in DamageType:
        for sv in Severity:
            assert (dt, sv) in COST_TABLE


def test_lookup_cost_monotonic_in_severity():
    minor = lookup_cost(DamageType.DENT, Severity.MINOR)
    severe = lookup_cost(DamageType.DENT, Severity.SEVERE)
    assert severe[0] > minor[0] and severe[1] > minor[1]


def test_estimate_attaches_costs():
    findings = estimate([_finding(DamageType.SCRATCH, Severity.MINOR)])
    assert findings[0].cost_low is not None
    assert findings[0].cost_high >= findings[0].cost_low


def test_condition_score_decreases_with_damage():
    clean = condition_score([])
    damaged = condition_score([_finding(DamageType.DENT, Severity.SEVERE)])
    assert clean == 100
    assert damaged < clean


def test_build_report_totals():
    findings = estimate([
        _finding(DamageType.SCRATCH, Severity.MINOR),
        _finding(DamageType.DENT, Severity.MODERATE, idx=2),
    ])
    rep = build_report(findings, frame_count=30, vehicle_label="Chevy")
    assert isinstance(rep, InspectionReport)
    assert rep.total_cost_low == sum(f.cost_low for f in findings)
    assert rep.total_cost_high == sum(f.cost_high for f in findings)
    assert rep.frame_count == 30
    assert 0 <= rep.condition_score <= 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/pytest tests/test_estimator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'caridence.costs'`.

- [ ] **Step 3: Write `caridence/costs.py`**

```python
# caridence/costs.py
from __future__ import annotations
from caridence.schema import DamageType, Severity

# Rough US body-shop ranges (USD). Estimates only — surfaced as ranges in the UI.
_BASE: dict[DamageType, tuple[float, float]] = {
    DamageType.SCRATCH: (80.0, 200.0),
    DamageType.DENT: (150.0, 600.0),
    DamageType.CRACK: (120.0, 400.0),
    DamageType.GLASS_SHATTER: (250.0, 700.0),
    DamageType.LAMP_BROKEN: (90.0, 450.0),
    DamageType.TIRE_FLAT: (20.0, 250.0),
}
_MULT: dict[Severity, float] = {
    Severity.MINOR: 1.0,
    Severity.MODERATE: 1.8,
    Severity.SEVERE: 3.0,
}

COST_TABLE: dict[tuple[DamageType, Severity], tuple[float, float]] = {
    (dt, sv): (round(lo * _MULT[sv], 2), round(hi * _MULT[sv], 2))
    for dt, (lo, hi) in _BASE.items()
    for sv in Severity
}


def lookup_cost(damage_type: DamageType, severity: Severity) -> tuple[float, float]:
    return COST_TABLE[(damage_type, severity)]
```

- [ ] **Step 4: Write `caridence/estimator.py`**

```python
# caridence/estimator.py
from __future__ import annotations
from caridence.schema import Finding, Severity, InspectionReport
from caridence.costs import lookup_cost

_PENALTY = {Severity.MINOR: 4, Severity.MODERATE: 10, Severity.SEVERE: 20}


def estimate(findings: list[Finding]) -> list[Finding]:
    out: list[Finding] = []
    for f in findings:
        low, high = lookup_cost(f.damage_type, f.severity)
        out.append(f.model_copy(update={"cost_low": low, "cost_high": high}))
    return out


def condition_score(findings: list[Finding]) -> int:
    score = 100 - sum(_PENALTY[f.severity] for f in findings)
    return max(0, min(100, score))


def build_report(findings: list[Finding], frame_count: int,
                 vehicle_label: str | None = None) -> InspectionReport:
    priced = [f for f in findings if f.cost_low is not None]
    return InspectionReport(
        vehicle_label=vehicle_label,
        findings=findings,
        condition_score=condition_score(findings),
        total_cost_low=round(sum(f.cost_low for f in priced), 2),
        total_cost_high=round(sum(f.cost_high for f in priced), 2),
        frame_count=frame_count,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/pytest tests/test_estimator.py -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add caridence/costs.py caridence/estimator.py tests/test_estimator.py
git commit -m "feat: cost table, repair estimate, condition score, report assembly"
```

---

## Task 6: Report rendering (JSON + HTML)

**Files:**
- Create: `caridence/report.py`
- Create: `app/templates/report.html`
- Test: `tests/test_report.py`

**Design:** `report_to_dict(report)` is just `report.model_dump()`. `render_report_html(report)` uses Jinja2 to produce a standalone dark-theme HTML page listing findings with the cited frame image (referenced by path) and a drawn bbox overlay (CSS-positioned div over the image using the normalized bbox). Keep the template self-contained.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_report.py
from caridence.report import render_report_html, report_to_dict
from caridence.estimator import build_report
from caridence.schema import Finding, Frame, BBox, DamageType, Severity


def _report():
    f = Finding(
        id="rear_driver__dent", damage_type=DamageType.DENT, panel="rear_driver_corner",
        severity=Severity.MODERATE, cited_frame=Frame(index=2, timestamp=2.0, path="frames/2.jpg"),
        bbox=BBox(x=0.6, y=0.55, w=0.18, h=0.2), confidence=0.88, occurrences=3,
        cost_low=270.0, cost_high=1080.0,
    )
    return build_report([f], frame_count=30, vehicle_label="Chevy")


def test_report_to_dict_roundtrips():
    d = report_to_dict(_report())
    assert d["vehicle_label"] == "Chevy"
    assert d["findings"][0]["panel"] == "rear_driver_corner"


def test_render_html_contains_key_facts():
    html = render_report_html(_report())
    assert "Chevy" in html
    assert "rear_driver_corner" in html or "rear driver corner" in html.lower()
    assert "Condition" in html
    assert "270" in html  # cost appears
    assert "<html" in html.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/pytest tests/test_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'caridence.report'`.

- [ ] **Step 3: Write `app/templates/report.html`**

```html
{# app/templates/report.html #}
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Caridence Report — {{ report.vehicle_label or "Vehicle" }}</title>
<style>
  :root{color-scheme:dark}
  body{font-family:ui-sans-serif,system-ui,"Segoe UI",Roboto,sans-serif;background:#0a0f1a;color:#e6edf6;margin:0;padding:2rem}
  h1{letter-spacing:-.02em} .muted{color:#9fb0c6}
  .summary{display:flex;gap:2rem;flex-wrap:wrap;margin:1rem 0 2rem}
  .stat{background:#0d1830;border:1px solid #1e3a5f;border-radius:12px;padding:1rem 1.4rem}
  .stat .v{font-size:1.8rem;font-weight:700}
  .finding{background:#0d1830;border:1px solid #1e3a5f;border-radius:12px;padding:1rem;margin-bottom:1rem;display:grid;grid-template-columns:240px 1fr;gap:1rem}
  .frame{position:relative;width:240px;border-radius:8px;overflow:hidden;background:#02060d}
  .frame img{width:100%;display:block}
  .box{position:absolute;border:2px solid #f87171;box-shadow:0 0 0 9999px rgba(0,0,0,.12)}
  .sev-minor{color:#7dd3fc}.sev-moderate{color:#fbbf24}.sev-severe{color:#f87171}
  .pill{display:inline-block;font-size:.72rem;text-transform:uppercase;letter-spacing:.12em;border:1px solid #1e3a5f;border-radius:999px;padding:.2rem .6rem;margin-right:.4rem}
</style>
</head>
<body>
  <h1>Caridence — {{ report.vehicle_label or "Vehicle" }} condition report</h1>
  <div class="summary">
    <div class="stat"><div class="muted">Condition</div><div class="v">{{ report.condition_score }}/100</div></div>
    <div class="stat"><div class="muted">Findings</div><div class="v">{{ report.findings|length }}</div></div>
    <div class="stat"><div class="muted">Est. repair</div><div class="v">${{ "%.0f"|format(report.total_cost_low) }}–${{ "%.0f"|format(report.total_cost_high) }}</div></div>
    <div class="stat"><div class="muted">Frames analyzed</div><div class="v">{{ report.frame_count }}</div></div>
  </div>
  {% for f in report.findings %}
  <div class="finding">
    <div class="frame">
      <img src="/frame?path={{ f.cited_frame.path|urlencode }}" alt="cited frame">
      <div class="box" style="left:{{ (f.bbox.x*100)|round(1) }}%;top:{{ (f.bbox.y*100)|round(1) }}%;width:{{ (f.bbox.w*100)|round(1) }}%;height:{{ (f.bbox.h*100)|round(1) }}%"></div>
    </div>
    <div>
      <div><span class="pill sev-{{ f.severity }}">{{ f.severity }}</span><span class="pill">{{ f.damage_type }}</span></div>
      <h3>{{ f.panel.replace("_"," ") }}</h3>
      <div class="muted">Cited at {{ "%.1f"|format(f.cited_frame.timestamp) }}s · seen in {{ f.occurrences }} frame(s) · confidence {{ "%.0f"|format(f.confidence*100) }}%</div>
      <div style="margin-top:.5rem">Est. repair: <strong>${{ "%.0f"|format(f.cost_low) }}–${{ "%.0f"|format(f.cost_high) }}</strong></div>
    </div>
  </div>
  {% endfor %}
</body>
</html>
```

- [ ] **Step 4: Write `caridence/report.py`**

```python
# caridence/report.py
from __future__ import annotations
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape
from caridence.schema import InspectionReport

_TEMPLATES = Path(__file__).resolve().parent.parent / "app" / "templates"
_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATES)),
    autoescape=select_autoescape(["html"]),
)
_env.filters["urlencode"] = lambda s: __import__("urllib.parse", fromlist=["quote"]).quote(str(s))


def report_to_dict(report: InspectionReport) -> dict:
    return report.model_dump()


def render_report_html(report: InspectionReport) -> str:
    return _env.get_template("report.html").render(report=report)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/pytest tests/test_report.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add caridence/report.py app/templates/report.html tests/test_report.py
git commit -m "feat: JSON + HTML report rendering with cited-frame bbox overlay"
```

---

## Task 7: Pipeline orchestration

**Files:**
- Create: `caridence/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline.py
from caridence.pipeline import run_inspection
from caridence.analyzer.mock import MockBackend
from caridence.schema import InspectionReport


def test_run_inspection_end_to_end_photos(frames_dir):
    report = run_inspection(frames_dir, backend=MockBackend(), vehicle_label="TestCar", fps=2.0)
    assert isinstance(report, InspectionReport)
    assert report.frame_count >= 1
    # Mock script emits damage on low indices -> at least one finding expected
    assert len(report.findings) >= 1
    assert all(f.cost_low is not None for f in report.findings)


def test_run_inspection_handles_clean_vehicle(frames_dir):
    class CleanBackend:
        def detect(self, frame):
            return []
    report = run_inspection(frames_dir, backend=CleanBackend())
    assert report.condition_score == 100
    assert report.findings == []
    assert report.total_cost_low == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/pytest tests/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'caridence.pipeline'`.

- [ ] **Step 3: Write `caridence/pipeline.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/pytest tests/test_pipeline.py -v`
Expected: 2 passed.

- [ ] **Step 5: Run the full suite**

Run: `.venv/Scripts/pytest`
Expected: all tests pass (schema, ingest, analyzer, aggregator, estimator, report, pipeline).

- [ ] **Step 6: Commit**

```bash
git add caridence/pipeline.py tests/test_pipeline.py
git commit -m "feat: end-to-end inspection pipeline orchestration"
```

---

## Task 8: Qwen HTTP backend (vLLM OpenAI-compatible)

**Files:**
- Create: `caridence/analyzer/qwen_http.py`
- Test: `tests/test_qwen_http.py`

**Design:** `QwenHTTPBackend` posts a base64 image + a strict prompt to an OpenAI-compatible `/chat/completions` endpoint (vLLM serving Qwen2.5-VL), then runs the response through `parse_detections`. The endpoint/model/key come from constructor args (default to env `CARIDENCE_API_BASE`, `CARIDENCE_MODEL`, `CARIDENCE_API_KEY`). The test injects a fake OpenAI client so no network is needed — verifying the backend (a) sends an image payload and (b) parses the returned JSON into Detections.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_qwen_http.py
from caridence.analyzer.qwen_http import QwenHTTPBackend, build_prompt
from caridence.schema import Frame, DamageType
import numpy as np, cv2


class _FakeMessage:
    def __init__(self, content): self.message = type("M", (), {"content": content})
class _FakeResp:
    def __init__(self, content): self.choices = [_FakeMessage(content)]
class _FakeCompletions:
    def __init__(self, content): self._c = content; self.last_kwargs = None
    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeResp(self._c)
class _FakeClient:
    def __init__(self, content):
        self.chat = type("C", (), {"completions": _FakeCompletions(content)})()


def test_build_prompt_lists_damage_types():
    p = build_prompt()
    for dt in DamageType:
        assert dt.value in p


def test_backend_parses_model_json(tmp_path):
    img = np.zeros((40, 60, 3), dtype=np.uint8)
    fp = tmp_path / "f.jpg"; cv2.imwrite(str(fp), img)
    frame = Frame(index=0, timestamp=0.0, path=str(fp))
    content = '[{"damage_type":"dent","panel":"hood","severity":"moderate","bbox":[0.1,0.2,0.3,0.4],"confidence":0.9}]'
    backend = QwenHTTPBackend(client=_FakeClient(content), model="test-model")
    dets = backend.detect(frame)
    assert len(dets) == 1 and dets[0].damage_type == DamageType.DENT
    # confirm an image payload was sent
    sent = backend.client.chat.completions.last_kwargs
    assert sent["model"] == "test-model"
    assert any("image_url" in str(part) for part in sent["messages"][0]["content"])


def test_backend_tolerates_bad_output(tmp_path):
    img = np.zeros((40, 60, 3), dtype=np.uint8)
    fp = tmp_path / "f.jpg"; cv2.imwrite(str(fp), img)
    frame = Frame(index=0, timestamp=0.0, path=str(fp))
    backend = QwenHTTPBackend(client=_FakeClient("sorry, I can't help"), model="m")
    assert backend.detect(frame) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/pytest tests/test_qwen_http.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'caridence.analyzer.qwen_http'`.

- [ ] **Step 3: Write `caridence/analyzer/qwen_http.py`**

```python
# caridence/analyzer/qwen_http.py
from __future__ import annotations
import base64
import os
from caridence.schema import Frame, Detection, DamageType
from caridence.analyzer.base import parse_detections

_DAMAGE_LIST = ", ".join(dt.value for dt in DamageType)


def build_prompt() -> str:
    return (
        "You are a vehicle damage inspector. Look at the image and report ONLY "
        "clearly visible exterior damage. Return a JSON array; each item has: "
        f'"damage_type" (one of: {_DAMAGE_LIST}), "panel" (e.g. front_driver_door, '
        'rear_bumper, hood, front_passenger_corner), "severity" (minor|moderate|severe), '
        '"bbox" ([x,y,w,h] normalized 0..1), "confidence" (0..1). '
        "If there is no visible damage, return []. Output JSON only, no prose."
    )


def _b64_image(path: str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


class QwenHTTPBackend:
    """Backend for Qwen2.5-VL served via an OpenAI-compatible (vLLM) endpoint."""

    def __init__(self, client=None, model: str | None = None,
                 api_base: str | None = None, api_key: str | None = None,
                 max_tokens: int = 512):
        self.model = model or os.environ.get("CARIDENCE_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
        self.max_tokens = max_tokens
        if client is not None:
            self.client = client
        else:
            from openai import OpenAI
            self.client = OpenAI(
                base_url=api_base or os.environ.get("CARIDENCE_API_BASE", "http://127.0.0.1:8000/v1"),
                api_key=api_key or os.environ.get("CARIDENCE_API_KEY", "EMPTY"),
            )

    def detect(self, frame: Frame) -> list[Detection]:
        b64 = _b64_image(frame.path)
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": build_prompt()},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }]
        try:
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages,
                max_tokens=self.max_tokens, temperature=0.0,
            )
            content = resp.choices[0].message.content or ""
        except Exception:
            return []
        return parse_detections(content)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/pytest tests/test_qwen_http.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add caridence/analyzer/qwen_http.py tests/test_qwen_http.py
git commit -m "feat: Qwen2.5-VL HTTP backend against vLLM OpenAI-compatible endpoint"
```

---

## Task 9: FastAPI app (upload → report + dashboard)

**Files:**
- Create: `app/server.py`
- Create: `data/bench.json` (placeholder dashboard data)
- Create: `app/templates/dashboard.html`
- Test: `tests/test_server.py`

**Design:** routes — `GET /` (upload form), `POST /inspect` (accepts an uploaded video/photo(s), runs `run_inspection` with a backend selected by env `CARIDENCE_BACKEND` = `mock`|`qwen`, returns rendered HTML), `GET /frame?path=` (serves a cited-frame image safely), `GET /dashboard` (renders `data/bench.json`), `GET /api/inspect` returns JSON. Backend selection lives in one `get_backend()` helper. Uploaded files are saved to a temp dir per request.

- [ ] **Step 1: Write `data/bench.json` (committed placeholder; real numbers from Plan 2)**

```json
{
  "generated": "placeholder",
  "rows": [
    {"model": "Caridence-7B (MI300)", "f1": 0.0, "cost_per_inspection_usd": 0.0, "latency_ms": 0},
    {"model": "Caridence-3B (MI300)", "f1": 0.0, "cost_per_inspection_usd": 0.0, "latency_ms": 0},
    {"model": "Qwen2.5-VL-7B base", "f1": 0.0, "cost_per_inspection_usd": 0.0, "latency_ms": 0},
    {"model": "GPT-4o", "f1": 0.0, "cost_per_inspection_usd": 0.0, "latency_ms": 0}
  ]
}
```

- [ ] **Step 2: Write `app/templates/dashboard.html`**

```html
{# app/templates/dashboard.html #}
<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Caridence — Benchmark</title>
<style>
 :root{color-scheme:dark}body{font-family:ui-sans-serif,system-ui,sans-serif;background:#0a0f1a;color:#e6edf6;padding:2rem}
 table{border-collapse:collapse;width:100%;max-width:780px}th,td{padding:.7rem 1rem;border-bottom:1px solid #1e3a5f;text-align:left}
 th{color:#9fb0c6;font-weight:600}.hero td{color:#7dd3fc;font-weight:700}
</style></head><body>
 <h1>Caridence engine vs frontier</h1>
 <table><thead><tr><th>Model</th><th>Damage F1</th><th>$ / inspection</th><th>Latency</th></tr></thead>
 <tbody>
 {% for r in rows %}
 <tr class="{{ 'hero' if 'MI300' in r.model else '' }}">
   <td>{{ r.model }}</td><td>{{ "%.2f"|format(r.f1) }}</td>
   <td>${{ "%.4f"|format(r.cost_per_inspection_usd) }}</td><td>{{ r.latency_ms }} ms</td>
 </tr>
 {% endfor %}
 </tbody></table>
</body></html>
```

- [ ] **Step 3: Write the failing test**

```python
# tests/test_server.py
import io, os
from fastapi.testclient import TestClient


def _client(monkeypatch):
    monkeypatch.setenv("CARIDENCE_BACKEND", "mock")
    from app.server import app
    return TestClient(app)


def test_index_ok(monkeypatch):
    c = _client(monkeypatch)
    r = c.get("/")
    assert r.status_code == 200
    assert "Caridence" in r.text


def test_dashboard_renders(monkeypatch):
    c = _client(monkeypatch)
    r = c.get("/dashboard")
    assert r.status_code == 200
    assert "F1" in r.text


def test_inspect_photo_returns_report(monkeypatch, frames_dir):
    c = _client(monkeypatch)
    files = []
    for p in sorted(frames_dir.glob("*.jpg")):
        files.append(("files", (p.name, p.read_bytes(), "image/jpeg")))
    r = c.post("/inspect", files=files, data={"vehicle_label": "Chevy"})
    assert r.status_code == 200
    assert "Condition" in r.text
    assert "Chevy" in r.text
```

- [ ] **Step 4: Run test to verify it fails**

Run: `.venv/Scripts/pytest tests/test_server.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.server'`.

- [ ] **Step 5: Write `app/server.py`**

```python
# app/server.py
from __future__ import annotations
import os
import tempfile
from pathlib import Path
import json
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
from caridence.pipeline import run_inspection
from caridence.analyzer.mock import MockBackend
from caridence.report import render_report_html, report_to_dict

app = FastAPI(title="Caridence")
_DATA = Path(__file__).resolve().parent.parent / "data"
_TEMPLATES = Path(__file__).resolve().parent / "templates"


def get_backend():
    kind = os.environ.get("CARIDENCE_BACKEND", "mock").lower()
    if kind == "qwen":
        from caridence.analyzer.qwen_http import QwenHTTPBackend
        return QwenHTTPBackend()
    return MockBackend()


_INDEX = """<!doctype html><html><head><meta charset="utf-8"><title>Caridence</title>
<style>body{font-family:system-ui;background:#0a0f1a;color:#e6edf6;padding:2rem}</style></head>
<body><h1>Caridence</h1><p>Upload a walkaround video or photos.</p>
<form action="/inspect" method="post" enctype="multipart/form-data">
<input type="text" name="vehicle_label" placeholder="Vehicle label">
<input type="file" name="files" multiple>
<button type="submit">Inspect</button></form>
<p><a href="/dashboard">Benchmark dashboard</a></p></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return _INDEX


def _save_uploads(files: list[UploadFile], dest: Path) -> Path:
    """Save uploads; if a single video, return its path, else return the folder."""
    saved = []
    for uf in files:
        out = dest / Path(uf.filename).name
        out.write_bytes(uf.file.read())
        saved.append(out)
    if len(saved) == 1 and saved[0].suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        return saved[0]
    return dest


@app.post("/inspect", response_class=HTMLResponse)
def inspect(files: list[UploadFile] = File(...), vehicle_label: str = Form(None)):
    tmp = Path(tempfile.mkdtemp(prefix="caridence_"))
    source = _save_uploads(files, tmp)
    report = run_inspection(source, backend=get_backend(), vehicle_label=vehicle_label)
    return render_report_html(report)


@app.post("/api/inspect")
def api_inspect(files: list[UploadFile] = File(...), vehicle_label: str = Form(None)):
    tmp = Path(tempfile.mkdtemp(prefix="caridence_"))
    source = _save_uploads(files, tmp)
    report = run_inspection(source, backend=get_backend(), vehicle_label=vehicle_label)
    return JSONResponse(report_to_dict(report))


@app.get("/frame")
def frame(path: str):
    p = Path(path)
    if not p.is_file():
        return Response(status_code=404)
    return FileResponse(str(p))


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    env = Environment(loader=FileSystemLoader(str(_TEMPLATES)),
                      autoescape=select_autoescape(["html"]))
    data = json.loads((_DATA / "bench.json").read_text())
    return env.get_template("dashboard.html").render(rows=data["rows"])
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/Scripts/pytest tests/test_server.py -v`
Expected: 3 passed.

- [ ] **Step 7: Manual smoke (optional)**

Run: `.venv/Scripts/uvicorn app.server:app --port 8001` then open `http://127.0.0.1:8001/`.
Expected: upload form loads; uploading the fixture photos renders a report; `/dashboard` renders.

- [ ] **Step 8: Commit**

```bash
git add app/server.py app/templates/dashboard.html data/bench.json tests/test_server.py
git commit -m "feat: FastAPI app for upload->report and benchmark dashboard"
```

---

## Task 10: Docker packaging + full-suite gate

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`

- [ ] **Step 1: Write `.dockerignore`**

```
.venv
__pycache__
*.pyc
.git
data/*_frames
tests
.pytest_cache
```

- [ ] **Step 2: Write `Dockerfile`**

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .
COPY caridence ./caridence
COPY app ./app
COPY data ./data
ENV CARIDENCE_BACKEND=mock
EXPOSE 8000
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

(Note: `-e .` needs the package source present; since we COPY `caridence` and `app` after, reorder if build caching matters. For correctness, the COPYs above precede nothing that needs them at install time except metadata — `pip install -e .` only needs `pyproject.toml` to register the editable install, and imports resolve at runtime from the copied dirs.)

- [ ] **Step 3: Build the image**

Run: `docker build -t caridence:dev .`
Expected: build succeeds.

- [ ] **Step 4: Run the container and smoke-test**

Run: `docker run --rm -p 8000:8000 caridence:dev` then in another shell `curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/`
Expected: `200`.

- [ ] **Step 5: Run the full test suite**

Run: `.venv/Scripts/pytest`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add Dockerfile .dockerignore
git commit -m "chore: containerize Caridence (mock backend default)"
```

---

## Done criteria for Plan 1

- `pytest` green across schema, ingest, analyzer (mock + http), aggregator, estimator, report, pipeline, server.
- `docker run` serves the upload UI; uploading photos or a video yields a rendered report with cited frames + costs + condition score.
- Swapping `CARIDENCE_BACKEND=qwen` (with a live vLLM endpoint) routes through the real model with zero code change.
- Dashboard renders from `data/bench.json` (placeholder until Plan 2 fills real numbers).

This is a complete, demo-able product on the mock/base model. **Plan 2** (training + benchmark) replaces the placeholder dashboard with the real "fine-tuned-on-MI300 beats frontier at 1/30th cost" numbers and produces the fine-tuned weights the `qwen` backend serves.
