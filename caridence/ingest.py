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
