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
