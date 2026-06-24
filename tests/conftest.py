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
