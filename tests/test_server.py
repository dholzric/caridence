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
    assert "Recall" in r.text


def test_frame_rejects_outside_temp(monkeypatch):
    import os
    c = _client(monkeypatch)
    # pyproject.toml exists in the project root, which is outside the temp dir
    project_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pyproject.toml"))
    r = c.get(f"/frame?path={project_file}")
    assert r.status_code == 404


def test_inspect_photo_returns_report(monkeypatch, frames_dir):
    c = _client(monkeypatch)
    files = []
    for p in sorted(frames_dir.glob("*.jpg")):
        files.append(("files", (p.name, p.read_bytes(), "image/jpeg")))
    r = c.post("/inspect", files=files, data={"vehicle_label": "Chevy"})
    assert r.status_code == 200
    assert "Condition" in r.text
    assert "Chevy" in r.text
