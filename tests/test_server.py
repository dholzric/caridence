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
