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
    import tempfile
    allowed = Path(tempfile.gettempdir()).resolve()
    p = Path(path).resolve()
    if allowed not in p.parents or not p.is_file():
        return Response(status_code=404)
    return FileResponse(str(p))


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    env = Environment(loader=FileSystemLoader(str(_TEMPLATES)),
                      autoescape=select_autoescape(["html"]))
    data = json.loads((_DATA / "bench.json").read_text())
    return env.get_template("dashboard.html").render(rows=data["rows"])
