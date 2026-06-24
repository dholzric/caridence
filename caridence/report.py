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
