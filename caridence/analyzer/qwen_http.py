# caridence/analyzer/qwen_http.py
from __future__ import annotations
import base64
import logging
import mimetypes
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
        mime = mimetypes.guess_type(frame.path)[0] or "image/jpeg"
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": build_prompt()},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ],
        }]
        try:
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages,
                max_tokens=self.max_tokens, temperature=0.0,
            )
            content = resp.choices[0].message.content or ""
        except Exception as exc:
            logging.warning("QwenHTTPBackend.detect failed for %s: %s", frame.path, exc)
            return []
        return parse_detections(content)
