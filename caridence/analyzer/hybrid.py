# caridence/analyzer/hybrid.py
"""Hybrid detection: a high-recall detector proposes candidates, a verifier
(typically a VLM) confirms each one to restore precision.

This is the product architecture: the detector catches ~98% of damage at the
vehicle level; the verifier crops each candidate and confirms it, killing the
detector's false positives while keeping its recall.
"""
from __future__ import annotations
import base64
import os
import re
from typing import Protocol
from caridence.schema import Frame, Detection


class Verifier(Protocol):
    def verify(self, frame: Frame, det: Detection) -> bool:
        ...


class HybridBackend:
    """detector.detect -> candidates -> verifier.verify filter.

    With no verifier it is a pure high-recall detector pass-through.
    """

    def __init__(self, detector, verifier: Verifier | None = None):
        self.detector = detector
        self.verifier = verifier

    def detect(self, frame: Frame) -> list[Detection]:
        candidates = self.detector.detect(frame)
        if self.verifier is None:
            return candidates
        return [d for d in candidates if self.verifier.verify(frame, d)]


def _crop_b64(image_path: str, det: Detection, pad: float = 0.08) -> str:
    """Crop the image to the detection bbox (with padding) -> base64 JPEG."""
    import cv2
    img = cv2.imread(image_path)
    H, W = img.shape[:2]
    b = det.bbox
    x1 = max(0, int((b.x - pad) * W))
    y1 = max(0, int((b.y - pad) * H))
    x2 = min(W, int((b.x + b.w + pad) * W))
    y2 = min(H, int((b.y + b.h + pad) * H))
    crop = img if (x2 <= x1 or y2 <= y1) else img[y1:y2, x1:x2]
    ok, buf = cv2.imencode(".jpg", crop)
    return base64.b64encode(buf.tobytes()).decode("ascii")


class VLMVerifier:
    """Confirms a candidate by showing the VLM the cropped region and asking a
    yes/no question. Uses an OpenAI-compatible endpoint (vLLM serving the VLM)."""

    def __init__(self, client=None, model: str | None = None,
                 api_base: str | None = None, api_key: str | None = None,
                 pad: float = 0.08, max_tokens: int | None = None):
        self.model = model or os.environ.get("CARIDENCE_VERIFY_MODEL",
                                             "Qwen/Qwen2.5-VL-7B-Instruct")
        self.pad = pad
        # Reasoning models think out loud before answering, so leave room for
        # the answer to arrive; plain VLMs stop after one token anyway.
        self.max_tokens = max_tokens if max_tokens is not None else int(
            os.environ.get("CARIDENCE_VERIFY_MAX_TOKENS", "512"))
        # e.g. "none" to disable thinking on Fireworks reasoning models.
        self.reasoning_effort = os.environ.get("CARIDENCE_VERIFY_REASONING_EFFORT")
        if client is not None:
            self.client = client
        else:
            from openai import OpenAI
            self.client = OpenAI(
                base_url=api_base or os.environ.get("CARIDENCE_API_BASE", "http://127.0.0.1:8000/v1"),
                api_key=api_key or os.environ.get("CARIDENCE_API_KEY", "EMPTY"))

    def verify(self, frame: Frame, det: Detection) -> bool:
        b64 = _crop_b64(frame.path, det, self.pad)
        question = (
            f"This is a close-up crop of a vehicle. Is there clearly visible "
            f"{det.damage_type.value.replace('_', ' ')} damage in this image? "
            f"Answer with only 'yes' or 'no'.")
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }]
        kwargs = {}
        if self.reasoning_effort:
            kwargs["extra_body"] = {"reasoning_effort": self.reasoning_effort}
        try:
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages,
                max_tokens=self.max_tokens, temperature=0.0, **kwargs)
            answer = (resp.choices[0].message.content or "").strip().lower()
        except Exception:
            # On verifier failure, keep the candidate (favor recall).
            return True
        # The verdict is the LAST yes/no: reasoning models emit thinking text
        # ("could this be a scratch? Yes, but...") before the final answer.
        verdicts = re.findall(r"\b(yes|no)\b", answer)
        if not verdicts:
            return True  # unparseable -> keep the candidate (favor recall)
        return verdicts[-1] == "yes"
