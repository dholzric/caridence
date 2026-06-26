# tests/test_hybrid.py
from caridence.analyzer.hybrid import HybridBackend, VLMVerifier
from caridence.schema import Frame, Detection, DamageType, Severity, BBox


def _det(dt):
    return Detection(damage_type=dt, panel="left_upper", severity=Severity.MINOR,
                     bbox=BBox(x=0.1, y=0.1, w=0.2, h=0.2), confidence=0.6)


class _StubDetector:
    def __init__(self, dets): self._dets = dets
    def detect(self, frame): return list(self._dets)


class _RejectScratch:
    def verify(self, frame, det): return det.damage_type != DamageType.SCRATCH


def test_hybrid_no_verifier_is_passthrough():
    det = _StubDetector([_det(DamageType.DENT), _det(DamageType.SCRATCH)])
    h = HybridBackend(detector=det)
    out = h.detect(Frame(index=0, timestamp=0.0, path="x.jpg"))
    assert len(out) == 2


def test_hybrid_verifier_filters_false_positives():
    det = _StubDetector([_det(DamageType.DENT), _det(DamageType.SCRATCH)])
    h = HybridBackend(detector=det, verifier=_RejectScratch())
    out = h.detect(Frame(index=0, timestamp=0.0, path="x.jpg"))
    assert [d.damage_type for d in out] == [DamageType.DENT]


# --- VLMVerifier (crop -> ask VLM yes/no), with an injected fake client ---

class _FakeMsg:
    def __init__(self, content): self.message = type("M", (), {"content": content})
class _FakeResp:
    def __init__(self, content): self.choices = [_FakeMsg(content)]
class _FakeCompletions:
    def __init__(self, content): self._c = content; self.last = None
    def create(self, **kwargs): self.last = kwargs; return _FakeResp(self._c)
class _FakeClient:
    def __init__(self, content):
        self.chat = type("C", (), {"completions": _FakeCompletions(content)})()


def _frame_with_image(tmp_path):
    import numpy as np, cv2
    p = tmp_path / "car.jpg"
    cv2.imwrite(str(p), np.zeros((80, 120, 3), dtype=np.uint8))
    return Frame(index=0, timestamp=0.0, path=str(p))


def test_vlm_verifier_yes(tmp_path):
    v = VLMVerifier(client=_FakeClient("Yes, there is a dent."), model="m")
    assert v.verify(_frame_with_image(tmp_path), _det(DamageType.DENT)) is True


def test_vlm_verifier_no(tmp_path):
    v = VLMVerifier(client=_FakeClient("No, the panel is clean."), model="m")
    assert v.verify(_frame_with_image(tmp_path), _det(DamageType.DENT)) is False


def test_vlm_verifier_sends_image(tmp_path):
    v = VLMVerifier(client=_FakeClient("yes"), model="m")
    v.verify(_frame_with_image(tmp_path), _det(DamageType.DENT))
    sent = v.client.chat.completions.last
    assert sent["model"] == "m"
    assert any("image_url" in str(part) for part in sent["messages"][0]["content"])
