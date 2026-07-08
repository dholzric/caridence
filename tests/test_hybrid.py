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


def test_vlm_verifier_reasoning_model_answer_at_end(tmp_path):
    # Reasoning models (e.g. Kimi K2.6 on Fireworks) emit thinking text before
    # the answer; the verdict is the LAST yes/no in the reply.
    content = ("The user wants me to check this crop for a dent. There is no "
               "obvious deformation at first, but looking closely at the "
               "shading there is a depression. Final answer: yes.")
    v = VLMVerifier(client=_FakeClient(content), model="m")
    assert v.verify(_frame_with_image(tmp_path), _det(DamageType.DENT)) is True


def test_vlm_verifier_reasoning_model_no_at_end(tmp_path):
    content = ("Hmm, could this be a scratch? Yes, there is a line — but it "
               "is a reflection of the roofline. Answer: no.")
    v = VLMVerifier(client=_FakeClient(content), model="m")
    assert v.verify(_frame_with_image(tmp_path), _det(DamageType.DENT)) is False


def test_vlm_verifier_unparseable_reply_fails_open(tmp_path):
    # No yes/no anywhere -> keep the candidate (favor recall).
    v = VLMVerifier(client=_FakeClient("I cannot determine that."), model="m")
    assert v.verify(_frame_with_image(tmp_path), _det(DamageType.DENT)) is True


def test_vlm_verifier_max_tokens_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CARIDENCE_VERIFY_MAX_TOKENS", "777")
    v = VLMVerifier(client=_FakeClient("yes"), model="m")
    v.verify(_frame_with_image(tmp_path), _det(DamageType.DENT))
    assert v.client.chat.completions.last["max_tokens"] == 777


def test_vlm_verifier_reasoning_effort_env(tmp_path, monkeypatch):
    # Reasoning models on Fireworks accept reasoning_effort (e.g. "none") to
    # skip thinking; pass it through when configured.
    monkeypatch.setenv("CARIDENCE_VERIFY_REASONING_EFFORT", "none")
    v = VLMVerifier(client=_FakeClient("yes"), model="m")
    v.verify(_frame_with_image(tmp_path), _det(DamageType.DENT))
    assert v.client.chat.completions.last["extra_body"] == {"reasoning_effort": "none"}


def test_vlm_verifier_no_reasoning_effort_by_default(tmp_path):
    v = VLMVerifier(client=_FakeClient("yes"), model="m")
    v.verify(_frame_with_image(tmp_path), _det(DamageType.DENT))
    assert "extra_body" not in v.client.chat.completions.last
