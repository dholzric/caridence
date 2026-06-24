# tests/test_qwen_http.py
from caridence.analyzer.qwen_http import QwenHTTPBackend, build_prompt
from caridence.schema import Frame, DamageType
import numpy as np, cv2


class _FakeMessage:
    def __init__(self, content): self.message = type("M", (), {"content": content})
class _FakeResp:
    def __init__(self, content): self.choices = [_FakeMessage(content)]
class _FakeCompletions:
    def __init__(self, content): self._c = content; self.last_kwargs = None
    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeResp(self._c)
class _FakeClient:
    def __init__(self, content):
        self.chat = type("C", (), {"completions": _FakeCompletions(content)})()


def test_build_prompt_lists_damage_types():
    p = build_prompt()
    for dt in DamageType:
        assert dt.value in p


def test_backend_parses_model_json(tmp_path):
    img = np.zeros((40, 60, 3), dtype=np.uint8)
    fp = tmp_path / "f.jpg"; cv2.imwrite(str(fp), img)
    frame = Frame(index=0, timestamp=0.0, path=str(fp))
    content = '[{"damage_type":"dent","panel":"hood","severity":"moderate","bbox":[0.1,0.2,0.3,0.4],"confidence":0.9}]'
    backend = QwenHTTPBackend(client=_FakeClient(content), model="test-model")
    dets = backend.detect(frame)
    assert len(dets) == 1 and dets[0].damage_type == DamageType.DENT
    # confirm an image payload was sent
    sent = backend.client.chat.completions.last_kwargs
    assert sent["model"] == "test-model"
    assert any("image_url" in str(part) for part in sent["messages"][0]["content"])


def test_backend_tolerates_bad_output(tmp_path):
    img = np.zeros((40, 60, 3), dtype=np.uint8)
    fp = tmp_path / "f.jpg"; cv2.imwrite(str(fp), img)
    frame = Frame(index=0, timestamp=0.0, path=str(fp))
    backend = QwenHTTPBackend(client=_FakeClient("sorry, I can't help"), model="m")
    assert backend.detect(frame) == []
