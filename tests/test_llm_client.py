import pytest

from app.llm.client import LLMClient, LLMError, Usage, extract_json


class FakeResp:
    def __init__(self, status, payload):
        self.status_code, self._p = status, payload

    def json(self):
        return self._p


class FakeSession:
    def __init__(self, resp):
        self.resp, self.sent = resp, []

    def post(self, url, json, timeout, headers):
        self.sent.append({"url": url, "json": json, "headers": headers})
        return self.resp


def test_openai_request_shape_and_usage(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    c = LLMClient("openai", "gpt-4.1-mini")
    c._session = FakeSession(FakeResp(200, {"choices": [{"message": {"content": '{"a": 1}'}}],
                                            "usage": {"prompt_tokens": 1000, "completion_tokens": 100}}))
    u = Usage()
    assert c.complete_json("sys", "user", agent="document_reader", usage=u, run_id="r") == {"a": 1}
    body = c._session.sent[0]["json"]
    assert body["temperature"] == 0 and body["response_format"] == {"type": "json_object"}
    assert u.input_tokens == 1000 and u.cost_usd == pytest.approx((1000 * 0.40 + 100 * 1.60) / 1e6)


def test_http_error_and_mock_raise():
    c = LLMClient("openai", "gpt-4.1-mini")
    c._session = FakeSession(FakeResp(500, {}))
    with pytest.raises(LLMError):
        c.complete_json("s", "u", agent="x", usage=Usage(), run_id="r")
    with pytest.raises(LLMError):
        LLMClient("mock", "mock").complete_json("s", "u", agent="x", usage=Usage(), run_id="r")


def test_extract_json():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Here: {"a": {"b": 2}} done') == {"a": {"b": 2}}
