from __future__ import annotations

import io
import json
import os

import pytest

import llm
from llm import RealLLM


def test_real_llm_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        RealLLM()


def test_real_llm_sends_json_chat_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-a-secret")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return io.BytesIO(
            json.dumps(
                {"choices": [{"message": {"content": '{"ok": true}'}}]}
            ).encode("utf-8")
        )

    monkeypatch.setattr(llm, "urlopen", fake_urlopen)
    result = RealLLM(timeout=5).complete(
        [{"role": "user", "content": "Return JSON with ok set to true."}]
    )

    assert result == '{"ok": true}'
    assert captured["url"] == RealLLM.API_URL
    assert captured["authorization"] == "Bearer test-key-not-a-secret"
    assert captured["timeout"] == 5
    request_body = captured["body"]
    assert request_body["model"] == "test-model"
    assert request_body["response_format"] == {"type": "json_object"}
    assert request_body["messages"] == [
        {"role": "user", "content": "Return JSON with ok set to true."}
    ]


def test_real_llm_reports_read_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-a-secret")

    def fake_urlopen(request, timeout):
        raise TimeoutError("The read operation timed out.")

    monkeypatch.setattr(llm, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="timed out after 5 seconds.*--timeout"):
        RealLLM(timeout=5).complete([{"role": "user", "content": "Hello"}])


@pytest.mark.real_llm
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY is not set",
)
def test_openai_live_smoke() -> None:
    client = RealLLM()
    response = client.complete(
        [
            {
                "role": "system",
                "content": "Reply with a JSON object only.",
            },
            {
                "role": "user",
                "content": 'Return exactly {"ok": true}. This is a connectivity check.',
            },
        ]
    )

    assert json.loads(response).get("ok") is True
