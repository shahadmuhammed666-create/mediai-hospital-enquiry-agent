from __future__ import annotations

import io
import json
import os

import pytest

import llm
from llm import GeminiLLM


def test_gemini_llm_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiLLM()


def test_gemini_llm_sends_request_and_returns_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-secret")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["api_key"] = request.get_header("X-goog-api-key")
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return io.BytesIO(
            json.dumps(
                {"candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}]}
            ).encode("utf-8")
        )

    monkeypatch.setattr(llm, "urlopen", fake_urlopen)
    result = GeminiLLM(timeout=5).complete(
        [
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": "Return the result."},
            {"role": "user", "content": "Use the requested format."},
            {"role": "assistant", "content": '{"step": 1}'},
        ]
    )

    assert result == '{"ok": true}'
    assert captured["url"] == f"{GeminiLLM.API_URL}/test-model:generateContent"
    assert captured["api_key"] == "test-key-not-a-secret"
    assert captured["timeout"] == 5
    assert captured["body"] == {
        "systemInstruction": {"parts": [{"text": "Return JSON only."}]},
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": "Return the result.\n\nUse the requested format."}
                ],
            },
            {"role": "model", "parts": [{"text": '{"step": 1}'}]},
        ],
        "generationConfig": {"responseMimeType": "application/json"},
    }


def test_gemini_llm_includes_api_error_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from urllib.error import HTTPError

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-secret")

    def fake_urlopen(request, timeout):
        assert timeout == 30.0
        raise HTTPError(
            request.full_url,
            429,
            "Too Many Requests",
            {},
            io.BytesIO(b'{"error":{"message":"Quota exceeded."}}'),
        )

    monkeypatch.setattr(llm, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="HTTP 429: Quota exceeded"):
        GeminiLLM().complete([{"role": "user", "content": "Hello"}])


def test_gemini_llm_reports_read_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-secret")

    def fake_urlopen(request, timeout):
        raise TimeoutError("The read operation timed out.")

    monkeypatch.setattr(llm, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="timed out after 5 seconds.*--timeout"):
        GeminiLLM(timeout=5).complete([{"role": "user", "content": "Hello"}])


@pytest.mark.real_llm
@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY is not set",
)
def test_gemini_live_smoke() -> None:
    client = GeminiLLM()
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
