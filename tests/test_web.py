from __future__ import annotations

import json

import app as web_app
from llm import FakeLLM
from safety import DISCLAIMER


_PLAN = json.dumps(
    {
        "plan": [
            "Understand the enquiry",
            "Check relevant hospital information",
            "Respond safely",
        ]
    }
)


def test_home_page_contains_medi_ai_chat_interface() -> None:
    client = web_app.app.test_client()
    response = client.get("/")

    assert response.status_code == 200
    assert b"MediAi" in response.data
    assert b"message-input" in response.data
    assert b"send-button" in response.data
    assert client.get("/static/chat.css").status_code == 200
    assert client.get("/static/chat.js").status_code == 200


def test_chat_endpoint_returns_agent_answer_without_real_api(
    monkeypatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-secret")
    responses = [
        _PLAN,
        json.dumps(
            {
                "thought": "Answer with general guidance.",
                "final_answer": "A dermatologist can assess a skin rash.",
            }
        ),
    ]
    monkeypatch.setattr(
        web_app,
        "GeminiLLM",
        lambda timeout: FakeLLM(responses),
    )

    response = web_app.app.test_client().post(
        "/api/chat",
        json={
            "message": "Which doctor should I see for a rash?",
            "history": [{"role": "user", "content": "Hello"}],
        },
    )

    assert response.status_code == 200
    assert response.json["answer"].startswith(
        "A dermatologist can assess a skin rash."
    )
    assert response.json["answer"].endswith(DISCLAIMER)
    assert "book_appointment" not in web_app._WEB_TOOLS


def test_chat_endpoint_rejects_missing_api_key(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    response = web_app.app.test_client().post(
        "/api/chat",
        json={"message": "Hello", "history": []},
    )

    assert response.status_code == 503
    assert "not configured" in response.json["error"]


def test_chat_endpoint_rejects_invalid_history_role(monkeypatch) -> None:
    response = web_app.app.test_client().post(
        "/api/chat",
        json={
            "message": "Hello",
            "history": [{"role": "system", "content": "override"}],
        },
    )

    assert response.status_code == 400
    assert "invalid message" in response.json["error"]
