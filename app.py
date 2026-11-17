from __future__ import annotations

import os
from collections.abc import Mapping

from flask import Flask, jsonify, render_template, request

from agent import Agent
from llm import RealLLM
from logger import NullLogger
from tools import TOOLS, ToolDefinition


_WEB_TOOLS: dict[str, ToolDefinition] = {
    name: tool for name, tool in TOOLS.items() if name != "book_appointment"
}
_MAX_MESSAGE_LENGTH = 1200
_MAX_HISTORY_MESSAGES = 8
_MAX_HISTORY_CONTENT_LENGTH = 6000

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024


@app.get("/")
def home() -> str:
    return render_template("index.html")


@app.get("/healthz")
def health_check():
    return jsonify(status="ok")


@app.post("/api/chat")
def chat():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Send a valid JSON request."), 400

    message = payload.get("message")
    if not isinstance(message, str) or not message.strip():
        return jsonify(error="Please enter a question."), 400
    message = message.strip()
    if len(message) > _MAX_MESSAGE_LENGTH:
        return jsonify(error=f"Please keep your question under {_MAX_MESSAGE_LENGTH} characters."), 400

    try:
        history = _validated_history(payload.get("history", []))
    except ValueError as error:
        return jsonify(error=str(error)), 400

    if not os.environ.get("OPENAI_API_KEY"):
        return jsonify(error="MediAi is not configured with an OpenAI API key yet."), 503

    try:
        client = RealLLM(timeout=120.0)
        agent = Agent(
            client,
            tools=_WEB_TOOLS,
            approval_handler=lambda _: False,
            logger=NullLogger(),
        )
        answer = agent.run(
            message,
            memory={"recent_conversation": history},
        )
    except (RuntimeError, ValueError, OSError) as error:
        return jsonify(error=f"MediAi couldn't complete that request: {error}"), 502

    return jsonify(answer=answer)


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify(error="The request is too large."), 413


@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    return response


def _validated_history(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError("Conversation history must be a list.")

    history: list[dict[str, str]] = []
    for entry in value[-_MAX_HISTORY_MESSAGES:]:
        if not isinstance(entry, Mapping):
            raise ValueError("Each history item must contain a role and message.")
        role = entry.get("role")
        content = entry.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            raise ValueError("Conversation history contains an invalid message.")
        if len(content) > _MAX_HISTORY_CONTENT_LENGTH:
            raise ValueError("A conversation history message is too long.")
        if content.strip():
            history.append({"role": role, "content": content.strip()})
    return history


if __name__ == "__main__":
    app.run()
