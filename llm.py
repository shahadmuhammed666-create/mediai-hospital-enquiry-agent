from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


Message = Mapping[str, str]


class LLMClient(ABC):
    @abstractmethod
    def complete(self, messages: Sequence[Message]) -> str:
        """Return a model response for the supplied conversation."""


class FakeLLM(LLMClient):
    def __init__(self, scripted_responses: Iterable[str]) -> None:
        responses = list(scripted_responses)
        if not all(isinstance(response, str) for response in responses):
            raise TypeError("Every scripted response must be a string.")
        self._responses = deque(responses)

    def complete(self, messages: Sequence[Message]) -> str:
        if not self._responses:
            raise RuntimeError("FakeLLM has no scripted responses left.")
        return self._responses.popleft()


class RealLLM(LLMClient):
    API_URL = "https://api.openai.com/v1/chat/completions"
    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(self, model: str | None = None, timeout: float = 30.0) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY to use RealLLM.")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero.")

        self._api_key = api_key
        self._model = model or os.environ.get("OPENAI_MODEL") or self.DEFAULT_MODEL
        self._timeout = timeout

    def complete(self, messages: Sequence[Message]) -> str:
        request_messages = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                raise ValueError("Every message must contain string role and content fields.")
            request_messages.append({"role": role, "content": content})

        body = json.dumps(
            {
                "model": self._model,
                "messages": request_messages,
                "response_format": {"type": "json_object"},
            }
        ).encode("utf-8")
        request = Request(
            self.API_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self._timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except TimeoutError as error:
            raise RuntimeError(
                f"OpenAI API request timed out after {self._timeout:g} seconds. "
                "Check your connection or retry with a longer --timeout."
            ) from error
        except HTTPError as error:
            raise RuntimeError(
                f"OpenAI API returned HTTP {error.code}."
            ) from error
        except URLError as error:
            raise RuntimeError(f"Could not reach the OpenAI API: {error.reason}") from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("OpenAI API returned an invalid JSON response.") from error

        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("OpenAI API response did not contain a message.") from error
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("OpenAI API returned an empty message.")
        return content


class GeminiLLM(LLMClient):
    API_URL = "https://generativelanguage.googleapis.com/v1beta/models"
    DEFAULT_MODEL = "gemini-3.5-flash-lite"

    def __init__(self, model: str | None = None, timeout: float = 30.0) -> None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Set GEMINI_API_KEY to use GeminiLLM.")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero.")

        self._api_key = api_key
        self._model = model or os.environ.get("GEMINI_MODEL") or self.DEFAULT_MODEL
        self._timeout = timeout

    def complete(self, messages: Sequence[Message]) -> str:
        system_parts: list[str] = []
        contents: list[dict[str, object]] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                raise ValueError("Every message must contain string role and content fields.")
            if role == "system":
                system_parts.append(content)
                continue
            if role == "assistant":
                role = "model"
            elif role not in ("user", "model"):
                raise ValueError(f"Gemini does not support the message role {role!r}.")

            if contents and contents[-1]["role"] == role:
                parts = contents[-1]["parts"]
                if isinstance(parts, list) and parts:
                    parts[0]["text"] += f"\n\n{content}"
            else:
                contents.append({"role": role, "parts": [{"text": content}]})

        request_body: dict[str, object] = {
            "contents": contents,
            "generationConfig": {"responseMimeType": "application/json"},
        }
        if system_parts:
            request_body["systemInstruction"] = {
                "parts": [{"text": "\n\n".join(system_parts)}]
            }

        request = Request(
            f"{self.API_URL}/{quote(self._model, safe='')}:generateContent",
            data=json.dumps(request_body).encode("utf-8"),
            headers={
                "x-goog-api-key": self._api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self._timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except TimeoutError as error:
            raise RuntimeError(
                f"Gemini API request timed out after {self._timeout:g} seconds. "
                "Check your connection or retry with a longer --timeout."
            ) from error
        except HTTPError as error:
            detail = _api_error_detail(error)
            suffix = f": {detail}" if detail else "."
            raise RuntimeError(
                f"Gemini API returned HTTP {error.code}{suffix}"
            ) from error
        except URLError as error:
            raise RuntimeError(f"Could not reach the Gemini API: {error.reason}") from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("Gemini API returned an invalid JSON response.") from error

        try:
            parts = payload["candidates"][0]["content"]["parts"]
            content = "".join(part["text"] for part in parts if isinstance(part.get("text"), str))
        except (KeyError, IndexError, TypeError, AttributeError) as error:
            raise RuntimeError("Gemini API response did not contain a text message.") from error
        if not content.strip():
            raise RuntimeError("Gemini API returned an empty message.")
        return content


def _api_error_detail(error: HTTPError) -> str | None:
    try:
        payload = json.loads(error.read().decode("utf-8"))
        detail = payload["error"]["message"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
        return None
    return detail if isinstance(detail, str) and detail else None
