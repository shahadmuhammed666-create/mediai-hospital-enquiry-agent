from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JsonLinesLogger:
    def __init__(self, path: str | Path | None = None) -> None:
        self._path = (
            Path(path) if path is not None else Path(__file__).parent / "logs" / "agent.log"
        )

    def log(
        self,
        task_id: str,
        event: str,
        *,
        sensitive_values: tuple[str, ...] = (),
        **details: Any,
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task_id": task_id,
            "event": event,
            **self._sanitize(details, sensitive_values),
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(entry, ensure_ascii=True, default=str) + "\n")

    @classmethod
    def _sanitize(cls, value: Any, sensitive_values: tuple[str, ...]) -> Any:
        if isinstance(value, dict):
            sanitized = {}
            for key, item in value.items():
                if key.casefold() in {"patient_name", "patient_full_name"}:
                    sanitized[key] = cls._first_name(item)
                else:
                    sanitized[key] = cls._sanitize(item, sensitive_values)
            return sanitized
        if isinstance(value, list):
            return [cls._sanitize(item, sensitive_values) for item in value]
        if isinstance(value, tuple):
            return [cls._sanitize(item, sensitive_values) for item in value]
        if isinstance(value, str):
            for sensitive_value in sensitive_values:
                if sensitive_value.strip():
                    first_name = cls._first_name(sensitive_value)
                    value = re.sub(
                        re.escape(sensitive_value.strip()),
                        first_name,
                        value,
                        flags=re.IGNORECASE,
                    )
            return value
        return value

    @staticmethod
    def _first_name(value: Any) -> str | None:
        if isinstance(value, str) and value.strip():
            return value.strip().split()[0]
        return None


class NullLogger(JsonLinesLogger):
    def __init__(self) -> None:
        pass

    def log(
        self,
        task_id: str,
        event: str,
        *,
        sensitive_values: tuple[str, ...] = (),
        **details: Any,
    ) -> None:
        return None
