from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MemoryStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self._path = (
            Path(path) if path is not None else Path(__file__).parent / "memory" / "memory.json"
        )

    def load_memory(self, patient_id: str) -> dict[str, Any]:
        patient_id = self._validate_patient_id(patient_id)
        if not self._path.exists():
            return self._empty_record()

        with self._path.open(encoding="utf-8") as memory_file:
            data = json.load(memory_file)
        if not isinstance(data, dict):
            raise ValueError("Memory file must contain a JSON object.")
        record = data.get(patient_id, {})
        if not isinstance(record, dict):
            raise ValueError(f"Memory record for {patient_id} must be an object.")
        return {**self._empty_record(), **record}

    def save_summary(self, patient_id: str, summary: dict[str, Any]) -> None:
        patient_id = self._validate_patient_id(patient_id)
        if not self._path.exists():
            data: dict[str, Any] = {}
        else:
            with self._path.open(encoding="utf-8") as memory_file:
                data = json.load(memory_file)
            if not isinstance(data, dict):
                raise ValueError("Memory file must contain a JSON object.")

        existing_record = data.get(patient_id, {})
        if not isinstance(existing_record, dict):
            raise ValueError(f"Memory record for {patient_id} must be an object.")
        record = {**self._empty_record(), **existing_record}
        if summary.get("first_name"):
            record["first_name"] = self._first_name(summary["first_name"])
        for field in ("past_symptoms", "recommended_specialties", "appointments_booked"):
            values = summary.get(field, [])
            if not isinstance(values, list):
                raise ValueError(f"{field} must be a list.")
            record[field] = self._append_unique(record[field], values)
        preferences = summary.get("preferences", {})
        if not isinstance(preferences, dict):
            raise ValueError("preferences must be an object.")
        if not isinstance(record["preferences"], dict):
            raise ValueError("Stored preferences have an invalid format.")
        record["preferences"].update(preferences)
        data[patient_id] = record

        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as memory_file:
            json.dump(data, memory_file, ensure_ascii=True, indent=2)

    def clear_memory(self, patient_id: str) -> None:
        patient_id = self._validate_patient_id(patient_id)
        if not self._path.exists():
            return
        with self._path.open(encoding="utf-8") as memory_file:
            data = json.load(memory_file)
        if not isinstance(data, dict):
            raise ValueError("Memory file must contain a JSON object.")
        data.pop(patient_id, None)
        with self._path.open("w", encoding="utf-8") as memory_file:
            json.dump(data, memory_file, ensure_ascii=True, indent=2)

    @staticmethod
    def _validate_patient_id(patient_id: str) -> str:
        if not isinstance(patient_id, str) or not patient_id.strip():
            raise ValueError("patient_id must be a non-empty string.")
        return patient_id.strip()

    @staticmethod
    def _first_name(name: str) -> str:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("first_name must be a non-empty string.")
        return name.strip().split()[0]

    @staticmethod
    def _empty_record() -> dict[str, Any]:
        return {
            "first_name": None,
            "past_symptoms": [],
            "recommended_specialties": [],
            "appointments_booked": [],
            "preferences": {},
        }

    @staticmethod
    def _append_unique(current: list[Any], additions: list[Any]) -> list[Any]:
        if not isinstance(current, list):
            raise ValueError("Stored memory list has an invalid format.")
        result = list(current)
        for item in additions:
            if item not in result:
                result.append(item)
        return result


def clear_memory(patient_id: str) -> None:
    MemoryStore().clear_memory(patient_id)
