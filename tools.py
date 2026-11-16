from __future__ import annotations

import json
from datetime import date as Date, timedelta
from pathlib import Path
from typing import Callable, TypedDict
from uuid import uuid4


_DATA_DIR = Path(__file__).parent / "data"
_booked_slots: set[tuple[str, str, str]] = set()


class ToolDefinition(TypedDict):
    name: str
    description: str
    args: dict[str, str]
    returns: str
    function: Callable[..., object]


def _load_json(filename: str) -> object:
    with (_DATA_DIR / filename).open(encoding="utf-8") as data_file:
        return json.load(data_file)


def _require_text(value: str, argument_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{argument_name} must be a non-empty string.")
    return value.strip()


def _parse_date(value: str) -> Date:
    try:
        parsed_date = Date.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise ValueError("date must be a valid date in YYYY-MM-DD format.") from error
    if parsed_date.isoformat() != value:
        raise ValueError("date must be in YYYY-MM-DD format.")
    return parsed_date


def symptom_to_specialty(symptoms: str) -> dict[str, str]:
    """Map symptom keywords to a specialty and urgency, without diagnosing."""
    symptom_text = _require_text(symptoms, "symptoms").casefold()
    rules_data = _load_json("symptom_rules.json")
    if not isinstance(rules_data, dict):
        raise ValueError("Symptom rules data must be a JSON object.")

    for urgency in ("emergency", "urgent"):
        keywords = rules_data.get(f"{urgency}_keywords", [])
        if any(keyword.casefold() in symptom_text for keyword in keywords):
            return {
                "specialty": "Emergency Medicine",
                "urgency": urgency,
                "general_info": (
                    "These symptoms may need prompt medical attention. "
                    "A healthcare professional can assess them."
                ),
            }

    rules = rules_data.get("rules", [])
    for rule in rules:
        if any(keyword.casefold() in symptom_text for keyword in rule["keywords"]):
            return {
                "specialty": rule["specialty"],
                "urgency": "routine",
                "general_info": rule["general_info"],
            }

    return {
        "specialty": "General Practice",
        "urgency": "routine",
        "general_info": (
            "Symptoms can have many causes. A healthcare professional can "
            "assess your concerns and recommend next steps."
        ),
    }


def find_doctors(specialty: str) -> list[dict[str, str]]:
    """List mock doctors whose specialty matches, ignoring case."""
    requested_specialty = _require_text(specialty, "specialty").casefold()
    doctors = _load_json("doctors.json")
    if not isinstance(doctors, list):
        raise ValueError("Doctor data must be a JSON list.")
    return [
        {
            "doctor_id": doctor["doctor_id"],
            "name": doctor["name"],
            "specialty": doctor["specialty"],
        }
        for doctor in doctors
        if doctor["specialty"].casefold() == requested_specialty
    ]


def check_availability(
    doctor_id: str, date: str | None = None
) -> list[dict[str, str]]:
    """Return a doctor's open slots for a date or the next three days."""
    requested_doctor = _require_text(doctor_id, "doctor_id")
    if not any(
        doctor["doctor_id"] == requested_doctor for doctor in _load_json("doctors.json")
    ):
        raise ValueError(f"Unknown doctor_id: {requested_doctor}")

    requested_date = _parse_date(date) if date is not None else None
    today = Date.today()
    availability = _load_json("availability.json")
    slots: list[dict[str, str]] = []
    for slot in availability:
        if slot["doctor_id"] != requested_doctor:
            continue
        slot_date = today + timedelta(days=slot["days_ahead"])
        if requested_date is not None and slot_date != requested_date:
            continue
        slot_result = {"date": slot_date.isoformat(), "time": slot["time"]}
        if (requested_doctor, slot_result["date"], slot_result["time"]) not in _booked_slots:
            slots.append(slot_result)
    return slots


def book_appointment(
    doctor_id: str,
    date: str,
    time: str,
    patient_name: str,
    *,
    approved: bool = False,
) -> dict[str, str]:
    """Book a mock slot only after an explicit human approval decision."""
    if not approved:
        raise PermissionError("Human approval is required before booking.")

    requested_doctor = _require_text(doctor_id, "doctor_id")
    requested_date = _parse_date(_require_text(date, "date"))
    requested_time = _require_text(time, "time")
    _require_text(patient_name, "patient_name")

    slot_date = requested_date.isoformat()
    if not any(
        slot["date"] == slot_date and slot["time"] == requested_time
        for slot in check_availability(requested_doctor, slot_date)
    ):
        raise ValueError("The requested appointment slot is not available.")

    _booked_slots.add((requested_doctor, slot_date, requested_time))
    return {"status": "booked", "confirmation_id": str(uuid4())}


TOOLS: dict[str, ToolDefinition] = {
    "symptom_to_specialty": {
        "name": "symptom_to_specialty",
        "description": (
            "Map symptom keywords to a likely specialty and urgency using local "
            "rules; this is not a diagnosis."
        ),
        "args": {"symptoms": "str"},
        "returns": "{specialty: str, urgency: routine|urgent|emergency, general_info: str}",
        "function": symptom_to_specialty,
    },
    "find_doctors": {
        "name": "find_doctors",
        "description": "List doctors for a medical specialty.",
        "args": {"specialty": "str"},
        "returns": "list[{doctor_id: str, name: str, specialty: str}]",
        "function": find_doctors,
    },
    "check_availability": {
        "name": "check_availability",
        "description": (
            "List open slots for a doctor on a date, or over the next three days "
            "when date is omitted."
        ),
        "args": {"doctor_id": "str", "date": "str (optional, YYYY-MM-DD)"},
        "returns": "list[{date: str, time: str}]",
        "function": check_availability,
    },
    "book_appointment": {
        "name": "book_appointment",
        "description": (
            "Book an available slot. Human approval is required before booking."
        ),
        "args": {
            "doctor_id": "str",
            "date": "str (YYYY-MM-DD)",
            "time": "str",
            "patient_name": "str",
        },
        "returns": "{status: str, confirmation_id: str}",
        "function": book_appointment,
    },
}
