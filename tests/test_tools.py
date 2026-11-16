from datetime import date, timedelta

import pytest

from tools import (
    TOOLS,
    book_appointment,
    check_availability,
    find_doctors,
    symptom_to_specialty,
)


def test_symptom_to_specialty_matches_rash_and_urgency() -> None:
    result = symptom_to_specialty("I have rashes all over my body")

    assert result["specialty"] == "Dermatology"
    assert result["urgency"] == "routine"
    assert "doctor" in result["general_info"]


def test_emergency_keyword_takes_precedence_over_routine_match() -> None:
    result = symptom_to_specialty("rash and trouble breathing")

    assert result["urgency"] == "emergency"
    assert result["specialty"] == "Emergency Medicine"


def test_symptom_to_specialty_uses_general_practice_for_unknown_symptoms() -> None:
    result = symptom_to_specialty("something unusual")

    assert result["specialty"] == "General Practice"
    assert result["urgency"] == "routine"


def test_find_doctors_matches_specialty_case_insensitively() -> None:
    doctors = find_doctors("dermatology")

    assert len(doctors) == 2
    assert {doctor["doctor_id"] for doctor in doctors} == {"DOC001", "DOC002"}


def test_check_availability_defaults_to_next_three_days() -> None:
    slots = check_availability("DOC001")
    dates = {date.fromisoformat(slot["date"]) for slot in slots}

    assert len(slots) == 2
    assert all(
        date.today() < slot_date <= date.today() + timedelta(days=3)
        for slot_date in dates
    )


def test_check_availability_can_filter_by_date() -> None:
    target_date = (date.today() + timedelta(days=1)).isoformat()

    slots = check_availability("DOC001", target_date)

    assert slots == [{"date": target_date, "time": "09:30"}]


def test_check_availability_rejects_unknown_doctor_and_invalid_date() -> None:
    with pytest.raises(ValueError, match="Unknown doctor_id"):
        check_availability("UNKNOWN")
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        check_availability("DOC001", "not-a-date")


def test_booking_requires_approval_and_removes_slot_from_availability() -> None:
    target_date = (date.today() + timedelta(days=1)).isoformat()

    with pytest.raises(PermissionError, match="Human approval"):
        book_appointment("DOC001", target_date, "09:30", "Alex")

    booking = book_appointment(
        "DOC001", target_date, "09:30", "Alex", approved=True
    )

    assert booking["status"] == "booked"
    assert booking["confirmation_id"]
    assert check_availability("DOC001", target_date) == []


def test_registry_exposes_four_described_tools() -> None:
    assert set(TOOLS) == {
        "symptom_to_specialty",
        "find_doctors",
        "check_availability",
        "book_appointment",
    }
    assert all(
        tool["name"] == name
        and tool["description"]
        and tool["args"]
        and tool["returns"]
        and callable(tool["function"])
        for name, tool in TOOLS.items()
    )
