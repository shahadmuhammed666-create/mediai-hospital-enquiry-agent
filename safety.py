from __future__ import annotations

from typing import Any


DISCLAIMER = (
    "This is general information, not a medical diagnosis. Please consult a "
    "qualified healthcare professional."
)
EMERGENCY_MESSAGE = (
    "Your symptoms may require emergency care. Call emergency services now or "
    "go to the nearest emergency department. Do not wait for an appointment."
)
TOOL_FAILURE_MESSAGE = (
    "I'm sorry, I couldn't complete that part of your enquiry. Please contact "
    "the hospital reception for help."
)


def request_human_approval(action_description: str) -> bool:
    """Ask a person to approve a sensitive action in the command line."""
    print(f"Approval required: {action_description}")
    return input("Approve this action? (y/n): ").strip().casefold() == "y"


def is_emergency(result: Any) -> bool:
    return isinstance(result, dict) and result.get("urgency") == "emergency"


def add_medical_disclaimer(answer: str) -> str:
    if DISCLAIMER.casefold() in answer.casefold():
        return answer
    return f"{answer.rstrip()} {DISCLAIMER}"
