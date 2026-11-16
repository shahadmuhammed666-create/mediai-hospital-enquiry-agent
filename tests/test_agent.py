import json
from datetime import date, timedelta

import pytest

from agent import Agent
from llm import FakeLLM
from logger import JsonLinesLogger
from memory import MemoryStore
from safety import DISCLAIMER
from tools import TOOLS, check_availability


PLAN = json.dumps(
    {
        "plan": [
            "Check symptom urgency",
            "Find a suitable specialty",
            "Check doctors and their availability",
        ]
    }
)


class RecordingFakeLLM(FakeLLM):
    def __init__(self, scripted_responses: list[str]) -> None:
        super().__init__(scripted_responses)
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(list(messages))
        return super().complete(messages)


def test_agent_plans_then_runs_think_act_look_tool_flow(tmp_path) -> None:
    llm = RecordingFakeLLM(
        [
            PLAN,
            json.dumps(
                {
                    "thought": "Identify the relevant specialty.",
                    "action": "symptom_to_specialty",
                    "args": {"symptoms": "rash"},
                }
            ),
            json.dumps(
                {
                    "thought": "Find dermatologists.",
                    "action": "find_doctors",
                    "args": {"specialty": "Dermatology"},
                }
            ),
            json.dumps(
                {
                    "thought": "Check the first doctor's slots.",
                    "action": "check_availability",
                    "args": {"doctor_id": "DOC001"},
                }
            ),
            json.dumps(
                {
                    "thought": "I have enough information.",
                    "final_answer": (
                        "Dermatology may be appropriate. Dr. Maya Patel has a "
                        "09:30 slot. A doctor can assess your symptoms."
                    ),
                }
            ),
        ]
    )

    answer = make_agent(llm, tmp_path).run(
        "I have a rash; which doctor should I see?"
    )

    assert "Dermatology" in answer
    assert "09:30" in answer
    assert answer.endswith(DISCLAIMER)
    assert "Execution plan:" in llm.calls[1][-1]["content"]
    assert any(
        "Tool observation:" in message["content"]
        for message in llm.calls[-1]
        if message["role"] == "user"
    )
    log_entries = [
        json.loads(line)
        for line in (tmp_path / "logs" / "agent.log")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {
        "plan",
        "thought",
        "tool_call",
        "tool_result",
        "final_answer",
    }.issubset({entry["event"] for entry in log_entries})
    assert all(entry["task_id"] for entry in log_entries)


def test_agent_reprompts_once_after_invalid_json(tmp_path) -> None:
    llm = RecordingFakeLLM(
        [
            PLAN,
            "{invalid",
            json.dumps(
                {
                    "thought": "Answer directly.",
                    "final_answer": "Please contact the hospital for an assessment.",
                }
            ),
        ]
    )

    answer = make_agent(llm, tmp_path).run("Who can help with my symptoms?")

    assert answer.startswith("Please contact the hospital for an assessment.")
    assert answer.endswith(DISCLAIMER)
    assert len(llm.calls) == 3
    assert "Parse error:" in llm.calls[2][-1]["content"]


def test_agent_cannot_exceed_hard_step_limit() -> None:
    llm = RecordingFakeLLM([])

    with pytest.raises(ValueError, match="hard limit"):
        Agent(llm, max_steps=9)


def test_agent_returns_tool_errors_as_observations(tmp_path) -> None:
    llm = RecordingFakeLLM(
        [
            PLAN,
            json.dumps(
                {
                    "thought": "Try an unknown tool.",
                    "action": "missing_tool",
                    "args": {},
                }
            ),
            json.dumps(
                {
                    "thought": "Respond with a safe fallback.",
                    "final_answer": "Please contact hospital reception for help.",
                }
            ),
        ]
    )

    answer = make_agent(llm, tmp_path).run("Can I see a doctor?")

    assert "reception" in answer
    assert answer.endswith(DISCLAIMER)
    assert any(
        "Unknown tool name: missing_tool" in message["content"]
        for message in llm.calls[-1]
    )


def test_agent_reports_availability_tool_failure_without_guessing(tmp_path) -> None:
    llm = RecordingFakeLLM(
        [
            PLAN,
            json.dumps(
                {
                    "thought": "Check the selected doctor's availability.",
                    "action": "check_availability",
                    "args": {"doctor_id": "DOC001"},
                }
            ),
            json.dumps(
                {
                    "thought": "Availability lookup failed, so do not guess.",
                    "final_answer": (
                        "I couldn't check availability. Please contact hospital "
                        "reception for current slots."
                    ),
                }
            ),
        ]
    )

    def fail_availability(**arguments):
        raise RuntimeError("Availability service is unavailable.")

    failing_tool = {**TOOLS["check_availability"], "function": fail_availability}
    tools = {**TOOLS, "check_availability": failing_tool}

    answer = make_agent(llm, tmp_path, tools=tools).run(
        "Check when Dr. Maya Patel is available."
    )

    assert "couldn't check availability" in answer
    assert "contact hospital reception" in answer
    assert answer.endswith(DISCLAIMER)
    assert any(
        "Availability service is unavailable." in message["content"]
        for message in llm.calls[-1]
    )


def test_agent_marks_empty_tool_results_as_observations(tmp_path) -> None:
    llm = RecordingFakeLLM(
        [
            PLAN,
            json.dumps(
                {
                    "thought": "Search the specialty.",
                    "action": "find_doctors",
                    "args": {"specialty": "Neurosurgery"},
                }
            ),
            json.dumps(
                {
                    "thought": "No doctors were found.",
                    "final_answer": "No matching doctors were found; contact reception.",
                }
            ),
        ]
    )

    answer = make_agent(llm, tmp_path).run("Find a neurosurgeon.")

    assert "No matching doctors" in answer
    assert any('"empty": true' in message["content"] for message in llm.calls[-1])


def test_agent_stops_at_max_steps(tmp_path) -> None:
    llm = RecordingFakeLLM(
        [
            PLAN,
            json.dumps(
                {
                    "thought": "Try an unknown tool.",
                    "action": "missing_tool",
                    "args": {},
                }
            ),
            json.dumps(
                {
                    "thought": "Try again.",
                    "action": "missing_tool",
                    "args": {},
                }
            ),
        ]
    )

    answer = make_agent(llm, tmp_path, max_steps=2).run("Can I see a doctor?")

    assert "couldn't complete" in answer
    assert answer.endswith(DISCLAIMER)
    assert len(llm.calls) == 3


def test_agent_stops_after_two_retries_for_a_failing_tool(tmp_path) -> None:
    failed_tool_decision = json.dumps(
        {"thought": "Try the unavailable tool.", "action": "missing_tool", "args": {}}
    )
    llm = RecordingFakeLLM(
        [PLAN, failed_tool_decision, failed_tool_decision, failed_tool_decision]
    )

    answer = make_agent(llm, tmp_path).run("Can I see a doctor?")

    assert "contact the hospital reception" in answer
    assert answer.endswith(DISCLAIMER)
    assert len(llm.calls) == 4


def test_emergency_result_skips_normal_flow(tmp_path) -> None:
    llm = RecordingFakeLLM(
        [
            PLAN,
            json.dumps(
                {
                    "thought": "Check the symptom urgency.",
                    "action": "symptom_to_specialty",
                    "args": {"symptoms": "trouble breathing"},
                }
            ),
        ]
    )

    answer = make_agent(llm, tmp_path).run("I am having trouble breathing.")

    assert "Call emergency services now" in answer
    assert answer.endswith(DISCLAIMER)
    assert len(llm.calls) == 2


def test_booking_is_not_made_without_approval(tmp_path) -> None:
    target_date = (date.today() + timedelta(days=1)).isoformat()
    llm = RecordingFakeLLM(
        [
            PLAN,
            json.dumps(
                {
                    "thought": "Book the requested appointment.",
                    "action": "book_appointment",
                    "args": {
                        "doctor_id": "DOC001",
                        "date": target_date,
                        "time": "09:30",
                        "patient_name": "Alex Taylor",
                    },
                }
            ),
            json.dumps(
                {
                    "thought": "The human declined the booking.",
                    "final_answer": "The appointment was not booked.",
                }
            ),
        ]
    )
    approval_requests: list[str] = []

    answer = make_agent(
        llm,
        tmp_path,
        approval_handler=lambda description: approval_requests.append(description)
        or False,
    ).run("Please book a visit.", patient_id="patient-123")

    assert "not booked" in answer
    assert "Alex Taylor" in approval_requests[0]
    assert check_availability("DOC001", target_date) == [
        {"date": target_date, "time": "09:30"}
    ]


def test_booking_proceeds_only_after_approval(tmp_path) -> None:
    target_date = (date.today() + timedelta(days=1)).isoformat()
    llm = RecordingFakeLLM(
        [
            PLAN,
            json.dumps(
                {
                    "thought": "Book the approved slot.",
                    "action": "book_appointment",
                    "args": {
                        "doctor_id": "DOC002",
                        "date": target_date,
                        "time": "11:00",
                        "patient_name": "Jamie Park",
                    },
                }
            ),
            json.dumps(
                {
                    "thought": "The tool confirmed the appointment.",
                    "final_answer": "Your appointment has been booked.",
                }
            ),
        ]
    )
    approval_requests: list[str] = []

    answer = make_agent(
        llm,
        tmp_path,
        approval_handler=lambda description: approval_requests.append(description)
        or True,
    ).run("Please book a visit.")

    assert "booked" in answer
    assert approval_requests
    assert check_availability("DOC002", target_date) == []


def test_agent_persists_and_reuses_patient_memory(tmp_path) -> None:
    llm = RecordingFakeLLM(
        [
            PLAN,
            json.dumps(
                {
                    "thought": "Map the symptom.",
                    "action": "symptom_to_specialty",
                    "args": {"symptoms": "Morgan Reed rash"},
                }
            ),
            json.dumps(
                {
                    "thought": "Respond with the recommendation.",
                    "final_answer": "Dermatology may be appropriate.",
                }
            ),
            PLAN,
            json.dumps(
                {
                    "thought": "Respond using prior context.",
                    "final_answer": "I can use your previous enquiry as context.",
                }
            ),
        ]
    )
    agent = make_agent(llm, tmp_path)

    agent.run(
        "I have a rash and prefer morning appointments.",
        patient_id="patient-456",
        patient_name="Morgan Reed",
    )
    agent.run("I have another question.", patient_id="patient-456")

    second_plan_prompt = llm.calls[3][-1]["content"]
    assert "Morgan" in second_plan_prompt
    assert "rash" in second_plan_prompt
    assert "Dermatology" in second_plan_prompt
    saved_memory = (tmp_path / "memory" / "memory.json").read_text(encoding="utf-8")
    assert "Morgan Reed" not in saved_memory
    assert "Morgan rash" in saved_memory
    assert '"time_of_day": "morning"' in saved_memory


def make_agent(llm, tmp_path, **kwargs) -> Agent:
    return Agent(
        llm,
        logger=JsonLinesLogger(tmp_path / "logs" / "agent.log"),
        memory_store=MemoryStore(tmp_path / "memory" / "memory.json"),
        **kwargs,
    )
