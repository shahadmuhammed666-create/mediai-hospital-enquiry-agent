from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from typing import Any
from uuid import uuid4

from llm import LLMClient
from logger import JsonLinesLogger
from memory import MemoryStore
from safety import (
    DISCLAIMER,
    EMERGENCY_MESSAGE,
    TOOL_FAILURE_MESSAGE,
    add_medical_disclaimer,
    is_emergency,
    request_human_approval,
)
from tools import TOOLS, ToolDefinition


MAX_STEPS = 8
_MAX_TOOL_FAILURES = 3

_SYSTEM_PROMPT = """\
You are a hospital enquiry assistant. Follow the patient's request using only
the available tools and their observations. Never diagnose, state a condition
as certain, or recommend medication or doses. Explain that symptoms can have
many causes and recommend assessment by a healthcare professional. If a tool
observation reports emergency urgency, stop the normal flow and direct the
patient to emergency services or the nearest emergency department. Do not claim
an appointment was booked unless a tool confirms it. Never guess when a tool
returns no results; explain this and suggest contacting hospital reception.
Every medical response must end with the supplied medical disclaimer.

For each decision, return exactly one JSON object. Use either:
{"thought":"short reasoning","action":"tool_name","args":{"key":"value"}}
or:
{"thought":"I have enough information","final_answer":"Reply to the patient"}
"""

_STEP_LIMIT_MESSAGE = (
    "I'm sorry, I couldn't complete your enquiry. Please contact the hospital "
    "reception for help."
)
_INVALID_RESPONSE_MESSAGE = (
    "I'm sorry, I couldn't process the assistant response. Please contact the "
    "hospital reception for help."
)


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        tools: Mapping[str, ToolDefinition] = TOOLS,
        max_steps: int = MAX_STEPS,
        approval_handler: Callable[[str], bool] = request_human_approval,
        logger: JsonLinesLogger | None = None,
        memory_store: MemoryStore | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1.")
        if max_steps > MAX_STEPS:
            raise ValueError(f"max_steps cannot exceed the hard limit of {MAX_STEPS}.")
        self._llm = llm
        self._tools = tools
        self._max_steps = max_steps
        self._approval_handler = approval_handler
        self._logger = logger or JsonLinesLogger()
        self._memory_store = memory_store or MemoryStore()

    def run(
        self,
        user_message: str,
        memory: Mapping[str, Any] | None = None,
        *,
        patient_id: str | None = None,
        patient_name: str | None = None,
    ) -> str:
        if not isinstance(user_message, str) or not user_message.strip():
            raise ValueError("user_message must be a non-empty string.")
        if patient_id is not None and (
            not isinstance(patient_id, str) or not patient_id.strip()
        ):
            raise ValueError("patient_id must be a non-empty string when provided.")
        if patient_id is not None:
            patient_id = patient_id.strip()
        if patient_name is not None and (
            not isinstance(patient_name, str) or not patient_name.strip()
        ):
            raise ValueError("patient_name must be a non-empty string when provided.")

        task_id = str(uuid4())
        sensitive_values = [patient_name] if patient_name else []
        state: dict[str, Any] = {
            "past_symptoms": [],
            "recommended_specialties": [],
            "appointments_booked": [],
            "preferences": self._extract_preferences(user_message),
            "first_name": patient_name,
        }
        try:
            stored_memory = (
                self._memory_store.load_memory(patient_id) if patient_id else {}
            )
            if not state["first_name"]:
                state["first_name"] = stored_memory.get("first_name")
            task_memory = {**stored_memory, **(memory or {})}
            self._log(
                task_id,
                "task_started",
                sensitive_values,
                patient_id=patient_id,
                first_name=self._first_name(state["first_name"]),
            )

            answer = self._execute(
                user_message.strip(),
                task_memory,
                task_id,
                state,
                sensitive_values,
            )
            answer = add_medical_disclaimer(answer)
            if patient_id:
                summary: dict[str, Any] = {
                    "past_symptoms": state["past_symptoms"],
                    "recommended_specialties": state["recommended_specialties"],
                    "appointments_booked": state["appointments_booked"],
                    "preferences": state["preferences"],
                }
                if state["first_name"]:
                    summary["first_name"] = self._first_name(state["first_name"])
                self._memory_store.save_summary(patient_id, summary)
            self._log(
                task_id,
                "final_answer",
                sensitive_values,
                answer=answer,
            )
            return answer
        except Exception as error:
            self._log(
                task_id,
                "error",
                sensitive_values,
                error=str(error),
                error_type=type(error).__name__,
            )
            raise

    def _execute(
        self,
        user_message: str,
        memory: Mapping[str, Any],
        task_id: str,
        state: dict[str, Any],
        sensitive_values: list[str],
    ) -> str:
        memory_text = json.dumps(memory, ensure_ascii=True, default=str)
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    _SYSTEM_PROMPT
                    + f"\nRequired disclaimer: {DISCLAIMER}\n"
                    + self._format_tools()
                ),
            },
            {
                "role": "user",
                "content": (
                    "Before taking any action, make a plan of 3-6 short steps. "
                    'Return JSON in the form {"plan":["step 1","step 2",...]} .'
                    f"\nPatient enquiry: {user_message}"
                    f"\nRelevant memory: {memory_text}"
                ),
            },
        ]

        plan_response = self._request_json(messages, task_id, "plan", sensitive_values)
        plan = self._read_plan(plan_response)
        if plan is None:
            self._log(
                task_id,
                "error",
                sensitive_values,
                phase="plan",
                error="The LLM did not return a valid plan of 3-6 steps.",
            )
            return _INVALID_RESPONSE_MESSAGE

        self._log(task_id, "plan", sensitive_values, plan=plan)
        plan_text = "\n".join(
            f"{step_number}. {step}" for step_number, step in enumerate(plan, 1)
        )
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Execution plan:\n{plan_text}\n"
                    "Follow this plan. You may adjust your actions based on tool "
                    "observations, but do not exceed the task step limit."
                ),
            }
        )

        failure_counts: dict[tuple[str, str], int] = {}
        step_number = 0
        while step_number < self._max_steps:
            step_number += 1
            decision = self._request_json(
                messages, task_id, "decision", sensitive_values
            )
            if not isinstance(decision, dict):
                self._log(
                    task_id,
                    "error",
                    sensitive_values,
                    phase="decision",
                    error="The LLM response was not a JSON object.",
                )
                return _INVALID_RESPONSE_MESSAGE

            thought = decision.get("thought")
            if isinstance(thought, str):
                self._log(task_id, "thought", sensitive_values, thought=thought)

            final_answer = decision.get("final_answer")
            action = decision.get("action")
            if isinstance(final_answer, str) and final_answer.strip() and action is None:
                return final_answer.strip()

            if not isinstance(action, str):
                observation = {
                    "error": (
                        "Decision must contain either final_answer or a string action."
                    )
                }
                self._log(task_id, "error", sensitive_values, **observation)
                self._add_observation(messages, observation)
                continue

            if "args" not in decision:
                exhausted = self._record_tool_failure(
                    task_id,
                    messages,
                    failure_counts,
                    action,
                    None,
                    "Tool arguments are missing.",
                    sensitive_values,
                )
                if exhausted:
                    return TOOL_FAILURE_MESSAGE
                continue

            arguments = decision["args"]
            if not isinstance(arguments, dict):
                error = "Tool arguments must be a JSON object."
                exhausted = self._record_tool_failure(
                    task_id,
                    messages,
                    failure_counts,
                    action,
                    arguments,
                    error,
                    sensitive_values,
                )
                if exhausted:
                    return TOOL_FAILURE_MESSAGE
                continue

            patient_name_argument = arguments.get("patient_name")
            if isinstance(patient_name_argument, str) and patient_name_argument:
                sensitive_values.append(patient_name_argument)
                if not state["first_name"]:
                    state["first_name"] = self._first_name(patient_name_argument)

            self._log(
                task_id,
                "tool_call",
                sensitive_values,
                tool=action,
                arguments=arguments,
                step=step_number,
            )
            tool = self._tools.get(action)
            if tool is None:
                exhausted = self._record_tool_failure(
                    task_id,
                    messages,
                    failure_counts,
                    action,
                    arguments,
                    f"Unknown tool name: {action}",
                    sensitive_values,
                )
                if exhausted:
                    return TOOL_FAILURE_MESSAGE
                continue

            if action == "book_appointment":
                if "approved" in arguments:
                    exhausted = self._record_tool_failure(
                        task_id,
                        messages,
                        failure_counts,
                        action,
                        arguments,
                        "Approval is handled by the agent, not by tool arguments.",
                        sensitive_values,
                    )
                    if exhausted:
                        return TOOL_FAILURE_MESSAGE
                    continue
                description = (
                    "Book appointment using these details: "
                    + json.dumps(arguments, ensure_ascii=True)
                )
                self._log(
                    task_id,
                    "approval_requested",
                    sensitive_values,
                    action=description,
                )
                approved = self._approval_handler(description)
                self._log(
                    task_id,
                    "approval_decision",
                    sensitive_values,
                    approved=approved,
                    action=description,
                )
                if not approved:
                    self._add_observation(
                        messages,
                        {
                            "tool": action,
                            "result": {
                                "status": "not_booked",
                                "reason": "Human approval was denied.",
                            },
                        },
                    )
                    return "The appointment was not booked because approval was denied."
                arguments = {**arguments, "approved": True}

            try:
                result = tool["function"](**arguments)
            except Exception as error:
                exhausted = self._record_tool_failure(
                    task_id,
                    messages,
                    failure_counts,
                    action,
                    decision["args"],
                    str(error),
                    sensitive_values,
                )
                if exhausted:
                    return TOOL_FAILURE_MESSAGE
            else:
                empty_result = result is None or result == [] or result == {}
                self._log(
                    task_id,
                    "tool_result",
                    sensitive_values,
                    tool=action,
                    result=result,
                    empty=empty_result,
                )
                self._add_observation(
                    messages,
                    {"tool": action, "result": result, "empty": empty_result},
                )
                self._record_memory_result(
                    state, action, decision["args"], result, sensitive_values
                )
                if is_emergency(result):
                    return EMERGENCY_MESSAGE

        self._log(
            task_id,
            "step_limit",
            sensitive_values,
            max_steps=self._max_steps,
        )
        return _STEP_LIMIT_MESSAGE

    def _format_tools(self) -> str:
        descriptions = []
        for name, tool in self._tools.items():
            descriptions.append(
                f"- {name}: {tool['description']} "
                f"Arguments: {json.dumps(tool['args'], ensure_ascii=True)}. "
                f"Returns: {tool['returns']}."
            )
        return "\n\nAvailable tools:\n" + "\n".join(descriptions)

    def _request_json(
        self,
        messages: list[dict[str, str]],
        task_id: str,
        phase: str,
        sensitive_values: list[str],
    ) -> object | None:
        response = self._llm.complete(messages)
        messages.append({"role": "assistant", "content": response})
        try:
            return json.loads(response)
        except json.JSONDecodeError as error:
            self._log(
                task_id,
                "error",
                sensitive_values,
                phase=phase,
                error=f"Invalid LLM JSON: {error}",
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous response was not valid JSON. "
                        f"Parse error: {error}. Return one valid JSON object only."
                    ),
                }
            )
            retry_response = self._llm.complete(messages)
            messages.append({"role": "assistant", "content": retry_response})
            try:
                return json.loads(retry_response)
            except json.JSONDecodeError as retry_error:
                self._log(
                    task_id,
                    "error",
                    sensitive_values,
                    phase=phase,
                    error=f"Invalid LLM JSON after retry: {retry_error}",
                )
                return None

    def _record_tool_failure(
        self,
        task_id: str,
        messages: list[dict[str, str]],
        failure_counts: dict[tuple[str, str], int],
        tool_name: str,
        arguments: object,
        error: str,
        sensitive_values: list[str],
    ) -> bool:
        argument_key = json.dumps(arguments, sort_keys=True, ensure_ascii=True, default=str)
        key = (tool_name, argument_key)
        failure_counts[key] = failure_counts.get(key, 0) + 1
        failures = failure_counts[key]
        self._log(
            task_id,
            "error",
            sensitive_values,
            tool=tool_name,
            arguments=arguments,
            error=error,
            attempts=failures,
        )
        exhausted = failures >= _MAX_TOOL_FAILURES
        observation: dict[str, object] = {
            "error": error,
            "tool": tool_name,
            "attempt": failures,
        }
        if exhausted:
            observation["message"] = (
                "The tool failed after two retries. Do not guess; tell the patient "
                "to contact hospital reception."
            )
            self._log(
                task_id,
                "error",
                sensitive_values,
                tool=tool_name,
                error="Tool retry limit reached.",
            )
        self._add_observation(messages, observation)
        return exhausted

    def _log(
        self,
        task_id: str,
        event: str,
        sensitive_values: list[str],
        **details: Any,
    ) -> None:
        self._logger.log(
            task_id,
            event,
            sensitive_values=tuple(sensitive_values),
            **details,
        )

    @staticmethod
    def _record_memory_result(
        state: dict[str, Any],
        action: str,
        arguments: dict[str, Any],
        result: object,
        sensitive_values: list[str],
    ) -> None:
        if action == "symptom_to_specialty":
            symptoms = arguments.get("symptoms")
            if isinstance(symptoms, str) and symptoms.strip():
                state["past_symptoms"].append(
                    Agent._redact_sensitive_text(symptoms.strip(), sensitive_values)
                )
            if isinstance(result, dict):
                specialty = result.get("specialty")
                if isinstance(specialty, str) and specialty not in state["recommended_specialties"]:
                    state["recommended_specialties"].append(specialty)
        if (
            action == "book_appointment"
            and isinstance(result, dict)
            and result.get("status") == "booked"
        ):
            state["appointments_booked"].append(
                {
                    key: value
                    for key, value in {
                        "doctor_id": arguments.get("doctor_id"),
                        "date": arguments.get("date"),
                        "time": arguments.get("time"),
                        "confirmation_id": result.get("confirmation_id"),
                    }.items()
                    if value is not None
                }
            )

    @staticmethod
    def _redact_sensitive_text(text: str, sensitive_values: list[str]) -> str:
        for sensitive_value in sensitive_values:
            if sensitive_value.strip():
                first_name = Agent._first_name(sensitive_value)
                if first_name:
                    text = re.sub(
                        re.escape(sensitive_value.strip()),
                        first_name,
                        text,
                        flags=re.IGNORECASE,
                    )
        return text

    @staticmethod
    def _extract_preferences(user_message: str) -> dict[str, str]:
        preference = re.search(
            r"\b(?:prefer|preferred|would like|only available)\b"
            r"[^.!?]{0,50}\b(morning|afternoon|evening)s?\b",
            user_message,
            flags=re.IGNORECASE,
        )
        if preference is None:
            return {}
        return {"time_of_day": preference.group(1).casefold()}

    @staticmethod
    def _read_plan(response: object | None) -> list[str] | None:
        if not isinstance(response, dict):
            return None
        plan = response.get("plan")
        if (
            not isinstance(plan, list)
            or not 3 <= len(plan) <= 6
            or any(not isinstance(step, str) or not step.strip() for step in plan)
        ):
            return None
        return [step.strip() for step in plan]

    @staticmethod
    def _add_observation(
        messages: list[dict[str, str]], observation: dict[str, object]
    ) -> None:
        messages.append(
            {
                "role": "user",
                "content": "Tool observation: "
                + json.dumps(observation, ensure_ascii=True, default=str),
            }
        )

    @staticmethod
    def _first_name(name: object) -> str | None:
        if isinstance(name, str) and name.strip():
            return name.strip().split()[0]
        return None
