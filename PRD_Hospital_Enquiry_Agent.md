# PRD: Hospital Patient Enquiry AI Agent

**Build tool:** Antigravity
**Language:** Python 3.10+
**Hard constraint:** NO LangChain, LlamaIndex, CrewAI, or any agent framework. The agent loop must be written by hand in plain Python.

---

## 1. Overview

Build an AI agent that answers online enquiries from hospital patients. The agent reads a patient's message, makes a plan, uses tools to look up real information (which specialty fits the symptoms, which doctors exist, who is available), and replies with a clear, safe answer.

**Example enquiry the agent must handle:**
> "I am having rashes all over my body. What does that indicate and which doctor should I consult? Also check the availability of doctors."

**Expected behaviour:** The agent plans, maps the symptom to a specialty (e.g. Dermatology), finds doctors in that specialty, checks their availability, and replies with general information, the recommended specialty, available slots, and a safety note. It does NOT diagnose.

## 2. Goals and Non-Goals

### Goals
- Answer symptom-related "which doctor?" questions.
- Check real doctor availability using tools.
- Offer to book an appointment (only after patient confirmation).
- Be safe: no diagnosis, escalate emergencies, log everything.
- Be fully testable with a fake LLM, then a real one.

### Non-Goals
- Giving medical diagnoses or prescribing medication.
- Payments, insurance, or medical records access.
- A web UI (a command-line interface is enough for v1).

## 3. Users

- **Patient / visitor** sending an enquiry through a chat or form (simulated via CLI in v1).
- **Hospital staff** who review logs and approve sensitive actions (simulated via a console prompt in v1).

## 4. Core Requirements

### R1. Agent Loop: Think → Act → Look
Write a plain Python `while` loop with this cycle on every step:

1. **Think:** send the conversation, plan, and memory to the LLM. The LLM returns a JSON decision.
2. **Act:** if the decision is a tool call, run that tool with the given arguments.
3. **Look:** read the tool result (or error), add it to the history as an observation, then go back to Think.
4. The loop ends when the LLM returns a `final_answer`, or when the step limit is hit.

**LLM output format (strict JSON):**
```json
{"thought": "short reasoning", "action": "tool_name", "args": {"key": "value"}}
```
or
```json
{"thought": "I have enough info", "final_answer": "Reply text to the patient"}
```

If the LLM returns invalid JSON, the agent re-prompts it once with the parse error (see R3).

> Note on "L1, L3": treated here as the loop levels from the course/brief: L1 = the basic think-act-look loop, L3 = the loop with planning, memory, and error recovery layered on. Antigravity should keep these layers cleanly separated in code.

### R2. Tools (4 tools, each clearly described)
Each tool is a plain Python function with a registry entry containing `name`, `description`, `args` (names and types), and `returns`. The tool descriptions are injected into the LLM prompt.

| # | Tool | Purpose | Args | Returns |
|---|------|---------|------|---------|
| 1 | `symptom_to_specialty` | Maps a symptom description to a likely medical specialty plus an urgency flag (`routine` / `urgent` / `emergency`). Uses a local rules/keyword file, NOT diagnosis. | `symptoms: str` | `{specialty, urgency, general_info}` |
| 2 | `find_doctors` | Lists doctors in a given specialty. | `specialty: str` | list of `{doctor_id, name, specialty}` |
| 3 | `check_availability` | Returns open slots for a doctor on a date (or the next 3 days if no date). | `doctor_id: str`, `date: str (YYYY-MM-DD, optional)` | list of `{date, time}` |
| 4 | `book_appointment` | Books a slot. **Sensitive action: requires human approval (see R6).** | `doctor_id: str`, `date: str`, `time: str`, `patient_name: str` | `{status, confirmation_id}` |

**Data for v1:** use mock JSON files (`data/doctors.json`, `data/availability.json`, `data/symptom_rules.json`). Keep tool code behind a clean interface so real hospital APIs can replace the mocks later.

### R3. Error Recovery (fixing errors when a tool fails)
- Wrap every tool call in try/except. Never let a tool crash the loop.
- On failure, return the error to the LLM as an observation: `{"error": "message", "tool": "name"}`.
- The LLM must then try to fix the problem: correct bad arguments, try an alternative tool, or retry.
- Retry limit: max **2 retries per tool call**. After that, the agent tells the patient it could not complete that part and suggests contacting the hospital reception, instead of guessing.
- Handle these cases explicitly: unknown tool name, missing/invalid args, tool exception, empty results (e.g. no doctors found), invalid LLM JSON.

### R4. Memory Between Tasks
- **Short-term:** conversation history inside one task (messages, observations).
- **Long-term:** a JSON file (`memory/memory.json`) that persists across runs/enquiries. Store per patient session ID: name (if given), past symptoms discussed, specialties recommended, appointments booked, and stated preferences (e.g. preferred time of day).
- At the start of each task, load relevant memory and include it in the prompt. At the end, save a summary.
- Do not store sensitive data beyond what is needed. Provide a `clear_memory(patient_id)` function.

### R5. Plan Before Acting
- Before any tool call, the agent makes one LLM call that produces a short numbered plan (3-6 steps), for example:
  1. Check if symptoms indicate an emergency
  2. Map symptoms to a specialty
  3. Find doctors in that specialty
  4. Check availability
  5. Reply with summary and offer booking
- The plan is logged and included in every later prompt.
- If an observation changes things (e.g. emergency detected, no doctors available), the agent may revise the plan once and logs the revision.

### R6. Safety
1. **Asks first:** `book_appointment` (and any future action that changes data) always pauses and asks the human for approval via a console `y/n` prompt, showing exactly what will be done. If denied, the agent tells the patient the booking was not made.
2. **Step limit:** a hard `MAX_STEPS` (default 8) per task. If reached, stop and return a safe fallback message plus a log entry.
3. **Logging:** every plan, thought, tool call, tool result, error, approval decision, and final answer is written to `logs/agent.log` with timestamp and task ID (JSON lines format).
4. **Medical safety rules (enforced in the system prompt AND in code):**
   - Never diagnose or name a condition as certain. Say "this can be associated with..." and recommend seeing a doctor.
   - Never recommend medication or doses.
   - If `urgency == emergency` (e.g. chest pain, trouble breathing, severe allergic reaction with swelling, heavy bleeding), skip the normal flow and reply immediately telling the patient to call emergency services / go to the nearest ER.
   - Every medical reply ends with a short disclaimer.
5. **Privacy:** don't log full personal identifiers beyond patient ID and first name.

### R7. Tests: Fake LLM First, Then a Real One
- Define an `LLMClient` interface with one method: `complete(messages) -> str`.
- **`FakeLLM`:** returns scripted responses in order, so tests are fast, free, and deterministic.
- **`RealLLM`:** a thin wrapper around OpenAI Chat Completions. Read the API key from `OPENAI_API_KEY` and the optional model name from `OPENAI_MODEL`; never hardcode credentials.
- The agent receives the LLM client through its constructor (dependency injection), so swapping Fake ↔ Real needs no agent code changes.

**Required tests with `FakeLLM` (pytest):**
1. Happy path: rash enquiry → plan → symptom tool → find doctors → availability → final answer mentioning Dermatology and slots.
2. Tool failure: `check_availability` raises an error → agent recovers (retries or reports gracefully).
3. Invalid LLM JSON → agent re-prompts and continues.
4. Step limit: scripted LLM loops forever → agent stops at `MAX_STEPS`.
5. Approval denied for `book_appointment` → no booking happens.
6. Emergency symptom → immediate emergency reply, no booking offered.
7. Memory: second task for the same patient uses stored info from the first.
8. Unit tests for each tool against the mock data.

**Then real LLM:** run the same example enquiries manually with `python main.py --real "..."` plus a small set of smoke tests marked `@pytest.mark.real_llm` that are skipped unless `OPENAI_API_KEY` is present. Review the logs and compare the behaviour to the fake-LLM runs.

## 5. Suggested Project Structure

```
hospital_agent/
├── main.py                 # CLI entry point (--real flag)
├── agent.py                # Agent class: plan + think/act/look loop
├── llm.py                  # LLMClient interface, FakeLLM, RealLLM
├── tools.py                # 4 tools + registry with descriptions
├── memory.py               # load/save/clear long-term memory
├── safety.py               # approval prompt, step limit, emergency check
├── logger.py               # JSON-lines logging
├── prompts.py              # system prompt, planning prompt
├── data/
│   ├── doctors.json
│   ├── availability.json
│   └── symptom_rules.json
├── memory/memory.json
├── logs/agent.log
└── tests/
    ├── test_agent_fake_llm.py
    ├── test_tools.py
    └── test_real_llm.py
```

## 6. Example End-to-End Flow

**Patient:** "I am having rashes all over my body, what does that indicate and which doctor should I consult? Also check doctor availability."

1. **Plan:** (1) check urgency, (2) map symptom to specialty, (3) find doctors, (4) check availability, (5) reply and offer booking.
2. **Act:** `symptom_to_specialty("rashes all over body")` → `{specialty: "Dermatology", urgency: "routine"}`
3. **Act:** `find_doctors("Dermatology")` → 2 doctors.
4. **Act:** `check_availability(doctor_id)` for each → slots.
5. **Final answer:** general info (rashes can have many causes such as allergies or skin conditions, and only a doctor can say which), recommended specialty (Dermatology), doctor names with available slots, advice to seek urgent care if there is swelling of face/lips, trouble breathing, or fever, an offer to book, and a disclaimer.

## 7. Success Criteria

- All 8 fake-LLM tests pass.
- Agent never exceeds `MAX_STEPS`.
- No booking ever happens without explicit approval.
- Every run produces a complete, readable log.
- No agent framework is imported anywhere (verify with a grep for `langchain`).
- The same agent code runs with both `FakeLLM` and `RealLLM`.

## 8. Out of Scope for v1 (Future Ideas)

- Web or WhatsApp chat front end
- Integration with the real hospital scheduling system
- Multi-language support
- Voice enquiries

## 9. Instructions for Antigravity

Build the project exactly as specified above. Start with `llm.py` (interface + FakeLLM), then `tools.py` with mock data, then `agent.py`, then `safety.py`/`memory.py`/`logger.py`, then the tests. Run the fake-LLM tests until they all pass before wiring the real LLM. Keep the code simple, commented, and beginner-readable.
