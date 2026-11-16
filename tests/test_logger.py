import json

from logger import JsonLinesLogger


def test_logger_writes_json_lines_and_redacts_full_patient_name(tmp_path) -> None:
    path = tmp_path / "logs" / "agent.log"
    logger = JsonLinesLogger(path)

    logger.log(
        "task-1",
        "tool_call",
        sensitive_values=("Alex Taylor",),
        arguments={"patient_name": "Alex Taylor", "symptoms": "rash"},
        thought="Alex Taylor asked about a rash.",
    )

    entry = json.loads(path.read_text(encoding="utf-8").strip())
    assert entry["task_id"] == "task-1"
    assert entry["event"] == "tool_call"
    assert entry["arguments"]["patient_name"] == "Alex"
    assert entry["thought"] == "Alex asked about a rash."
    assert "Alex Taylor" not in path.read_text(encoding="utf-8")
