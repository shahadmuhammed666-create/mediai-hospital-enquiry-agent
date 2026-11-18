from __future__ import annotations

import argparse
import sys

from agent import Agent
from llm import GeminiLLM


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hospital patient enquiry agent.")
    parser.add_argument(
        "--real",
        action="store_true",
        help="use a real model API",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="model API read timeout in seconds (default: 60)",
    )
    parser.add_argument("--patient-id", help="optional patient session ID for memory")
    parser.add_argument("--patient-name", help="optional patient name")
    parser.add_argument(
        "message",
        nargs="?",
        help="patient enquiry; prompts for input when omitted",
    )
    args = parser.parse_args(argv)
    if not args.real:
        parser.error("Pass --real to use an API provider; FakeLLM is used by the test suite.")

    message = args.message if args.message is not None else input("Patient enquiry: ")
    if not message.strip():
        parser.error("The patient enquiry must not be empty.")

    try:
        llm = GeminiLLM(timeout=args.timeout)
        answer = Agent(llm).run(
            message,
            patient_id=args.patient_id,
            patient_name=args.patient_name,
        )
    except (RuntimeError, ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
