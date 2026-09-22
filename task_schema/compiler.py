"""Length-framed task prompt compiler with unambiguous trust boundaries."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping

from .schema import TaskContractError, validate_task_record


HEADER_PATTERN = re.compile(
    rb"<\|vf:([a-z][a-z0-9-]*):v1:([0-9]+):([a-f0-9]{64})\|>\n"
)


def _frame(section_type: str, payload: Mapping[str, Any]) -> bytes:
    body = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    digest = hashlib.sha256(body).hexdigest()
    header = f"<|vf:{section_type}:v1:{len(body)}:{digest}|>\n".encode("ascii")
    return header + body + b"\n"


def compile_task_record(value: Mapping[str, Any]) -> str:
    """Compile one validated record; body markers cannot escape length framing."""

    record = validate_task_record(value)
    sections: list[tuple[str, Mapping[str, Any]]] = [
        (
            "record-header",
            {
                "recordId": record["recordId"],
                "recordKind": record["recordKind"],
                "task": record["task"],
                "contractVersion": record["contractVersion"],
                "metadata": record["metadata"],
            },
        ),
        ("system", record["input"]["system"]),
        ("user", record["input"]["user"]),
        ("project-context", record["input"]["projectContext"]),
    ]
    sections.extend(("tool-evidence", item) for item in record["input"]["toolEvidence"])
    output = record.get("output")
    if output is None:
        sections.append(
            (
                "assistant-request",
                {
                    "allowedOutputTypes": [
                        "assistant-text",
                        "structured-action",
                        "citation",
                        "refusal",
                        "uncertainty",
                    ]
                },
            )
        )
    else:
        if output.get("assistantText") is not None:
            sections.append(("assistant-text", output["assistantText"]))
        if output.get("refusal") is not None:
            sections.append(("refusal", output["refusal"]))
        if output.get("uncertainty") is not None:
            sections.append(("uncertainty", output["uncertainty"]))
        sections.extend(("citation", item) for item in output.get("citations", []))
        sections.extend(
            ("structured-action", item) for item in output.get("structuredActions", [])
        )
    return b"".join(_frame(section, payload) for section, payload in sections).decode("utf-8")


def parse_compiled_sections(compiled: str) -> list[dict[str, Any]]:
    """Parse and checksum-verify compiled sections for tests and tooling."""

    raw = compiled.encode("utf-8")
    cursor = 0
    sections: list[dict[str, Any]] = []
    while cursor < len(raw):
        match = HEADER_PATTERN.match(raw, cursor)
        if match is None:
            raise TaskContractError(
                "TASK_PROMPT_FRAME_INVALID", f"Invalid task prompt frame at byte {cursor}."
            )
        section_type = match.group(1).decode("ascii")
        size = int(match.group(2))
        declared_hash = match.group(3).decode("ascii")
        body_start = match.end()
        body_end = body_start + size
        body = raw[body_start:body_end]
        if len(body) != size or raw[body_end:body_end + 1] != b"\n":
            raise TaskContractError("TASK_PROMPT_FRAME_INVALID", "Task prompt section length is invalid.")
        if hashlib.sha256(body).hexdigest() != declared_hash:
            raise TaskContractError(
                "TASK_PROMPT_CHECKSUM_MISMATCH", "Task prompt section checksum is invalid."
            )
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise TaskContractError("TASK_PROMPT_PAYLOAD_INVALID", "Task prompt payload is invalid JSON.") from error
        if not isinstance(payload, dict):
            raise TaskContractError("TASK_PROMPT_PAYLOAD_INVALID", "Task prompt payload must be an object.")
        sections.append({"sectionType": section_type, "payload": payload})
        cursor = body_end + 1
    return sections
