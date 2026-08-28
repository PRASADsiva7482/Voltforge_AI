"""Generate native v1 task records only after deterministic label verification."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Iterable, Mapping

from electronics_corpus import get_electronics_corpus
from task_schema.schema import validate_task_record

from synthetic_data.grammar import (
    BOARD_PROMPTS,
    COMPONENT_ALIAS,
    DETERMINISTIC_SEED,
    FIRMWARE_TARGETS,
    GENERATOR_ID,
    GENERATOR_VERSION,
    GRAMMAR_VERSION,
    NON_AVR_FIRMWARE_BOARDS,
    PIN_PROMPTS,
    SUPPORTED_BOARDS,
    UNSAFE_PROMPTS,
    UNSUPPORTED_QUERIES,
    WIRING_PROMPTS,
    render_firmware_sources,
)
from synthetic_data.verifiers import (
    content_sha256,
    expected_connections,
    lookup_unknown_board,
    safety_receipt,
    validate_compiler_receipt,
    verify_board,
    verify_netlist,
    verify_pin_route,
)


SOURCE_IDS = (
    "vf-src-curated-electronics-corpus-v1",
    "vf-src-verified-synthetic-pipeline-v1",
)
SYSTEM_TEXT = (
    "Answer only from VoltForge deterministic evidence. Preserve exact hardware variants, "
    "state uncertainty when evidence is incomplete, and keep every structured change proposal-only."
)
SYSTEM_POLICY_VERSION = "vf-synthetic-policy-v1"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _revision(material: Any) -> str:
    digest = hashlib.sha256(_canonical(material).encode("utf-8")).hexdigest()
    return f"synthetic:sha256:{digest}"


def _evidence(
    evidence_id: str,
    tool_name: str,
    tool_version: str,
    revision: str,
    summary: str,
    payload: Mapping[str, Any],
    *,
    status: str = "complete",
) -> dict[str, Any]:
    return {
        "type": "tool-evidence",
        "evidenceId": evidence_id,
        "toolName": tool_name,
        "toolVersion": tool_version,
        "status": status,
        "authority": "deterministic",
        "sourceProjectRevision": revision,
        "summary": summary,
        "payload": dict(payload),
    }


def _citation(record_id: str, evidence_refs: Iterable[str]) -> dict[str, Any]:
    result = get_electronics_corpus().record(record_id)
    if result.status != "found":
        raise ValueError(f"Cannot cite missing knowledge record {record_id}")
    record = result.records[0]
    return {
        "type": "citation",
        "citationId": "citation:" + hashlib.sha256(record_id.encode("utf-8")).hexdigest()[:24],
        "sourceId": "vf-src-curated-electronics-corpus-v1",
        "sourceRevision": record["provenance"]["sourceRevision"],
        "title": record["subject"]["name"],
        "locator": record_id,
        "contentSha256": content_sha256(record),
        "evidenceRefs": list(evidence_refs),
    }


def _record(
    *,
    task: str,
    user: str,
    board: str | None,
    context_payload: Mapping[str, Any],
    evidence_inputs: list[dict[str, Any]],
    assistant_text: str | None = None,
    actions: list[dict[str, Any]] | None = None,
    citations: list[dict[str, Any]] | None = None,
    refusal: dict[str, Any] | None = None,
    uncertainty: dict[str, Any] | None = None,
) -> dict[str, Any]:
    revision = _revision(
        {
            "seed": DETERMINISTIC_SEED,
            "grammarVersion": GRAMMAR_VERSION,
            "task": task,
            "user": user,
            "board": board,
            "context": context_payload,
        }
    )
    evidence = []
    for item in evidence_inputs:
        evidence.append(
            _evidence(
                item["evidenceId"],
                item["toolName"],
                item["toolVersion"],
                revision,
                item["summary"],
                item["payload"],
                status=item.get("status", "complete"),
            )
        )
    for action in actions or []:
        action["sourceProjectRevision"] = revision
    output: dict[str, Any] = {
        "structuredActions": actions or [],
        "citations": citations or [],
    }
    refs = [item["evidenceId"] for item in evidence]
    if assistant_text is not None:
        output["assistantText"] = {
            "type": "assistant-text",
            "text": assistant_text,
            "evidenceRefs": refs,
        }
    if refusal is not None:
        output["refusal"] = refusal
    if uncertainty is not None:
        output["uncertainty"] = uncertainty
    record = {
        "schemaVersion": 1,
        "contractVersion": "1.0.0",
        "recordId": "vf-task-v1-" + "0" * 24,
        "recordKind": "example",
        "task": task,
        "input": {
            "system": {
                "type": "system",
                "text": SYSTEM_TEXT,
                "policyVersion": SYSTEM_POLICY_VERSION,
            },
            "user": {"type": "user", "text": user},
            "projectContext": {
                "type": "project-context",
                "projectId": "project:synthetic-v1",
                "sourceProjectRevision": revision,
                "revisionSource": "synthetic",
                **({"boardType": board} if board else {}),
                "payload": dict(context_payload),
            },
            "toolEvidence": evidence,
        },
        "output": output,
        "metadata": {
            "sourceKind": "synthetic",
            "sourceIds": list(SOURCE_IDS),
            "generator": {"id": GENERATOR_ID, "version": GENERATOR_VERSION},
        },
    }
    material = {key: value for key, value in record.items() if key != "recordId"}
    record["recordId"] = "vf-task-v1-" + hashlib.sha256(
        _canonical(material).encode("utf-8")
    ).hexdigest()[:24]
    return validate_task_record(record)


def _knowledge_input(receipt: Mapping[str, Any], summary: str) -> list[dict[str, Any]]:
    return [
        {
            "evidenceId": "evidence:knowledge:" + str(receipt["recordId"]).split(".")[-1],
            "toolName": "tool:electronics-corpus",
            "toolVersion": "1.0.0",
            "summary": summary,
            "payload": dict(receipt),
        }
    ]


def _board_records() -> list[dict[str, Any]]:
    records = []
    for board in SUPPORTED_BOARDS:
        receipt = verify_board(board)
        claims = receipt["claims"]
        response = (
            f"{receipt['name']} ({receipt['variant']}) uses {claims['mcu']} on "
            f"{claims['architecture']}, with {claims['logic-voltage-v']} V logic, "
            f"{claims['flash-bytes']} bytes of flash, and {claims['sram-bytes']} bytes of SRAM."
        )
        evidence = _knowledge_input(receipt, "Exact board variant and required claims matched the curated corpus.")
        citations = [_citation(receipt["recordId"], [evidence[0]["evidenceId"]])]
        for prompt in BOARD_PROMPTS:
            records.append(
                _record(
                    task="domain_chat",
                    user=prompt.format(board=board),
                    board=board,
                    context_payload={"requestedFacts": list(claims), "exactVariantRequired": True},
                    evidence_inputs=deepcopy(evidence),
                    assistant_text=response,
                    citations=deepcopy(citations),
                )
            )
    return records


def _pin_records() -> list[dict[str, Any]]:
    records = []
    for board in SUPPORTED_BOARDS:
        receipt = verify_pin_route(board)
        route = receipt["route"]
        evidence_id = "evidence:pin:" + hashlib.sha256(board.encode("utf-8")).hexdigest()[:20]
        evidence = [
            {
                "evidenceId": evidence_id,
                "toolName": "tool:pin-router",
                "toolVersion": "1.0.0",
                "summary": "The default I2C route matched the exact curated pin map.",
                "payload": receipt,
            }
        ]
        response = (
            f"Use {route['supply']} for supply, {route['ground']} for ground, "
            f"{route['sda']} for SDA, and {route['scl']} for SCL on the exact {board} variant."
        )
        for prompt in PIN_PROMPTS:
            records.append(
                _record(
                    task="pin_routing",
                    user=prompt.format(board=board),
                    board=board,
                    context_payload={"bus": "i2c", "exactVariantRequired": True},
                    evidence_inputs=deepcopy(evidence),
                    assistant_text=response,
                    citations=[_citation(receipt["pinMapRecordId"], [evidence_id])],
                )
            )
    return records


def _wiring_records() -> list[dict[str, Any]]:
    records = []
    for board in SUPPORTED_BOARDS:
        recipe, connections = expected_connections(board, COMPONENT_ALIAS)
        receipt = verify_netlist(board, COMPONENT_ALIAS, connections, expected_valid=True)
        evidence_id = "evidence:netlist:" + hashlib.sha256(board.encode("utf-8")).hexdigest()[:20]
        evidence = [
            {
                "evidenceId": evidence_id,
                "toolName": "tool:netlist-verifier",
                "toolVersion": "1.0.0",
                "summary": "The complete terminal-to-pin netlist exactly matched the curated recipe.",
                "payload": receipt,
            }
        ]
        response = "Connect " + "; ".join(
            f"{item['componentTerminal']} to {item['boardPin']}" for item in connections
        ) + ". Power off before wiring and confirm the exact board and OLED variants."
        for prompt in WIRING_PROMPTS:
            records.append(
                _record(
                    task="wiring",
                    user=prompt.format(board=board, component=COMPONENT_ALIAS),
                    board=board,
                    context_payload={"component": COMPONENT_ALIAS, "connections": connections},
                    evidence_inputs=deepcopy(evidence),
                    assistant_text=response,
                    citations=[_citation(recipe["recordId"], [evidence_id])],
                )
            )
    return records


def _mutated_netlists(board: str) -> list[tuple[str, list[dict[str, str]], bool]]:
    _, exact = expected_connections(board, COMPONENT_ALIAS)
    missing_ground = [item for item in exact if item["componentTerminal"] != "GND"]
    swapped = deepcopy(exact)
    sda = next(item for item in swapped if item["componentTerminal"] == "DATA/SDA")
    scl = next(item for item in swapped if item["componentTerminal"] == "CLK/SCL")
    sda["boardPin"], scl["boardPin"] = scl["boardPin"], sda["boardPin"]
    return [("exact", exact, True), ("missing_ground", missing_ground, False), ("swapped_i2c", swapped, False)]


def _circuit_records() -> list[dict[str, Any]]:
    records = []
    for board in SUPPORTED_BOARDS:
        for case_name, connections, expected_valid in _mutated_netlists(board):
            receipt = verify_netlist(
                board, COMPONENT_ALIAS, connections, expected_valid=expected_valid
            )
            evidence_id = "evidence:circuit:" + hashlib.sha256(
                f"{board}:{case_name}".encode("utf-8")
            ).hexdigest()[:20]
            if expected_valid:
                response = "The netlist is valid for the exact variants and matches all four curated connections."
            else:
                response = (
                    "The netlist is invalid. Missing: "
                    + (_canonical(receipt["missingConnections"]) if receipt["missingConnections"] else "none")
                    + "; unexpected: "
                    + (_canonical(receipt["unexpectedConnections"]) if receipt["unexpectedConnections"] else "none")
                    + "."
                )
            records.append(
                _record(
                    task="circuit_validation",
                    user=f"Validate the {case_name} OLED netlist on {board} and report only verified findings.",
                    board=board,
                    context_payload={"component": COMPONENT_ALIAS, "connections": connections},
                    evidence_inputs=[
                        {
                            "evidenceId": evidence_id,
                            "toolName": "tool:netlist-verifier",
                            "toolVersion": "1.0.0",
                            "summary": "The candidate netlist was compared to the exact curated recipe.",
                            "payload": receipt,
                        }
                    ],
                    assistant_text=response,
                    citations=[_citation(receipt["knowledgeRecordId"], [evidence_id])],
                )
            )
    return records


def _structured_records() -> list[dict[str, Any]]:
    records = []
    for board in SUPPORTED_BOARDS:
        for case_name, connections, expected_valid in _mutated_netlists(board)[1:]:
            receipt = verify_netlist(board, COMPONENT_ALIAS, connections, expected_valid=expected_valid)
            evidence_id = "evidence:structured:" + hashlib.sha256(
                f"{board}:{case_name}".encode("utf-8")
            ).hexdigest()[:20]
            actions = []
            for index, connection in enumerate(receipt["missingConnections"] + receipt["unexpectedConnections"]):
                actions.append(
                    {
                        "type": "structured-action",
                        "actionId": f"action:wire:{hashlib.sha256(f'{board}:{case_name}:{index}'.encode()).hexdigest()[:20]}",
                        "actionKind": "wire-suggestion",
                        "applicationMode": "proposal-only",
                        "requiresUserConfirmation": True,
                        "evidenceRefs": [evidence_id],
                        "payload": {
                            "component": COMPONENT_ALIAS,
                            "componentTerminal": connection["componentTerminal"],
                            "boardPin": connection["boardPin"],
                            "operation": "review-and-correct",
                        },
                    }
                )
            records.append(
                _record(
                    task="structured_output",
                    user=f"Return proposal-only structured corrections for the {case_name} netlist on {board}.",
                    board=board,
                    context_payload={"component": COMPONENT_ALIAS, "connections": connections},
                    evidence_inputs=[
                        {
                            "evidenceId": evidence_id,
                            "toolName": "tool:netlist-verifier",
                            "toolVersion": "1.0.0",
                            "summary": "Structured corrections are derived from exact missing and unexpected nets.",
                            "payload": receipt,
                        }
                    ],
                    assistant_text="I found a verified wiring mismatch and prepared proposal-only corrections for review.",
                    actions=actions,
                    citations=[_citation(receipt["knowledgeRecordId"], [evidence_id])],
                )
            )
    return records


def _firmware_records(receipts: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    records = []
    cases = [case for case in render_firmware_sources() if case["task"] == "firmware_generation"]
    for case in cases:
        receipt = validate_compiler_receipt(case, receipts[case["caseId"]])
        board_receipt = verify_board(str(case["board"]))
        evidence_id = "evidence:compile:" + receipt["receiptId"].split("-")[-1]
        records.append(
            _record(
                task="firmware_generation",
                user=f"Generate the verified {case['templateId']} sketch for {case['board']}.",
                board=str(case["board"]),
                context_payload={"templateId": case["templateId"], "fqbn": case["fqbn"]},
                evidence_inputs=[
                    {
                        "evidenceId": evidence_id,
                        "toolName": "tool:arduino-cli-compiler",
                        "toolVersion": "1.5.1+arduino-avr-1.8.6",
                        "summary": "The exact emitted sketch compiled successfully for the pinned target FQBN.",
                        "payload": receipt,
                    }
                ],
                assistant_text="```cpp\n" + str(case["source"]) + "```",
                citations=[_citation(board_receipt["recordId"], [evidence_id])],
            )
        )
    return records


def _repair_records(receipts: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    records = []
    cases = [case for case in render_firmware_sources() if case["task"] == "compiler_repair"]
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for case in cases:
        key = (str(case["board"]), str(case["templateId"]))
        grouped.setdefault(key, {})["fixed" if case["expectedSuccess"] else "broken"] = case
    for (board, repair_id), pair in sorted(grouped.items()):
        broken = pair["broken"]
        fixed = pair["fixed"]
        broken_receipt = validate_compiler_receipt(broken, receipts[broken["caseId"]])
        fixed_receipt = validate_compiler_receipt(fixed, receipts[fixed["caseId"]])
        broken_evidence_id = "evidence:compile:" + broken_receipt["receiptId"].split("-")[-1]
        fixed_evidence_id = "evidence:compile:" + fixed_receipt["receiptId"].split("-")[-1]
        action_id = "action:code-fix:" + hashlib.sha256(
            f"{board}:{repair_id}".encode("utf-8")
        ).hexdigest()[:20]
        records.append(
            _record(
                task="compiler_repair",
                user=(
                    f"Repair this {repair_id} compiler failure for {board}:\n```cpp\n"
                    + str(broken["source"])
                    + "```"
                ),
                board=board,
                context_payload={"repairId": repair_id, "fqbn": fixed["fqbn"]},
                evidence_inputs=[
                    {
                        "evidenceId": broken_evidence_id,
                        "toolName": "tool:arduino-cli-compiler",
                        "toolVersion": "1.5.1+arduino-avr-1.8.6",
                        "summary": "The original source failed in the pinned target compiler.",
                        "payload": broken_receipt,
                    },
                    {
                        "evidenceId": fixed_evidence_id,
                        "toolName": "tool:arduino-cli-compiler",
                        "toolVersion": "1.5.1+arduino-avr-1.8.6",
                        "summary": "The proposed corrected source compiled in the pinned target compiler.",
                        "payload": fixed_receipt,
                    },
                ],
                assistant_text="The corrected sketch is:\n```cpp\n" + str(fixed["source"]) + "```",
                actions=[
                    {
                        "type": "structured-action",
                        "actionId": action_id,
                        "actionKind": "code-fix",
                        "applicationMode": "proposal-only",
                        "requiresUserConfirmation": True,
                        "evidenceRefs": [broken_evidence_id, fixed_evidence_id],
                        "payload": {"repairId": repair_id, "replacementSource": fixed["source"]},
                    }
                ],
            )
        )
    return records


def _uncertainty_records() -> list[dict[str, Any]]:
    records = []
    prompt_patterns = (
        "Give me the exact pin map for {query}.",
        "Select a firmware target for {query} without asking for a variant.",
        "Assume a common {query} board and provide exact electrical limits.",
        "Wire an OLED to {query}; the exact hardware revision is unavailable.",
    )
    for query, missing in UNSUPPORTED_QUERIES:
        receipt = lookup_unknown_board(query)
        evidence_id = "evidence:unknown:" + hashlib.sha256(query.encode("utf-8")).hexdigest()[:20]
        for prompt in prompt_patterns:
            records.append(
                _record(
                    task="uncertainty",
                    user=prompt.format(query=query),
                    board=query,
                    context_payload={"requestedBoard": query, "exactVariantProvided": False},
                    evidence_inputs=[
                        {
                            "evidenceId": evidence_id,
                            "toolName": "tool:electronics-corpus",
                            "toolVersion": "1.0.0",
                            "summary": "The generic board query did not resolve to one exact supported variant.",
                            "payload": receipt,
                            "status": "reported",
                        }
                    ],
                    assistant_text=(
                        f"I cannot provide an exact pin or electrical answer for {query} without the {missing}. "
                        "Select the exact hardware first; I will not substitute a likely variant."
                    ),
                    uncertainty={
                        "type": "uncertainty",
                        "level": "high",
                        "reason": "The hardware label maps to multiple electrically distinct variants.",
                        "missingEvidence": [missing],
                    },
                )
            )
    return records


def _refusal_records() -> list[dict[str, Any]]:
    records = []
    receipt = safety_receipt()
    evidence_id = "evidence:safety:" + hashlib.sha256(
        str(receipt["recordId"]).encode("utf-8")
    ).hexdigest()[:20]
    for index, board in enumerate(SUPPORTED_BOARDS):
        records.append(
            _record(
                task="refusal",
                user=UNSAFE_PROMPTS[index % len(UNSAFE_PROMPTS)].format(board=board),
                board=board,
                context_payload={"energyClass": "unknown-high-energy", "isolationEvidence": False},
                evidence_inputs=[
                    {
                        "evidenceId": evidence_id,
                        "toolName": "tool:safety-rule-verifier",
                        "toolVersion": "1.0.0",
                        "summary": "The request triggered the curated unknown high-energy safety rule.",
                        "payload": receipt,
                    }
                ],
                refusal={
                    "type": "refusal",
                    "reasonCode": "unsafe-request",
                    "message": (
                        "I cannot provide direct high-energy wiring instructions without verified ratings, "
                        "isolation, protection, and a qualified review."
                    ),
                    "recoverable": True,
                },
                uncertainty={
                    "type": "uncertainty",
                    "level": "unknown",
                    "reason": "Voltage, current, isolation, enclosure, and protection evidence are missing.",
                    "missingEvidence": ["load ratings", "isolation design", "protection design"],
                },
                citations=[_citation(receipt["recordId"], [evidence_id])],
            )
        )
    return records


def generate_candidates(
    compile_receipts: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate verified candidates plus explicit pre-generation rejection decisions."""

    records = [
        *_board_records(),
        *_pin_records(),
        *_wiring_records(),
        *_circuit_records(),
        *_structured_records(),
        *_firmware_records(compile_receipts),
        *_repair_records(compile_receipts),
        *_uncertainty_records(),
        *_refusal_records(),
    ]
    # Exercise the exact duplicate gate with one deterministic candidate; it must never enter a shard.
    records.append(deepcopy(records[0]))
    rejections = [
        {
            "candidate": f"firmware:{board}",
            "reasonCode": "firmware-target-not-in-v1-pinned-toolchain",
            "detail": "No firmware record is generated until that exact board core is version-pinned and compiled.",
        }
        for board in NON_AVR_FIRMWARE_BOARDS
    ]
    return records, rejections


def expected_task_names() -> set[str]:
    return {
        "domain_chat",
        "pin_routing",
        "wiring",
        "circuit_validation",
        "structured_output",
        "firmware_generation",
        "compiler_repair",
        "uncertainty",
        "refusal",
    }
