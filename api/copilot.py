"""Bounded deterministic evidence and proposals for streamed AI chat."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Callable

from api.schemas import ChatRequest
from config import AiSettings
from context_compiler import ContextCompilerError, compile_project_context
from engineering_tools import (
    EngineeringContractError,
    engineering_tools_health,
    run_authoritative_engineering_checks,
    tool_events_from_report,
)
from local_retrieval import (
    citations_from_response,
    response_metadata as retrieval_response_metadata,
    retrieval_tool_event,
    search_for_request,
)
from internet_retrieval import (
    citations_from_response as internet_citations_from_response,
    internet_tool_event,
    response_metadata as internet_response_metadata,
    search_for_request as search_internet_for_request,
)
from grounding import public_citation_catalog
from task_schema.adapters import runtime_request_to_task_record


@dataclass(frozen=True)
class GroundedContext:
    prompt_context: str
    tool_events: list[dict[str, object]]
    task_record: dict[str, object]
    response_metadata: dict[str, object] = field(default_factory=dict)
    proposal: dict[str, object] | None = None
    context_metadata: dict[str, object] = field(default_factory=dict)
    engineering_report: dict[str, object] = field(default_factory=dict)
    retrieval_report: dict[str, object] = field(default_factory=dict)
    internet_retrieval_report: dict[str, object] = field(default_factory=dict)


def prepare_grounded_context(
    request: ChatRequest,
    settings: AiSettings | None = None,
    check_cancelled: Callable[[], None] | None = None,
) -> GroundedContext:
    """Run allow-listed checks and construct model context from untrusted project data."""
    check = check_cancelled or (lambda: None)
    check()
    retrieval_response = search_for_request(request)
    check()
    retrieval_report = retrieval_response.model_dump(mode="json")
    local_retrieval_event = retrieval_tool_event(retrieval_response)
    internet_response = search_internet_for_request(request, retrieval_response, settings)
    check()
    internet_report = internet_response.model_dump(mode="json")
    internet_event = (
        internet_tool_event(internet_response)
        if internet_response.status != "not-requested"
        else None
    )
    engineering_events: list[dict[str, object]] = []
    try:
        check()
        report_model = run_authoritative_engineering_checks(request)
        check()
        engineering_report = report_model.model_dump(mode="json")
        engineering_events = tool_events_from_report(report_model)
    except (EngineeringContractError, ContextCompilerError) as error:
        health = engineering_tools_health()
        engineering_report = {
            "policyId": health["policyId"],
            "status": "unavailable",
            "code": getattr(error, "code", "ENGINEERING_TOOLS_UNAVAILABLE"),
            "criticalModelOverrideAllowed": False,
            "rawProjectContentStored": False,
        }
        engineering_events.append(
            {
                "name": "engineering-authority",
                "version": "1.0.0",
                "status": "unavailable",
                "authority": "deterministic",
                "summary": "Authoritative engineering checks were unavailable for this request.",
                "evidence": {
                    "policyId": health["policyId"],
                    "code": engineering_report["code"],
                    "blockingFindingIds": ["engineering-tools-unavailable"],
                    "modelOverrideAllowed": False,
                    "rawContentStored": False,
                },
            }
        )
    if engineering_events and engineering_events[0].get("name") == "engineering-authority-index":
        optional_retrieval_events = [local_retrieval_event]
        if internet_event is not None:
            optional_retrieval_events.append(internet_event)
        remaining = max(0, 8 - 1 - len(optional_retrieval_events))
        tool_events = [
            engineering_events[0],
            *optional_retrieval_events,
            *engineering_events[1 : 1 + remaining],
        ]
    else:
        tool_events = [*engineering_events, local_retrieval_event]
        if internet_event is not None:
            tool_events.append(internet_event)
        tool_events = tool_events[:8]

    try:
        check()
        compilation = compile_project_context(request, tool_events=tool_events)
        check()
        task_record = compilation.task_record
        prompt_context = compilation.prompt
        selected_tool_events = list(compilation.selected_tool_events)
        context_metadata = compilation.public_metadata
    except ContextCompilerError as error:
        # Deterministic tools remain available even when neural context cannot
        # be represented. This unbounded record is never passed to a model.
        task_record = runtime_request_to_task_record(
            request,
            tool_events=tool_events,
            project_payload={"contextCompilerUnavailable": True},
        )
        prompt_context = ""
        selected_tool_events = tool_events
        context_metadata = {
            "policyId": "vfai020-project-context-policy-v1",
            "status": "unavailable",
            "code": error.code,
            **error.details,
            "rawPromptStored": False,
            "rawProjectContextStored": False,
        }
    metadata = _engineering_response_metadata(
        engineering_report,
        {str(item.get("name")) for item in selected_tool_events},
    )
    metadata["localRetrieval"] = {
        **retrieval_response_metadata(retrieval_response),
        "selectedForModelContext": any(
            item.get("name") == "curated-local-retrieval"
            for item in selected_tool_events
        ),
    }
    metadata["internetRetrieval"] = {
        **internet_response_metadata(internet_response),
        "selectedForModelContext": any(
            item.get("name") == "secure-internet-evidence"
            for item in selected_tool_events
        ),
    }
    source_citations = [
        *citations_from_response(retrieval_response),
        *internet_citations_from_response(internet_response),
    ]
    metadata["citations"] = public_citation_catalog(task_record, source_citations)
    revision = str(task_record["input"]["projectContext"]["sourceProjectRevision"])
    evidence_refs = [
        str(item["evidenceId"])
        for item in task_record["input"]["toolEvidence"]
        if str(item.get("toolName", "")).startswith("tool:engineering")
    ]
    proposal = _proposal(metadata, revision, evidence_refs)
    check()
    return GroundedContext(
        prompt_context=prompt_context,
        tool_events=selected_tool_events,
        task_record=task_record,
        response_metadata=metadata,
        proposal=proposal,
        context_metadata=context_metadata,
        engineering_report=engineering_report,
        retrieval_report=retrieval_report,
        internet_retrieval_report=internet_report,
    )


def _engineering_response_metadata(
    report: dict[str, object], selected_tool_names: set[str]
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "wireSuggestions": [],
        "additions": [],
        "removals": [],
        "valueChanges": [],
        "codeFixes": [],
        "citations": [],
        "engineeringAuthorityActive": True,
    }
    if report.get("status") == "unavailable":
        metadata["engineeringAuthority"] = dict(report)
        metadata["engineeringFindings"] = []
        return metadata
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    metadata["engineeringAuthority"] = {
        "policyId": report.get("policyId"),
        "policySha256": report.get("policySha256"),
        "reportId": report.get("reportId"),
        "sourceProjectRevision": report.get("sourceProjectRevision"),
        "status": summary.get("status"),
        "toolRuns": summary.get("toolRuns"),
        "applicableToolRuns": summary.get("applicableToolRuns"),
        "findings": summary.get("findings"),
        "blockingFindings": summary.get("blockingFindings"),
        "criticalFindings": summary.get("criticalFindings"),
        "calculations": summary.get("calculations"),
        "criticalModelOverrideAllowed": False,
        "rawProjectContentStored": False,
    }
    findings: list[dict[str, object]] = []
    action_keys = {
        "wire-suggestion": "wireSuggestions",
        "component-addition": "additions",
        "component-removal": "removals",
        "value-change": "valueChanges",
        "code-fix": "codeFixes",
    }
    for run in report.get("toolRuns") or []:  # type: ignore[union-attr]
        if not isinstance(run, dict):
            continue
        findings.extend(
            item for item in run.get("findings") or [] if isinstance(item, dict)
        )
        if str(run.get("toolId")) not in selected_tool_names:
            continue
        for action in run.get("approvedActions") or []:
            if not isinstance(action, dict):
                continue
            target = action_keys.get(str(action.get("actionKind")))
            if target:
                values = list(metadata[target])  # type: ignore[arg-type]
                values.append(action.get("payload") or {})
                metadata[target] = values[:25]
    findings.sort(
        key=lambda item: (
            {"CRITICAL": 0, "HIGH": 1, "WARNING": 2, "UNKNOWN": 3, "INFO": 4}.get(
                str(item.get("severity")), 9
            ),
            str(item.get("findingId")),
        )
    )
    metadata["engineeringFindings"] = findings[:50]
    metadata["omittedEngineeringFindingCount"] = max(0, len(findings) - 50)
    return metadata


def _proposal(
    metadata: dict[str, object],
    source_project_revision: str,
    evidence_refs: list[str],
) -> dict[str, object] | None:
    action_keys = ("wireSuggestions", "additions", "removals", "valueChanges", "codeFixes")
    count = sum(len(metadata.get(key) or []) for key in action_keys)  # type: ignore[arg-type]
    if not count:
        return None
    action_kinds = {
        "wireSuggestions": "wire-suggestion",
        "additions": "component-addition",
        "removals": "component-removal",
        "valueChanges": "value-change",
        "codeFixes": "code-fix",
    }
    structured_actions = []
    for key, action_kind in action_kinds.items():
        for item in metadata.get(key) or []:  # type: ignore[union-attr]
            identity = json.dumps(
                {
                    "revision": source_project_revision,
                    "kind": action_kind,
                    "payload": item,
                    "evidenceRefs": sorted(evidence_refs),
                },
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
            structured_actions.append(
                {
                    "type": "structured-action",
                    "actionId": f"action:engineering:{hashlib.sha256(identity).hexdigest()[:20]}",
                    "actionKind": action_kind,
                    "sourceProjectRevision": source_project_revision,
                    "applicationMode": "proposal-only",
                    "requiresUserConfirmation": True,
                    "evidenceRefs": evidence_refs,
                    "payload": item if isinstance(item, dict) else {"value": item},
                }
            )
    proposal_identity = json.dumps(
        structured_actions, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return {
        "id": f"proposal:engineering:{hashlib.sha256(proposal_identity).hexdigest()[:20]}",
        "source": "deterministic-validation",
        "summary": f"{count} reviewable project action(s) are available; none have been applied.",
        "sourceProjectRevision": source_project_revision,
        "applicationMode": "proposal-only",
        "requiresUserConfirmation": True,
        "structuredActions": structured_actions,
        **{key: metadata.get(key) or [] for key in action_keys},
    }
