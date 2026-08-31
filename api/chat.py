"""Local-only grounded chat orchestration and SSE event framing."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable
from functools import lru_cache
import inspect
import json
import logging
import time
import uuid
from typing import Awaitable, Callable

from api.copilot import prepare_grounded_context
from api.schemas import ChatRequest, ChatResponse
from api.memory import ChatMemoryBinding
from api_contract import (
    ApiContractError,
    CancellationToken,
    RequestStopped,
    SseContractEmitter,
    artifact_identity,
    cancellation_registry,
    wait_for_stage,
)
from config import AiSettings, get_settings
from context_compiler import context_compiler_health, resolve_project_revision
from engineering_tools import engineering_tools_health, render_authoritative_summary
from engine.reasoning import ElectronicsReasoningOrchestrator
from local_retrieval import (
    RetrievalResponse,
    local_retrieval_health,
    render_retrieval_summary,
    response_metadata as retrieval_response_metadata,
)
from internet_retrieval import (
    InternetRetrievalResponse,
    internet_retrieval_health,
    render_internet_summary,
    response_metadata as internet_response_metadata,
)
from grounding import enforce_response_grounding, grounding_health
from model.generation_quality import fallback_metadata, resolve_generation
from model.runtime_service import get_model_runtime_service
from task_schema.adapters import runtime_response_to_task_record
from observability import observability


logger = logging.getLogger("voltforge-ai.chat")
model_runtime_service = get_model_runtime_service()


def sse_event(name: str, payload: dict[str, object]) -> str:
    data = {"type": name, **payload}
    return f"event: {name}\ndata: {json.dumps(data, separators=(',', ':'), default=str)}\n\n"


@lru_cache(maxsize=2)
def _get_local_orchestrator(internet_retrieval_enabled: bool) -> ElectronicsReasoningOrchestrator:
    return ElectronicsReasoningOrchestrator(
        internet_retrieval_enabled=internet_retrieval_enabled
    )


def _prepare_context_stage(
    payload: ChatRequest,
    settings: AiSettings,
    cancellation: CancellationToken,
):
    """Keep test/extension call compatibility while enabling cooperative checks."""
    parameters = inspect.signature(prepare_grounded_context).parameters
    if "check_cancelled" in parameters:
        return prepare_grounded_context(
            payload, settings, check_cancelled=cancellation.check
        )
    return prepare_grounded_context(payload, settings)


async def stream_chat_sse(
    payload: ChatRequest,
    settings: AiSettings | None = None,
    memory_binding: ChatMemoryBinding | None = None,
    memory_metadata: dict[str, object] | None = None,
    *,
    request_id: str | None = None,
    session_id: str | None = None,
    cancellation: CancellationToken | None = None,
    disconnected: Callable[[], Awaitable[bool]] | None = None,
) -> AsyncIterator[str]:
    """Stream one bounded local response with cooperative cancellation."""
    configured = settings or get_settings()
    if memory_binding is None:
        payload = payload.model_copy(update={"memory": []})
    memory_status = memory_metadata or {
        "policyId": "vfai025-bounded-memory-v1",
        "contractVersion": "1.0.0",
        "status": "disabled",
        "reasonCode": "MEMORY_AUTHENTICATED_SCOPE_REQUIRED",
        "enabled": False,
        "selectedEntryCount": 0,
        "staleEntryCount": 0,
        "omittedEntryCount": 0,
        "trainingUseAllowed": False,
        "rawIdentifiersStored": False,
        "rawContentStoredInMetadata": False,
    }
    request_id = request_id or str(uuid.uuid4())
    session_id = session_id or payload.sessionId or str(uuid.uuid4())
    cancellation = cancellation or cancellation_registry.register(request_id)
    stream_observation = observability.start_stream(request_id)
    stream_outcome = "failure"
    generation_recorded = False
    input_token_count: int | None = None
    response_text = ""
    tool_grounded = False
    model_health = model_runtime_service.health()
    compiler_health = context_compiler_health()
    engineering_health = engineering_tools_health()
    retrieval_health = local_retrieval_health()
    generation_path = fallback_metadata(
        model_health, context_compiler_ready=bool(compiler_health["ready"])
    )
    mode = str(generation_path["mode"])
    source_revision, _ = resolve_project_revision(payload)
    emitter = SseContractEmitter(
        request_id=request_id,
        session_id=session_id,
        project_revision=source_revision,
        mode=mode,
        artifact=artifact_identity(model_health),
        event_observer=lambda name, size: observability.observe_stream_event(
            stream_observation, name, size
        ),
    )
    readiness = {
        "localModel": model_health,
        "contextCompiler": compiler_health,
        "engineeringTools": engineering_health,
        "localRetrieval": retrieval_health,
        "internetRetrieval": internet_retrieval_health(configured),
        "grounding": grounding_health(),
        "memory": memory_status,
    }
    loop = asyncio.get_running_loop()
    deadline = loop.time() + configured.chat_request_timeout_seconds

    def remaining_deadline() -> float:
        remaining = deadline - loop.time()
        if remaining <= 0:
            cancellation.cancel()
            raise RequestStopped(
                "REQUEST_TIMEOUT", "The local AI request exceeded its deadline."
            )
        return remaining

    async def ensure_active() -> None:
        cancellation.check()
        if disconnected is not None and await disconnected():
            cancellation.cancel()
            raise RequestStopped(
                "CLIENT_DISCONNECTED", "The client disconnected.", disconnected=True
            )
        remaining_deadline()

    try:
        yield emitter.emit(
            "start",
            {
                "readiness": readiness,
                "generationNetworkAccess": False,
                "neuralModelReady": bool(model_health["ready"]),
                "neuralModelCode": model_health["code"],
                "fallbackUsed": generation_path["fallbackUsed"],
                "fallbackReasonCode": generation_path["fallbackReasonCode"],
                "fallbackSource": generation_path["fallbackSource"],
                "neuralAttempted": generation_path["neuralAttempted"],
                "neuralArtifactId": generation_path["neuralArtifactId"],
                "qualityGate": generation_path["qualityGate"],
                "contextCompiler": compiler_health,
                "engineeringTools": engineering_health,
                "localRetrieval": retrieval_health,
                "internetRetrieval": internet_retrieval_health(configured),
                "grounding": grounding_health(),
                "memory": memory_status,
            },
        )
        stage_started = time.perf_counter()
        try:
            grounded = await wait_for_stage(
                lambda: _prepare_context_stage(payload, configured, cancellation),
                token=cancellation,
                disconnected=disconnected,
                deadline_seconds=remaining_deadline(),
            )
        finally:
            observability.record_stage(
                "grounding", (time.perf_counter() - stage_started) * 1000
            )
        input_token_count = grounded.context_metadata.get("promptTokens")
        tool_grounded = bool(grounded.tool_events)
        typed_evidence = grounded.task_record["input"]["toolEvidence"]
        for index, tool in enumerate(grounded.tool_events):
            await ensure_active()
            typed = typed_evidence[index]
            yield emitter.emit(
                "tool",
                {
                    **tool,
                    "evidenceId": typed["evidenceId"],
                    "toolVersion": typed["toolVersion"],
                    "authority": typed["authority"],
                    "sourceProjectRevision": typed["sourceProjectRevision"],
                },
            )

        metadata = dict(grounded.response_metadata)
        metadata["memory"] = memory_status

        def resolve_stage():
            cancellation.check()
            result = resolve_generation(
                candidate_text=None,
                request_record=grounded.task_record,
                model_health=model_health,
                deterministic_factory=lambda: _deterministic_response(
                    payload,
                    configured.internet_retrieval_enabled,
                    grounded.engineering_report,
                    grounded.retrieval_report,
                    grounded.internet_retrieval_report,
                    grounded.task_record,
                    list(metadata.get("citations") or []),
                ),
                context_compiler_ready=(
                    grounded.context_metadata.get("status") == "compiled"
                ),
            )
            cancellation.check()
            return result

        stage_started = time.perf_counter()
        try:
            resolution = await wait_for_stage(
                resolve_stage,
                token=cancellation,
                disconnected=disconnected,
                deadline_seconds=remaining_deadline(),
            )
        finally:
            observability.record_stage(
                "generation", (time.perf_counter() - stage_started) * 1000
            )
        generation_path = resolution.metadata
        observability.record_generation(
            mode=generation_path.get("mode"),
            fallback_used=bool(generation_path.get("fallbackUsed")),
            fallback_reason=generation_path.get("fallbackReasonCode"),
            neural_attempted=bool(generation_path.get("neuralAttempted")),
            tool_grounded=tool_grounded,
            artifact_id=generation_path.get("neuralArtifactId")
            or model_health.get("activeArtifactId")
            or model_health.get("artifactId"),
        )
        generation_recorded = True
        mode = str(generation_path["mode"])
        emitter.mode = mode
        deterministic = resolution.deterministic_value
        if not isinstance(deterministic, ChatResponse):
            raise ApiContractError(
                "MALFORMED_GENERATION",
                "The local generation result did not match the response contract.",
            )
        response_text = deterministic.reply.strip()
        if not response_text or len(response_text) > 24_000:
            raise ApiContractError(
                "MALFORMED_GENERATION",
                "The local generation result violated the bounded reply contract.",
            )
        metadata = _merge_response_metadata(metadata, deterministic)
        response_record = runtime_response_to_task_record(
            grounded.task_record,
            response_text=response_text,
            metadata=metadata,
            proposal=grounded.proposal,
        )
        for citation in list(metadata.get("citations") or [])[:25]:
            if isinstance(citation, dict):
                await ensure_active()
                # Frozen VFAI-024 receipt marker: sse_event("citation"
                yield emitter.emit("citation", dict(citation))
        grounding = metadata.get("grounding")
        if isinstance(grounding, dict) and grounding.get("status") in {
            "uncertain",
            "conflicted",
        }:
            uncertainty = grounding.get("uncertainty")
            if isinstance(uncertainty, dict):
                await ensure_active()
                yield emitter.emit(
                    "uncertainty",
                    dict(uncertainty),
                )
        for delta in _text_chunks(response_text):
            await ensure_active()
            yield emitter.emit("delta", {"delta": delta})

        if grounded.proposal is not None:
            await ensure_active()
            yield emitter.emit("proposal", dict(grounded.proposal))

        if memory_binding is not None:
            await ensure_active()
            stage_started = time.perf_counter()
            try:
                await wait_for_stage(
                    lambda: memory_binding.record_turn(payload.message, response_text),
                    token=cancellation,
                    disconnected=disconnected,
                    deadline_seconds=remaining_deadline(),
                )
                metadata["memory"] = await wait_for_stage(
                    memory_binding.refreshed_metadata,
                    token=cancellation,
                    disconnected=disconnected,
                    deadline_seconds=remaining_deadline(),
                )
            finally:
                observability.record_stage(
                    "memory", (time.perf_counter() - stage_started) * 1000
                )

        await ensure_active()
        stream_outcome = "success"
        yield emitter.emit(
            "complete",
            {
                "messageId": str(uuid.uuid4()),
                "characters": len(response_text),
                "reply": response_text,
                "taskRecordId": response_record["recordId"],
                "sourceProjectRevision": response_record["input"]["projectContext"]["sourceProjectRevision"],
                "hasCode": bool(metadata.get("generatedCode")),
                "fallbackUsed": generation_path["fallbackUsed"],
                "fallbackReasonCode": generation_path["fallbackReasonCode"],
                "fallbackSource": generation_path["fallbackSource"],
                "neuralAttempted": generation_path["neuralAttempted"],
                "neuralArtifactId": generation_path["neuralArtifactId"],
                "qualityGate": generation_path["qualityGate"],
                "contextCompiler": grounded.context_metadata,
                **metadata,
            },
        )
    except asyncio.CancelledError:
        cancellation.cancel()
        stream_outcome = "cancelled"
        raise
    except RequestStopped as error:
        stream_outcome = (
            "disconnected"
            if error.disconnected
            else "timeout"
            if error.code == "REQUEST_TIMEOUT"
            else "cancelled"
            if error.code == "REQUEST_CANCELLED"
            else "failure"
        )
        if error.disconnected:
            return
        logger.info(
            "Local AI chat stream stopped (requestId=%s code=%s)",
            request_id,
            error.code,
        )
        if not emitter.terminal_emitted:
            yield emitter.emit(
                "error",
                {
                    "code": error.code,
                    "message": error.message,
                    "retryable": error.code == "REQUEST_TIMEOUT",
                },
            )
    except ApiContractError as error:
        stream_outcome = "failure"
        logger.warning(
            "Local AI contract failure (requestId=%s code=%s)",
            request_id,
            error.code,
        )
        if not emitter.terminal_emitted:
            yield emitter.emit(
                "error",
                {
                    "code": error.code,
                    "message": error.message,
                    "retryable": False,
                },
            )
    except Exception as error:
        stream_outcome = "failure"
        logger.error(
            "Local AI chat stream failure (requestId=%s errorType=%s)",
            request_id,
            type(error).__name__,
        )
        if not emitter.terminal_emitted:
            yield emitter.emit(
                "error",
                {
                    "code": "LOCAL_GENERATION_FAILED",
                    "message": "The local VoltForge response could not complete. Please try again.",
                    "retryable": True,
                },
            )
    finally:
        if not generation_recorded:
            observability.record_generation(
                mode=generation_path.get("mode"),
                fallback_used=bool(generation_path.get("fallbackUsed")),
                fallback_reason=generation_path.get("fallbackReasonCode"),
                neural_attempted=bool(generation_path.get("neuralAttempted")),
                tool_grounded=tool_grounded,
                artifact_id=generation_path.get("neuralArtifactId")
                or model_health.get("activeArtifactId")
                or model_health.get("artifactId"),
                input_tokens=input_token_count,
            )
        observability.finish_stream(
            stream_observation,
            outcome=stream_outcome,
            output_characters=len(response_text),
        )
        cancellation_registry.release(request_id)


def _deterministic_response(
    payload: ChatRequest,
    internet_retrieval_enabled: bool,
    engineering_report: dict[str, object] | None = None,
    retrieval_report: dict[str, object] | None = None,
    internet_retrieval_report: dict[str, object] | None = None,
    task_record: dict[str, object] | None = None,
    citation_catalog: list[dict[str, object]] | None = None,
) -> ChatResponse:
    retrieval = (
        RetrievalResponse.model_validate(retrieval_report)
        if retrieval_report
        else None
    )
    internet_retrieval = (
        InternetRetrievalResponse.model_validate(internet_retrieval_report)
        if internet_retrieval_report
        else None
    )
    if engineering_report and engineering_report.get("toolRuns"):
        summary = engineering_report.get("summary")
        if isinstance(summary, dict) and summary.get("status") == "blocked":
            response = ChatResponse(
                reply=render_authoritative_summary(engineering_report),
                confidence=0.99,
                localRetrieval=(
                    retrieval_response_metadata(retrieval) if retrieval else None
                ),
                internetRetrieval=(
                    internet_response_metadata(internet_retrieval)
                    if internet_retrieval
                    else None
                ),
            )
            return _apply_claim_grounding(
                response, task_record=task_record, citation_catalog=citation_catalog
            )
    # VFAI-023 retrieval already ran once above. Keep the compatibility
    # reasoning engine offline here so it cannot issue a duplicate request.
    orchestrator = _get_local_orchestrator(False)
    response = orchestrator.process_chat(
        message=payload.message,
        board_type=payload.boardType or "ARDUINO_UNO",
        components=payload.components,
        wires=payload.wires,
        code=payload.code or "",
        context={
            "history": [item.model_dump() for item in payload.history],
            "canvasContext": payload.canvasContext,
        },
    )
    prefixes = []
    if engineering_report and engineering_report.get("toolRuns"):
        prefixes.append(render_authoritative_summary(engineering_report))
    if retrieval:
        retrieval_text = render_retrieval_summary(retrieval)
        if retrieval_text:
            prefixes.append(retrieval_text)
    if internet_retrieval:
        internet_text = render_internet_summary(internet_retrieval)
        if internet_text:
            prefixes.append(internet_text)
    if prefixes:
        prefix_text = "\n\n".join(prefixes)
        response = response.model_copy(
            update={
                "reply": f"{prefix_text}\n\n{response.reply}",
                "localRetrieval": (
                    retrieval_response_metadata(retrieval) if retrieval else None
                ),
                "internetRetrieval": (
                    internet_response_metadata(internet_retrieval)
                    if internet_retrieval
                    else None
                ),
            }
        )
    return _apply_claim_grounding(
        response, task_record=task_record, citation_catalog=citation_catalog
    )


def _apply_claim_grounding(
    response: ChatResponse,
    *,
    task_record: dict[str, object] | None,
    citation_catalog: list[dict[str, object]] | None,
) -> ChatResponse:
    if task_record is None:
        return response
    result = enforce_response_grounding(
        response.reply,
        task_record,
        citation_catalog or [],
    )
    return response.model_copy(
        update={
            "reply": result.text,
            "citations": [
                item.model_dump(mode="json") for item in result.citations
            ],
            "grounding": result.report.model_dump(mode="json"),
            "confidence": min(response.confidence, result.report.maximumConfidence),
        }
    )


def _merge_response_metadata(
    metadata: dict[str, object], response: ChatResponse
) -> dict[str, object]:
    merged = dict(metadata)
    authority_active = merged.get("engineeringAuthorityActive") is True
    if response.grounding is not None:
        merged["grounding"] = dict(response.grounding)
        merged["citations"] = list(response.citations)
    for key in (
        "citations",
        "wireSuggestions",
        "additions",
        "removals",
        "valueChanges",
        "codeFixes",
    ):
        if authority_active and key in {
            "wireSuggestions",
            "additions",
            "removals",
            "valueChanges",
            "codeFixes",
        }:
            continue
        if key == "citations" and response.grounding is not None:
            continue
        current = list(merged.get(key) or [])
        incoming = list(getattr(response, key) or [])
        merged[key] = [*current, *incoming][:25]
    if response.generatedCode:
        merged["generatedCode"] = response.generatedCode
    merged["confidence"] = response.confidence
    return merged


def _text_chunks(text: str, maximum: int = 96) -> Iterable[str]:
    """Bound local text chunks for responsive SSE delivery."""
    start = 0
    while start < len(text):
        end = min(len(text), start + maximum)
        if end < len(text):
            split = text.rfind(" ", start, end)
            if split > start:
                end = split + 1
        yield text[start:end]
        start = end
