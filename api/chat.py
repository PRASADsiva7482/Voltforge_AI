"""Local-only grounded chat orchestration and SSE event framing."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable
from functools import lru_cache
import json
import logging
import time
import uuid

from api.copilot import prepare_grounded_context
from api.schemas import ChatRequest, ChatResponse
from config import AiSettings, get_settings
from engine.reasoning import ElectronicsReasoningOrchestrator
from model.artifact_registry import get_artifact_health
from task_schema.adapters import runtime_response_to_task_record


logger = logging.getLogger("voltforge-ai.chat")


def sse_event(name: str, payload: dict[str, object]) -> str:
    data = {"type": name, **payload}
    return f"event: {name}\ndata: {json.dumps(data, separators=(',', ':'), default=str)}\n\n"


@lru_cache(maxsize=2)
def _get_local_orchestrator(internet_retrieval_enabled: bool) -> ElectronicsReasoningOrchestrator:
    return ElectronicsReasoningOrchestrator(
        internet_retrieval_enabled=internet_retrieval_enabled
    )


async def stream_chat_sse(
    payload: ChatRequest,
    settings: AiSettings | None = None,
) -> AsyncIterator[str]:
    """Stream local deterministic output; model generation never calls a host."""
    configured = settings or get_settings()
    request_id = str(uuid.uuid4())
    session_id = payload.sessionId or str(uuid.uuid4())
    started = time.perf_counter()
    model_health = get_artifact_health()
    model = "voltforge-local-engine-v1"
    mode = "local-deterministic"

    yield sse_event(
        "start",
        {
            "requestId": request_id,
            "sessionId": session_id,
            "model": model,
            "mode": mode,
            "generationNetworkAccess": False,
            "neuralModelReady": bool(model_health["ready"]),
            "neuralModelCode": model_health["code"],
            "internetRetrieval": {
                "enabled": configured.internet_retrieval_enabled,
                "policy": "optional-evidence-only",
                "generationDependency": False,
            },
        },
    )

    grounded = await asyncio.to_thread(prepare_grounded_context, payload, configured)
    typed_evidence = grounded.task_record["input"]["toolEvidence"]
    for index, tool in enumerate(grounded.tool_events):
        typed = typed_evidence[index]
        yield sse_event(
            "tool",
            {
                "requestId": request_id,
                **tool,
                "evidenceId": typed["evidenceId"],
                "toolVersion": typed["toolVersion"],
                "authority": typed["authority"],
                "sourceProjectRevision": typed["sourceProjectRevision"],
            },
        )
    if grounded.proposal is not None:
        yield sse_event("proposal", {"requestId": request_id, **grounded.proposal})

    metadata = dict(grounded.response_metadata)
    try:
        deterministic = await asyncio.to_thread(
            _deterministic_response,
            payload,
            configured.internet_retrieval_enabled,
        )
        response_text = deterministic.reply.strip()
        metadata = _merge_response_metadata(metadata, deterministic)
        response_record = runtime_response_to_task_record(
            grounded.task_record,
            response_text=response_text,
            metadata=metadata,
            proposal=grounded.proposal,
        )
        for delta in _text_chunks(response_text):
            yield sse_event("delta", {"requestId": request_id, "delta": delta})

        if configured.store_conversations:
            await asyncio.to_thread(
                _save_exchange,
                payload,
                session_id,
                response_text,
                metadata,
                int((time.perf_counter() - started) * 1_000),
            )

        yield sse_event(
            "complete",
            {
                "requestId": request_id,
                "sessionId": session_id,
                "messageId": str(uuid.uuid4()),
                "model": model,
                "mode": mode,
                "characters": len(response_text),
                "reply": response_text,
                "taskRecordId": response_record["recordId"],
                "sourceProjectRevision": response_record["input"]["projectContext"]["sourceProjectRevision"],
                "hasCode": bool(metadata.get("generatedCode")),
                **metadata,
            },
        )
    except Exception:
        logger.exception("Local AI chat stream failure (requestId=%s)", request_id)
        yield sse_event(
            "error",
            {
                "requestId": request_id,
                "sessionId": session_id,
                "code": "LOCAL_GENERATION_FAILED",
                "message": "The local VoltForge response could not complete. Please try again.",
                "retryable": True,
            },
        )


def _deterministic_response(
    payload: ChatRequest,
    internet_retrieval_enabled: bool,
) -> ChatResponse:
    orchestrator = _get_local_orchestrator(internet_retrieval_enabled)
    return orchestrator.process_chat(
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


def _merge_response_metadata(
    metadata: dict[str, object], response: ChatResponse
) -> dict[str, object]:
    merged = dict(metadata)
    for key in (
        "citations",
        "wireSuggestions",
        "additions",
        "removals",
        "valueChanges",
        "codeFixes",
    ):
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


def _save_exchange(
    payload: ChatRequest,
    session_id: str,
    response_text: str,
    metadata: dict[str, object],
    latency_ms: int,
) -> None:
    try:
        from api.database import DatabaseManager

        DatabaseManager().save_chat_turn(
            session_id=session_id,
            project_id=payload.projectId,
            user_id=None,
            user_message=payload.message,
            assistant_reply=response_text,
            board_type=payload.boardType or "ARDUINO_UNO",
            confidence=float(metadata.get("confidence") or 0.0),
            generated_code=str(metadata.get("generatedCode") or "") or None,
            citations=list(metadata.get("citations") or []),
            suggested_actions={
                key: metadata.get(key) or []
                for key in (
                    "wireSuggestions",
                    "additions",
                    "removals",
                    "valueChanges",
                    "codeFixes",
                )
            },
            latency_ms=latency_ms,
        )
    except Exception:
        # Conversation persistence is optional and must never break a response.
        logger.exception("Could not persist local AI chat exchange")
