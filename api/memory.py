"""Gateway-authenticated memory identity and chat-context integration."""

from __future__ import annotations

from dataclasses import dataclass
import hmac
import sqlite3
from typing import Any

from fastapi import HTTPException, status

from api.schemas import ChatRequest
from config import AiSettings
from memory_store import (
    BoundedMemoryStore,
    MemoryContractError,
    MemoryScope,
    MemoryStoreError,
    get_memory_store,
)


@dataclass(frozen=True)
class ChatMemoryBinding:
    store: BoundedMemoryStore
    scope: MemoryScope
    project_revision: str | None
    metadata: dict[str, Any]

    def record_turn(self, user_message: str, assistant_reply: str) -> bool:
        return self.store.record_turn(
            self.scope,
            self.project_revision,
            user_message,
            assistant_reply,
        )

    def refreshed_metadata(self) -> dict[str, Any]:
        state = self.store.inspect(
            self.scope, self.project_revision, context_only=True
        )
        return self.store.public_metadata(state)


def authenticated_memory_scope(
    settings: AiSettings,
    *,
    user_id: str | None,
    project_id: str | None,
    session_id: str | None,
) -> tuple[BoundedMemoryStore, MemoryScope]:
    if not settings.bounded_memory_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "MEMORY_GLOBALLY_DISABLED", "message": "Bounded memory is disabled."},
        )
    if not settings.api_token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "MEMORY_GATEWAY_AUTH_REQUIRED",
                "message": "Private gateway authentication must be configured for memory.",
            },
        )
    if not user_id or not project_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "MEMORY_AUTHENTICATED_SCOPE_REQUIRED",
                "message": "Authenticated user and project identity are required.",
            },
        )
    try:
        scope = MemoryScope.from_identifiers(user_id, project_id, session_id)
        store = get_memory_store(settings.memory_database_path)
    except (MemoryContractError, MemoryStoreError, OSError, sqlite3.Error) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": getattr(error, "code", "MEMORY_SCOPE_INVALID"),
                "message": str(error),
            },
        ) from error
    return store, scope


def prepare_chat_memory(
    request: ChatRequest,
    settings: AiSettings,
    *,
    user_id: str | None,
    project_id: str | None,
    session_id: str | None,
) -> tuple[ChatRequest, ChatMemoryBinding | None, dict[str, Any]]:
    unavailable = {
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
    # Caller-provided memory is never accepted as approved persisted memory.
    sanitized_request = request.model_copy(update={"memory": []})
    if (
        not settings.bounded_memory_enabled
        or not settings.api_token
        or not user_id
        or not project_id
        or not request.projectId
    ):
        return sanitized_request, None, unavailable
    if not hmac.compare_digest(project_id, request.projectId):
        return (
            sanitized_request,
            None,
            {**unavailable, "reasonCode": "MEMORY_PROJECT_SCOPE_MISMATCH"},
        )
    if request.sessionId and (
        not session_id or not hmac.compare_digest(session_id, request.sessionId)
    ):
        return (
            sanitized_request,
            None,
            {**unavailable, "reasonCode": "MEMORY_SESSION_SCOPE_MISMATCH"},
        )
    try:
        store, scope = authenticated_memory_scope(
            settings,
            user_id=user_id,
            project_id=project_id,
            session_id=session_id,
        )
        entries, metadata = store.context_entries(scope, request.projectRevision)
    except HTTPException:
        return sanitized_request, None, unavailable
    runtime_request = request.model_copy(update={"memory": entries})
    binding = ChatMemoryBinding(store, scope, request.projectRevision, metadata)
    return runtime_request, binding, metadata


def memory_http_error(error: MemoryStoreError) -> HTTPException:
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    if error.code == "MEMORY_NOT_FOUND":
        status_code = status.HTTP_404_NOT_FOUND
    elif error.code in {"MEMORY_DISABLED", "MEMORY_VERSION_CONFLICT"}:
        status_code = status.HTTP_409_CONFLICT
    return HTTPException(
        status_code=status_code,
        detail={"code": error.code, "message": error.message},
    )
