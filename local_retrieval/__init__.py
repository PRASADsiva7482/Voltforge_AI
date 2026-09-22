"""Versioned, offline, project-owned curated retrieval."""

from local_retrieval.schema import (
    RETRIEVAL_INDEX_ID,
    RETRIEVAL_POLICY_ID,
    RetrievalContractError,
    RetrievalQuery,
    RetrievalResponse,
    validate_retrieval_index,
    validate_retrieval_response,
)
from local_retrieval.service import (
    LocalRetrievalService,
    citations_from_response,
    local_retrieval_health,
    render_retrieval_summary,
    response_metadata,
    retrieval_tool_event,
    search_curated_local,
    search_for_request,
)

__all__ = [
    "LocalRetrievalService",
    "RETRIEVAL_INDEX_ID",
    "RETRIEVAL_POLICY_ID",
    "RetrievalContractError",
    "RetrievalQuery",
    "RetrievalResponse",
    "citations_from_response",
    "local_retrieval_health",
    "render_retrieval_summary",
    "response_metadata",
    "retrieval_tool_event",
    "search_curated_local",
    "search_for_request",
    "validate_retrieval_index",
    "validate_retrieval_response",
]
