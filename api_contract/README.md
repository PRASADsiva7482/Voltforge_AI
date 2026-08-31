# FastAPI and SSE contract v1

VFAI-026 defines the stable boundary between the Spring gateway and the local
VoltForge AI service. Chat requests and synchronous responses default to
`schemaVersion: 1` and `contractVersion: 1.0.0`; an unknown version or field is
rejected with a content-free typed error. The checked request, response, and
SSE JSON schemas are generated directly from the executable Pydantic models.

Every SSE message has one canonical envelope containing a unique event ID,
contiguous sequence, request/session IDs, exact source project revision, local
model identity, generation mode, and artifact identity. The typed payload is
also flattened temporarily for compatibility with the existing Spring and UI
consumers. The supported event names are `start`, `tool`, `citation`,
`uncertainty`, `delta`, `proposal`, `complete`, and `error`; only `complete` or
`error` may terminate a connected stream. Hidden-reasoning events are not part
of the contract.

Requests, replies, deltas, individual events, total stream bytes, event counts,
tool events, citations, proposals, and deadlines are bounded. The start event
and `/contract` health route expose independent model, compiler, deterministic
tool, local retrieval, internet retrieval, grounding, and memory readiness.
An unavailable neural artifact is reported honestly while the approved
deterministic fallback remains usable.

Each active stream has a cooperative cancellation token. Client disconnect,
`DELETE /chat/requests/{request_id}`, or deadline expiry stops subsequent
retrieval/tool/generation/memory stages. A disconnected client receives no more
events; an attached explicitly cancelled or timed-out client receives a typed
terminal error. Cancelled turns are never written to bounded memory.

Run from `Voltforge_AI` with the pinned Python 3.12 environment:

```powershell
.toolchains/gen1/Scripts/python.exe tools/evaluate_api_contract.py evaluate
.toolchains/gen1/Scripts/python.exe tools/evaluate_api_contract.py verify
.toolchains/gen1/Scripts/python.exe -m pytest tests/test_api_contract.py tests/test_chat_stream.py -q
```

The evaluator uses no live network and writes only content-free metrics and
booleans to `evaluation/reports/fastapi-sse-contract-v1.json`.
