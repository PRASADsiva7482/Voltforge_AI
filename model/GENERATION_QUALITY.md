# Neural generation quality boundary

VFAI-019 adds a fail-closed boundary between local neural decoding and every
API-visible response. A decoder result is untrusted text until it passes
`vfai019-generation-quality-policy-v1`. The current experimental Gen1 artifact
is still not connected to chat and cannot become active through this work.

## Accepted neural shape

The gate accepts one bounded UTF-8 JSON envelope with exact fields. Visible
content is split into grounded claims, safety warnings, or honest uncertainty.
Grounded claims must remain in the electronics domain, cite existing typed tool
evidence, have lexical support in that evidence, and exactly match the envelope
citation list. Confidence is capped by evidence authority. Repetition,
truncation, instruction overrides, unsafe electrical directions, and malformed
or unknown evidence fail closed.

Structured circuit and firmware actions are constructed only after every text
gate succeeds. They must use the canonical task schema, bind to the exact
project revision, remain proposal-only, require user confirmation, and cite
deterministic evidence. Their action kind and payload must exactly match an
`approvedActions` proposal emitted by that deterministic tool; a related
finding alone is insufficient. Code fixes additionally require
compiler-confirmed evidence. Unexpected execution controls are rejected by the
typed schema.

This is a conservative support proxy, not full semantic entailment. VFAI-024
will strengthen claim-to-source entailment and conflict handling after local
and internet retrieval exist.

## Deterministic fallback

`resolve_generation` is the single neural/fallback decision boundary. A missing
model, unavailable context compiler, generation-disabled runtime, or rejected
candidate invokes the existing deterministic electronics engine. Rejected
neural text and its attempted actions are never returned. An accepted
candidate is the only path that can carry neural text and a neural-built typed
response record.

The SSE `start` and `complete` events expose fallback use, reason, source,
whether neural inference was attempted, artifact identity, and quality-gate
status. Health endpoints expose content-free counters by reason/code. Raw
prompts and raw model candidates are not retained in those metrics.

The API currently reports `deterministic-fallback` with
`NO_APPROVED_MODEL_ARTIFACT`. If an experimental artifact is inspected offline,
that does not change service readiness or create a neural API response.

VFAI-020 now supplies the context compiler boundary, but all immutable Gen1
artifacts declare a context below its minimum and remain inactive. Even if a
future test fixture reports a ready model, chat reports
`NEURAL_GENERATION_NOT_ENABLED` until an approved local generation invocation
is deliberately connected behind both the context compiler and this quality
gate. See [`../context_compiler/README.md`](../context_compiler/README.md).

## Reproduce the gate evidence

Use the pinned Python 3.12 environment:

```powershell
.toolchains/gen1/Scripts/python.exe tools/evaluate_generation_quality_gates.py evaluate
.toolchains/gen1/Scripts/python.exe tools/evaluate_generation_quality_gates.py verify
.toolchains/gen1/Scripts/python.exe -m pytest tests/test_generation_quality.py tests/test_chat_stream.py -q
```

The committed evaluation receipt stores fixture IDs, decision codes, hashes,
and content-free metrics. It stores neither candidate text nor deterministic
reply text.

## VFAI-021 authoritative engineering findings

The quality gate treats `tool:engineering-authority-index` as a non-optional
safety boundary. If its typed payload reports blocking findings, an accepted
neural envelope must render them as a `safety-warning` citing that exact event.
Silence fails with `QG_AUTHORITATIVE_FINDING_OMITTED`; contradictory reassurance
or an attempt to downgrade the finding fails with `QG_AUTHORITATIVE_OVERRIDE`.
The model may explain the deterministic result, but cannot replace its severity,
blocking state, evidence, affected IDs, or reviewable fix. See
[`../engineering_tools/README.md`](../engineering_tools/README.md).
