# VoltForge task record v1

`task-record.schema.json` is the single contract used by generated training
examples, tokenizer/training ingestion, held-out evaluation normalization, and
runtime prompt compilation. `tools/check_task_schema.py` proves the checked-in
JSON Schema exactly matches the executable Pydantic contract.

The envelope separates input from output and gives every semantic section an
explicit type:

- `system`, `user`, and `project-context` are required inputs.
- `tool-evidence` is input evidence with tool version, authority, status,
  project revision, summary, and bounded JSON payload.
- `assistant-text`, `structured-action`, `citation`, `refusal`, and
  `uncertainty` are output-only records.

All objects reject unknown fields. Hidden reasoning/chain-of-thought has no
schema field and therefore cannot enter a v1 shard. Legacy `[THINK]` blocks are
removed by the migration adapter rather than becoming training targets.

Every structured action is permanently `proposal-only`, requires explicit user
confirmation, and carries the same `sourceProjectRevision` as its project
context. `autoApply` and similar undeclared fields fail JSON Schema validation.
Tool evidence and actions using a different revision fail semantic validation.

VFAI-021 engineering events carry their actual policy-pinned tool version
through the runtime adapter instead of receiving a generic version label. The
full strict `EngineeringAuthorityReport` remains the API contract; its compact
tool events enter `task-record-v1` with the same project revision and evidence
identities. Critical findings never become executable actions, and any approved
fix remains proposal-only.

VFAI-022 local-index results enter the same tool-evidence contract with
`authority: retrieved`, the actual retrieval contract version, exact source
revision, stable citation IDs, and content hashes. They remain distinct from
client-reported retrieval. A stale or unavailable index contributes no factual
result payload and cannot be cited as if retrieval succeeded.

VFAI-024 derives response citations only from the exact project context and
tool-evidence payload selected for the model. Assistant-text evidence references
contain only evidence actually used by supported claims; the wider citation
catalog remains reference-only. Datasheet ratings, pin capabilities, library
APIs, and time-sensitive web claims are either bound to eligible citation IDs or
replaced with typed uncertainty/conflict output.

VFAI-025 bounded memory is authenticated user context, never evidence or
training data. The chat boundary discards client-supplied memory and may add
only server-managed, enabled, revision-current entries marked
`authority: user-memory`, `modelEvidenceAllowed: false`, and
`trainingUseAllowed: false`. These entries can help continuity but cannot
support citations, resolve conflicts, authorize actions, or override the
engineering authority index.

VFAI-026 wraps synchronous chat and every streamed event in the versioned
FastAPI/SSE contract without changing task-record-v1 semantics. The exact
source project revision and final task record ID are exposed in the public
contract envelope, while prompt bytes, project content, raw model output, and
hidden reasoning remain absent from contract telemetry. Cancellation or a
deadline prevents subsequent task stages and memory persistence.

The prompt compiler does not concatenate role text with informal delimiters.
Each section has a typed header, UTF-8 byte length, and SHA-256 checksum. A
marker-like string inside user/project content remains JSON payload and cannot
escape into a tool or assistant section. `parse_compiled_sections` verifies the
same framing for tests and diagnostics.

VFAI-020 builds the runtime project prompt through the versioned
[`context_compiler`](../context_compiler/README.md). It canonicalizes every
supported project source, applies an exact owned-tokenizer budget, preserves
mandatory task/safety/selection/source-boundary sections, and exposes only
content-free hashes and counts to production telemetry. The schema compiler
still owns framing; the context compiler owns normalization, priority, and
bounded section selection.

Shard writers validate the complete batch before opening the destination and
atomically replace the JSONL file only after every record passes. Readers repeat
JSON Schema and semantic validation before tokenizer or model ingestion.

Commands from `Voltforge_AI`:

```powershell
python tools/check_task_schema.py
pytest -q tests/test_task_schema.py
```
