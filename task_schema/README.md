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

The prompt compiler does not concatenate role text with informal delimiters.
Each section has a typed header, UTF-8 byte length, and SHA-256 checksum. A
marker-like string inside user/project content remains JSON payload and cannot
escape into a tool or assistant section. `parse_compiled_sections` verifies the
same framing for tests and diagnostics.

Shard writers validate the complete batch before opening the destination and
atomically replace the JSONL file only after every record passes. Readers repeat
JSON Schema and semantic validation before tokenizer or model ingestion.

Commands from `Voltforge_AI`:

```powershell
python tools/check_task_schema.py
pytest -q tests/test_task_schema.py
```
