# Claim-level grounding, conflicts, and uncertainty

VFAI-024 creates an exact citation catalog only from project context and tool
events selected by the context compiler. Unselected local or internet results
cannot appear as model evidence. Every citation carries the exact selected
payload hash, source revision, evidence class, authority, content hash, and
task evidence reference.

The response gate classifies four high-risk factual claim types: datasheet
ratings, pin capabilities, library APIs, and current web facts. Such a claim is
kept only when eligible evidence contains its critical values or identifiers
and sufficient subject/property support. The gate appends the resolving
citation ID to visible text and records a content-free claim binding. An
unsupported statement is removed and replaced with explicit cannot-verify
wording. Conflicting structured facts or source assertions replace the claim
with visible conflict wording and cap confidence.

Evidence is exposed as `project`, `deterministic`, `local`, or `internet`.
Deterministic engineering remains authoritative; retrieval never authorizes a
structured action, and internet evidence remains untrusted. Citation and
uncertainty SSE events are emitted before response deltas. No prompt, project
payload, model output, or claim text is retained in grounding metadata.

The checked `grounding-report.schema.json` must exactly match the executable
Pydantic contract. Reproduce the content-free release receipt from
`Voltforge_AI` with:

```powershell
.toolchains/gen1/Scripts/python.exe tools/evaluate_claim_grounding.py verify
.toolchains/gen1/Scripts/python.exe -m pytest tests/test_grounding.py -q
```
