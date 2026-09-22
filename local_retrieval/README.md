# Curated local retrieval v1.1

VFAI-022 adds a project-owned lexical retrieval layer over the approved
VoltForge electronics corpus. It runs fully offline, uses no pretrained model,
embedding service, or external index, and cannot read the quarantined legacy
datasets. Embeddings are explicitly disabled until a later governed evaluation
shows that they are necessary and safe.

## Indexed source and trust boundary

The only v1.1 source is `vf-src-curated-electronics-corpus-v1`, approved by data
governance for `runtime-retrieval`. The source contains 61 checksum-verified
normalized records across boards, pin maps, components, wiring recipes,
firmware APIs, compiler diagnostics, simulation behavior, and safety rules.
External documents are represented only by bounded evidence references; copied
source prose is not stored in the index.

The builder flattens each typed claim into atomic exact facts carrying the
claim ID, JSON pointer, status, conditions, unit, and evidence IDs. It creates
104 bounded chunks and a deterministic inverted lexical index. BM25 scoring is
performed with fixed Decimal parameters. Board, component, and record-type
filters narrow or boost exact variants without silently substituting an
ambiguous family.

Every returned result contains:

- the exact record, source, corpus, and effective record revisions;
- bounded facts and evidence copied from the validated source record;
- a stable citation ID and locator;
- a content SHA-256 covering the complete returned chunk;
- content-free ranking metadata and explicit support status.

The service verifies the policy checksum, data-governance approval, source
revision, source-entry digest, corpus catalog/content hashes, builder hash,
index self-checksum, checked schemas, and a complete reproducible rebuild. Any
mismatch returns an explicit unavailable/degraded state with zero results.
Stale evidence is never served.

## Runtime integration

`POST /voltForge-ai/api/v1/model/retrieval/search` returns the complete strict
retrieval response. Chat automatically derives bounded board/component filters
from the current project and emits `curated-local-retrieval` as `retrieved`
tool evidence. The model-facing event contains at most two citations and small
exact facts so it fits the VFAI-020 context budget. The full API response remains
available for review.

The VFAI-021 `engineering-authority-index` remains mandatory and is never
displaced by retrieval. Curated retrieval receives priority over optional
per-tool detail but remains optional if the exact context budget cannot fit it.
Client-supplied `retrievedEvidence` remains separately marked client-reported
and does not gain local-source authority.

Health and SSE start metadata publish the index ID/version/checksum, source
revision, record/chunk counts, algorithm, embedding state, and stale-result
policy without storing raw queries or project context.

## Quality evidence

The held-out v1 suite has 27 evaluation-only queries spanning all eight record
types and exact board/component-filtered wiring. The committed evaluator checks
recall@5, mean reciprocal rank, exact citation entailment, unsupported citation
rate, deterministic ordering, bounded output, network denial, privacy, chat
selection, API shape, and stale-index failure. Query text is stored only in the
project-owned held-out fixture; runtime responses and receipts contain hashes.

Rebuild and verify with the pinned Python 3.12 environment:

```powershell
.toolchains/gen1/Scripts/python.exe tools/build_local_retrieval_index.py write
.toolchains/gen1/Scripts/python.exe tools/build_local_retrieval_index.py verify
.toolchains/gen1/Scripts/python.exe tools/evaluate_local_retrieval.py evaluate
.toolchains/gen1/Scripts/python.exe tools/evaluate_local_retrieval.py verify
.toolchains/gen1/Scripts/python.exe -m pytest tests/test_local_retrieval.py -q
```

Run `write` only after an intentional approved policy, corpus, schema, or
builder revision. Ordinary service startup and search are read-only.
