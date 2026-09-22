# Bounded project and conversation memory

VFAI-025 memory is disabled by default and is available only behind the
authenticated Spring gateway. The gateway supplies its private service token
plus the authenticated JWT subject and active project/session identity in
headers. Raw user, project, and session identifiers are domain-separated and
SHA-256 hashed before SQLite persistence.

Users explicitly enable memory per user/project. Facts, decisions, and
summaries require an approved typed write. Enabling memory also approves short,
redacted recent-turn summaries for the active session. Entries have TTLs,
project revision bindings, strict character/count/byte limits, stable eviction,
optimistic correction versions, inspect/delete/clear/disable controls, and are
never admitted to training.

Secrets are replaced before persistence. Prompt-injection instructions, hidden
reasoning, JSON/full-project snapshots, and secret-only content are rejected.
Stale-revision entries remain inspectable but cannot enter model context.
Memory is user context only: it is not deterministic or retrieved evidence and
cannot support VFAI-024 high-risk factual claims.

The executable contract is pinned by `policy.v1.json` and the generated
`memory-state.schema.json`. The content-free evaluator exercises authenticated
scope isolation, redaction, unsafe-content rejection, revision staleness, TTL,
stable eviction, user controls, and cross-layer integration without network
access. Run from `Voltforge_AI` with the pinned Python 3.12 environment:

```powershell
.toolchains/gen1/Scripts/python.exe tools/evaluate_bounded_memory.py evaluate
.toolchains/gen1/Scripts/python.exe tools/evaluate_bounded_memory.py verify
.toolchains/gen1/Scripts/python.exe -m pytest tests/test_memory_store.py -q
```
