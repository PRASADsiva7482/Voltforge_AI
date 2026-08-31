# Project context compiler v1

VFAI-020 provides the deterministic boundary between a VoltForge project and
future local Gen1 inference. It accepts the bounded chat request plus
deterministic tool evidence, normalizes every supported source, and emits one
versioned `task-record-v1` prompt. It does not call a model, retrieval service,
or network endpoint.

## Inputs and trust boundary

The compiler recognizes the board, components, pins, wires, netlist, active
selection, firmware files, diagnostics, simulation state, recent memory,
retrieved evidence, and deterministic tool output. Project and user-provided
content is always framed as untrusted data with source identity, source
revision, and payload SHA-256. It cannot become a system instruction or escape
the typed prompt section that contains it.

Equivalent maps and set-like project collections are canonicalized before the
project revision and prompt are calculated. Therefore semantically equivalent
requests produce the same prompt bytes and digest even when their input object
keys or project collections arrive in a different order.

VFAI-021 supplies one mandatory `engineering-authority-index` tool event. It
preserves the highest-priority blocking findings, affected IDs, fixes, and
evidence hashes when the token budget omits lower-priority per-tool detail. The
index is revision-bound and cannot be displaced by user text, memory, retrieval,
or optional tool events.

VFAI-022 adds optional `curated-local-retrieval` evidence ahead of ordinary
per-tool detail. Its compact event carries at most two exact local citations so
it can fit realistic context budgets while the full strict response remains at
the retrieval API. A retrieval unavailable state is visible but not treated as
a safety blocker, and it cannot displace the mandatory engineering authority
index. Client-provided retrieval remains non-authoritative.

## Exact token budget

[`policy.v1.json`](policy.v1.json) binds the compiler to the project-owned
`vfdlm-byte-bpe` tokenizer contract. The complete framed prompt is counted with
the same BOS convention used by Gen1; character estimates are never used.
The policy reserves output tokens first and rejects a context window smaller
than 768 tokens. Prompt selection is strict and priority ordered:

1. task, system contract, board revision, and safety evidence;
2. active selection, relevant nets, and the first complete active-firmware
   chunk;
3. diagnostics and deterministic tool evidence;
4. remaining firmware, simulation, components, pins, wires, and netlist;
5. retrieved evidence and recent memory.

Mandatory sections fail closed if they cannot fit. Optional sections are
included only as complete typed sections. Firmware is split on declared
source boundaries with line ranges and whole-file hashes; no partial section or
partial source chunk is emitted. Once a higher-priority optional section does
not fit, lower-priority content is omitted as well.

The current immutable experimental Gen1 artifacts declare only 128 context
tokens. They are incompatible with this policy and remain inactive. A larger
context must be selected through a governed train-and-evaluate revision rather
than by relabeling existing weights.

## Privacy and observability

Production metadata contains policy and tokenizer identities, token counts,
section counts, omission reasons, and prompt/source digests. It does not
contain project text, filenames, selected IDs, memory, evidence, or model
output. Prompt snapshots are generated only from synthetic evaluation fixtures
and written only by an explicit evaluation command; runtime code has no prompt
snapshot or raw-content logging path.

## Verification

Run from `Voltforge_AI` with the pinned Python 3.12 environment:

```powershell
.toolchains/gen1/Scripts/python.exe tools/evaluate_context_compiler.py evaluate
.toolchains/gen1/Scripts/python.exe tools/evaluate_context_compiler.py verify
.toolchains/gen1/Scripts/python.exe -m pytest tests/test_context_compiler.py tests/test_chat_stream.py -q
```

The committed synthetic snapshot proves exact framing and reproducibility. The
content-free report verifies ordering stability, mandatory preservation,
source-boundary truncation, privacy, registry inactivity, and fail-closed
behavior for all current 128-token artifacts.

VFAI-023 sanitized internet evidence is optional and ranked below curated
local retrieval. It remains `retrieved` and untrusted, and an offline or
degraded internet event is never mandatory, so it cannot displace the
deterministic engineering authority index.

VFAI-025 admits only server-managed, explicitly enabled memory carrying
`authority: user-memory`, `modelEvidenceAllowed: false`, and
`trainingUseAllowed: false`. Client-provided memory remains untrusted and is
discarded at the chat boundary. Exact project-revision mismatches are omitted
from model context while remaining visible through the authenticated memory
inspection API. Managed memory remains below deterministic and retrieved
evidence and cannot establish an engineering fact or authorize an action.
