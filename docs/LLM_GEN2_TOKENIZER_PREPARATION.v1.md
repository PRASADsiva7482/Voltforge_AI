# Gen2 owned tokenizer preparation

Task 024's [corpus-trained tokenizer release](LLM_TRAINED_GEN2_TOKENIZER.v1.md)
now supplies the selected 0.1.0 artifact and real model/context integration.
This document preserves the earlier implementation-fixture checkpoint; its
fixture artifacts remain unchanged and cannot stand in for the trained release.

## Implemented contract

`model/gen2_tokenizer/` composes the existing project-owned BPE mathematics without
editing `model/tokenizer.py`, importing another tokenizer, or loading pretrained
weights. Namespace `vfdlm-g2-byte-bpe` reserves production version `0.1.0`.
Published controls have version **0.1.0-fixture.1** and cannot serve or train a model.

All 256 bytes are available at IDs 16–271, followed by deterministic byte merges.
Text uses UTF-8 with surrogatepass and no normalization. Arbitrary bytes use
`encode_bytes`/`decode_bytes`; invalid UTF-8 fails strict text decoding. Incremental
text decoding buffers incomplete multibyte characters rather than emitting
replacement characters. Tests independently recount BPE pairs and implement a
separate exhaustive encoder to check the reused linked-list mathematics.

| ID | Boundary |
| --- | --- |
| 0–3 | PAD, UNK, BOS, EOS |
| 4–7 | system, user, assistant, tool |
| 8–9 | end_message, tool_call |
| 10–13 | project_context, tool_evidence, citation, refusal |
| 14–15 | uncertainty, reserved_15 |

Each spelling is `<|name|>`, with lowercase names from this table. These IDs are
defined by contract `0.1.0`; template `vfdlm-g2-chat-v1` is version `1.0.0`.
All marker-looking text in content is encoded as ordinary byte tokens. Only the
trusted caller can insert structural IDs. Tool-call/context/citation markers are
reserved here; task 046/049 must define their higher-level evidence semantics.

`context_compiler/gen2.py` emits token IDs in this order:

```
BOS (ROLE CONTENT_BYTES END_MESSAGE)* ASSISTANT_PREFIX
```

Completed conversations instead end with EOS. Framing costs exactly `2 + 2N`
tokens for N messages. Literal marker bytes retain their actual BPE cost, which
can exceed one token; no escaping estimate or character-count approximation is
used. The 4096-token contract includes an explicit output reservation. Oversized
messages fail rather than silently dropping content. Never decode this ID array
to text and reparse it for inference: that would erase the structural distinction.

The adapter checks exact vocabulary/merges, special IDs, contract and template
digests against a shared model binding. A matching fixture binding tests the
interface only. Runtime use rejects with `GEN2_TOKENIZER_RELEASE_NOT_ADMITTED`.
Task 025 must wire the descriptor into the actual Gen2 model configuration;
tasks 042/046 must wire verified artifacts into inference and authoritative
context selection. Existing application routing is not switched by these tests.

## Fixture artifacts and measurements

Eight public, project-authored implementation inputs cover text, C-like code,
SI symbols, Unicode, marker strings and all bytes. Six separate measurement
fixtures include a 320-line code sample. Neither set reads protected evaluation
prompts or corpus validation/acceptance shards. These controls are not a training
corpus and confer no language, model quality or corpus adequacy credit.

Candidate ceilings are **16,384 and 32,768** entries. Small fixtures stop when
eligible pairs run out; reports publish actual sizes and `targetReached` for
both. The existing immutable 3,072-entry tokenizer is a diagnostic baseline.
The recorded fixture comparison produced **911 actual entries under each ceiling**;
both targets remain unreached.
Its fitting data differs, so fixture compression cannot select the production
vocabulary. Reports measure reversible bytes, tokens per whitespace unit and
code point, bytes per token, byte-token reduction, long-code overhead, literal
marker cost and short encode/decode throughput. Whitespace fertility is a proxy;
host measurements establish no server capacity.

Artifacts live under `model/tokenizers/fixtures/gen2/v1/<content-hash>/` and bind
authored input hashes, implementation hashes, contract/template, byte vocabulary,
merges and actual/target counts. Publication is locked, manifest-last and
write-once. Readers reconstruct the expected metadata, prohibit alternate file
paths and reject self-rehashed contract changes. A fixture manifest is not an
approval signature. Historical Gen1 artifacts and corpus manifests remain intact.

Run from `Voltforge_AI`:

```powershell
.toolchains\gen1\Scripts\python.exe -B tools/build_gen2_tokenizer.py fixtures
.toolchains\gen1\Scripts\python.exe -B tools/build_gen2_tokenizer.py fixtures --report evaluation/reports/llm-task-024-comparison.json
.toolchains\gen1\Scripts\python.exe -B tools/build_gen2_tokenizer.py verify-fixture --release model/tokenizers/fixtures/gen2/v1/<content-hash> --recompute
.toolchains\gen1\Scripts\python.exe -B -m pytest -q tests/test_gen2_tokenizer.py
```

Existing report paths are write-once; use a new report name for another timing
measurement. Fixture artifacts can be regenerated and verified byte-for-byte.

## Completed follow-through

The admitted-corpus adapter, real candidate fits and comparison, immutable
release, corpus recount and real tensor/context binding have been implemented
in the separate release linked above. The frozen fixture-only `check-corpus`
command continues to reject corpus fitting; use `train_gen2_tokenizer.py` for
the verified corpus-trained artifact. No legacy gate or artifact was rewritten.
