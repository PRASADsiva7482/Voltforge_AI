# Task 023 source admission and corpus expansion

This document records a historical candidate checkpoint. The current task-023
[admitted pilot/input release](LLM_PILOT_INPUT_CORPUS.v1.md) closes its input
admission gaps while preserving the candidate bytes and original decisions.

The source-expansion substep records exact source-use decisions for **24 files
from five new repositories** and builds a reproducible expanded candidate.
Task 023 remains **IN_PROGRESS**: source-input permission is separate from
corpus, tokenizer, model and training-run approval.

## Recorded source decisions

The [decision ledger](../data_governance/source_expansion/admission.v1.json)
binds the exact [acquisition manifest](../corpus/source-expansion/acquisition-v1/6d8a00721ddcb89a037c75282cabbd417300a19672a8706cfc474780baf40328/manifest.json).
Codex recorded the source-evidence review under the user's authorized owned-LLM
build. The ledger does not assert a separate owner signature or legal opinion.
It retains permission notices, privacy scope, permitted uses and removal/retention
obligations. These are new decisions; old source registries are unchanged.

| Repository and exact revision | Selected files | Reserved role | Permission evidence |
| --- | ---: | --- | --- |
| `quii/learn-go-with-tests` at `4675d96ad50b815d3e9fdc2122ab3dbc14636252` | 6 chapters | train | Root [MIT notice](https://github.com/quii/learn-go-with-tests/blob/4675d96ad50b815d3e9fdc2122ab3dbc14636252/LICENSE.md) and README license link |
| `rust-lang/book` at `1500248d8f230566e4ec9f27fcbb8fe9e2898ab1` | 6 chapters | train | Root [MIT grant](https://github.com/rust-lang/book/blob/1500248d8f230566e4ec9f27fcbb8fe9e2898ab1/LICENSE-MIT); both root license files retained |
| `squidfunk/mkdocs-material` at `9d65447eb4039c153edefbc378029257886737ff` | 6 documentation files | validation | Root MIT grant, README, explicit [documentation license](https://github.com/squidfunk/mkdocs-material/blob/9d65447eb4039c153edefbc378029257886737ff/docs/license.md) |
| `DaveGamble/cJSON` at `fb16e5cf358798aabb049655975cde8427101056` | 4 C/header files | train | Root [MIT grant](https://github.com/DaveGamble/cJSON/blob/fb16e5cf358798aabb049655975cde8427101056/LICENSE), README and retained source notices |
| `rxi/log.c` at `f9ea34994bd58ed342d2245cd4110bb5c6790153` | 2 C/header files | validation | Root [MIT grant](https://github.com/rxi/log.c/blob/f9ea34994bd58ed342d2245cd4110bb5c6790153/LICENSE) and README license statement |

Only these selected bytes are covered. The source-input verifier rejects
unselected files, changed bytes, unknown uses, altered scope and any attempt to
fit reserved validation families. Runtime retrieval and redistribution are not
approved by this ledger. Source conditions require retaining notices and tracing
later source-removal requests to dependent data/artifacts.

The acquirer downloads named Git blobs and the directory metadata needed to
inspect their ancestor license scope. It avoids recursively downloading asset
trees. Each file matches its pinned Git blob identity and retained SHA-256;
ancestor tree references and all selected/denied entries are checked. Redirects,
symlinks, unsafe paths, unexpected license scopes and secret patterns fail closed.
The complete MIT grant is compared, allowing its observed NON-INFRINGEMENT spelling
variant. Missing clauses or added restrictions are rejected.

The bounded selection acquired **546,522 bytes**, including permission and Git
metadata responses. All 24 selected files passed acquisition checks. Public source
markup and code are preserved; renderers, includes, scripts, submodules, images,
vendor dependencies and remote agent instructions are not executed or imported.
No Gutenberg or other literature corpus was acquired. Technical prose does not
establish broad conversational or general-knowledge coverage.

## Expanded candidate

The [expanded manifest](../corpus/pretraining-candidates/source-expansion-v1/733a4f704e7433f3d19f1f3c95e6109eceaf3f838be56acf6efff27322081f30/manifest.json)
binds the new input decisions, source packet and unchanged previous corpus.
Its 62 records combine the previous 38 with 24 new source files.

| Split | Records | Existing-tokenizer proxy tokens |
| --- | ---: | ---: |
| Train | 42 | 919,275 |
| Validation | 7 | 31,097 |
| Test | 5 | 1,492 |
| Quarantine | 8 | 93,479 |

Training now contains 108,047 technical English tokens from two independent
repository families, 807,902 C tokens, 2,031 electronics tokens and 1,295 math
tokens. Validation contains 27,492 technical English and 3,605 C tokens from
separate reserved repositories. One new validation file is excluded by the
unchanged duplicate policy; the other 23 new files survive current checks.

All seven previous corpus quarantines and all 50 earlier domain exclusions
remain excluded. Previously included train/test assignments are reserved before
the expanded scan. No validation family is fitted or moved into training; no
matcher threshold is changed. Repository ancestry, protected evidence and
cross-split duplicate checks remain active.

These counts use the immutable owned tokenizer v1.1.0, preserving exact source
markup with no BOS/EOS or special parsing. Old record counts are reused only
after verifying the original text/hash bindings; new counts are recomputed.
They are **not Gen2 counts**, an adequate model training budget or neural quality
evidence. The expanded candidate retains `trainingAllowed=false` and zero actual
training exposures. Existing Gen2 fixture artifacts remain unchanged.

## Commands and remaining work

From `Voltforge_AI`:

```powershell
.toolchains\gen1\Scripts\python.exe -B tools/expand_pretraining_sources.py verify-admission
.toolchains\gen1\Scripts\python.exe -B tools/expand_pretraining_sources.py build
.toolchains\gen1\Scripts\python.exe -B tools/expand_pretraining_sources.py verify --release corpus/pretraining-candidates/source-expansion-v1/733a4f704e7433f3d19f1f3c95e6109eceaf3f838be56acf6efff27322081f30 --recompute
.toolchains\gen1\Scripts\python.exe -B tools/expand_pretraining_sources.py source-impact --release corpus/pretraining-candidates/source-expansion-v1/733a4f704e7433f3d19f1f3c95e6109eceaf3f838be56acf6efff27322081f30 --source-id vf-g2-mkdocs-9d65447e
.toolchains\gen1\Scripts\python.exe -B -m pytest -q tests/test_source_expansion.py
```

Acquisition is separately explicit: `acquire` can access only the declared pinned
sources; subsequent verification/build is offline. Artifacts are write-once and
manifest-last. Source-impact output contains document identities, hashes, roles
and counts, performs no deletion and states the limits of its artifact lookup.
`require-use --usage tokenizer-fitting` remains denied with exit 2.

The inherited source decisions, independent math/electronics validation,
provisional budget, real-input measurements and admitted corpus release have
now been completed in the separate pilot/input release linked above. These
source-expansion artifacts remain unchanged and candidate-only.

No model training, model activation, application routing change or acceptance
of server/backup/restore requirements occurs in this source-input review.
