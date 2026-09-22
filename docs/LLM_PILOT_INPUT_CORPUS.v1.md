# Task 023 admitted pilot/input corpus v1

Follow-through: task 024 now supplies the [owned 0.1.0 tokenizer and Gen2 token recount](LLM_TRAINED_GEN2_TOKENIZER.v1.md). This task-023 document retains its original proxy counts and acceptance scope.

This release closes the source, quality, split, coverage and provisional-budget
requirements recorded for task 023. It supplies verified inputs for later owned
tokenizer/scaling work. No tokenizer fit, model training, deployment or activation
is performed by this task. Production corpus and mixture acceptance remain in
tasks 031/032 under the already recorded acceptance staging.

The immutable [manifest](../corpus/pilot-input/v1/04830f1057093c4a71930e9f49940c6507c59496425926679e5cb789897c0f94/manifest.json),
[scorecard](../corpus/pilot-input/v1/04830f1057093c4a71930e9f49940c6507c59496425926679e5cb789897c0f94/scorecard.json), [mixture](../corpus/pilot-input/v1/04830f1057093c4a71930e9f49940c6507c59496425926679e5cb789897c0f94/mixture.json)
and [resource receipt](../corpus/pilot-input/resources-v1/9f8da044245b900490bd789870691c9ab2da8a45b899dff38f6007230f11ca42/measurement.json) bind the exact
source snapshots, code, policies, exclusions, notices and per-file checksums.
Repeated computation reproduces every shard and manifest using the retained
resource receipt. A new measurement gets its own immutable receipt and release
identity; elapsed time itself is not expected to be deterministic.

## Completed implementation plan

1. Review and record exact surviving FreeRTOS and owned-domain input permissions,
   alongside the five previously reviewed public source repositories.
2. Add independent, predeclared validation families for math and electronics.
3. Reapply normalization, secret checks, full proxy token counting, duplicate and
   protected-content checks, family reservations and source-use restrictions.
4. Enforce the finite input budget, source/domain/language accounting, explicit
   40/30/20/10 scheduling experiment and three-pass repetition limit.
5. Measure the full pipeline in a fresh process at the actual input scale,
   publish immutable corpus bytes, and verify consumer handoffs and rejection
   controls before recording task completion.

## Corpus and token accounting

There are **66 records: 42 train, 11 validation, five test
and eight quarantine**. All eight previous corpus exclusions and 50 older
owned-domain exclusions remain excluded. No source was edited to match or evade
protected prompts. The unchanged matcher checks 1,005
protected descriptors; included protected matches, cross-split duplicate edges
and cross-split lineage keys are all zero.

| Domain | Train unique proxy tokens | Train families | Validation unique proxy tokens | Validation families |
| --- | ---: | ---: | ---: | ---: |
| code | 807,902 | 2 | 3,605 | 1 |
| electronics | 2,031 | 7 | 1,747 | 2 |
| language | 108,047 | 2 | 27,492 | 1 |
| math | 1,295 | 5 | 1,618 | 2 |

Total training input is **919,275 unique proxy tokens**;
validation has **34,462**. Counts use the owned Gen1
v1.1.0 tokenizer on every document, not a pretrained tokenizer or a guessed
character ratio. Gen2 token counts remain zero until task 024 recounts its own
selected tokenizer. The scorecard also reports per-source, domain, language and
split counts; unique means token positions in admitted deduplicated documents,
not the number of vocabulary types. Actual training exposures are zero.

The provisional input budget was declared before supplement rendering: 900,000
to 1,200,000 unique train proxy tokens, at least 31,000 validation tokens, at
least two train families and one validation family per domain, and at least 200
validation tokens in each domain. This is sufficient to exercise real corpus
input and tokenizer plumbing at this finite scale. It is not a statistical
claim of adequate base-model language learning.

The bounded scheduling experiment exposes exactly **20,000 proxy tokens**:
8,000 language, 6,000 code, 4,000 electronics and 2,000 math. No document exceeds
three passes. Small math/electronics pools constrain larger balanced mixtures;
the remaining language/code bytes are available for tokenizer input. Prefix
sampling and these weights are experimental choices requiring later measured
model results. Infeasible budgets return no schedule and cannot be admitted.

## Source permissions and validation independence

[Exact input decisions](../data_governance/pilot_corpus/source-decisions.v1.json)
approve 58 specific surviving source records. They bind 14 FreeRTOS files with
complete retained MIT grants, 17 original owned-domain survivors, 23 surviving
files from the five expanded repositories and four new original references.
The excluded Python files and MkDocs duplicate receive no new input permission.
Existing candidate and source-evidence artifacts are preserved unchanged.

The original validation references cover polynomial integration, finite
expectation, capacitor charge redistribution and a two-node conductance system.
Their family roles are declared before rendering. Separate exact-arithmetic
identities check each calculation: Simpson weights, equiprobable enumeration,
charge/energy identities, and branch-current/power balances. Corrupted answers
are rejected. These are algorithmically checked reference examples, not human
review or measured component ratings. Their text is original; no evaluation
prompts, private projects, external teacher completions or downloaded weights
are generation inputs.

The permission review records Codex as the reviewer under the user's explicit
task authorization. It asserts no separate owner signature. Source grants,
corpus input admission, training-run approval and model release are distinct.

## Resource evidence and limits

The fresh-process pipeline checked **1,697,550 normalized
bytes**, with **1,865,375 retained acquisition bytes**,
in **38.195 seconds**, using a process peak of
**130,158,592 resident bytes**. Limits are 96 documents,
512 KiB per normalized document, 2 MiB normalized input, 8 MiB retained
acquisition, 300 seconds and 2 GiB peak process memory. Both acquisition sources
already enforce bounded selected-file download policies. This measurement
rechecks retained hashes/licenses/tree evidence offline; it does not remeasure
network download latency or imply server/GPU training performance.

Technical English and C dominate this input corpus. The math/electronics pools
are small. No broad conversational, general-knowledge, semantic-contamination,
server-capacity or trained-model adequacy claim is made. Owner backup/restore
acceptance remains required for later tokenizer/model release.

## Reproduce and consume

From `Voltforge_AI`, verify without network access:

```powershell
rtk proxy .toolchains\gen1\Scripts\python.exe -B tools/release_pilot_corpus.py verify --release corpus/pilot-input/v1/04830f1057093c4a71930e9f49940c6507c59496425926679e5cb789897c0f94 --recompute
rtk proxy .toolchains\gen1\Scripts\python.exe -B tools/release_pilot_corpus.py require-use --release corpus/pilot-input/v1/04830f1057093c4a71930e9f49940c6507c59496425926679e5cb789897c0f94 --usage tokenizer-fitting-input --split train
rtk proxy .toolchains\gen1\Scripts\python.exe -B tools/release_pilot_corpus.py source-impact --release corpus/pilot-input/v1/04830f1057093c4a71930e9f49940c6507c59496425926679e5cb789897c0f94 --source-id vf-owned-pilot-reference-v1
```

Rebuild the same bytes using the bound resource measurement:

```powershell
rtk proxy .toolchains\gen1\Scripts\python.exe -B tools/release_pilot_corpus.py build --resource corpus/pilot-input/resources-v1/9f8da044245b900490bd789870691c9ab2da8a45b899dff38f6007230f11ca42/measurement.json
```

The Python consumer is `data_governance.pilot_corpus.release.read_inputs`.
It recomputes admission before returning data and binds the bytes again during
handoff. `tokenizer-fitting-input` exposes only train records. Validation/test
require their respective input uses and cannot enter fitting. The bounded
pilot input use returns only the scheduled 20,000 Gen1 proxy token positions,
with document IDs, pass numbers and explicit `trainingRunApproved=false`.
It does not supply a training command. Production-training and runtime uses
are denied. The frozen task-024 fixture adapter still denies corpus fitting;
its eventual integration is task 024 work and was not started here.

For source removal, use the metadata-only impact command, append the affected
source/corpus identity and reason to
`data_governance/pilot-input-revocations.v1.json`, and publish a newly reviewed
replacement release. `require_use` fails closed on relevant revocations or a
missing/malformed ledger. Never erase old audit bytes or mutate an immutable
release; future derived artifacts must retain this manifest as lineage.

Implementation and acceptance controls are in
[release.py](../data_governance/pilot_corpus/release.py),
[policy](../data_governance/pilot_corpus/policy.v1.json) and
[tests](../tests/test_pilot_corpus.py). Historical candidate, precision-review,
source-expansion and Gen2 fixture checks remain part of the regression run.
