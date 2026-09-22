# Pretraining corpus preparation — LLM-TASK-023

This document records a historical candidate checkpoint. The current task-023
[admitted pilot/input release](LLM_PILOT_INPUT_CORPUS.v1.md) closes its input
admission gaps while preserving the candidate bytes and original decisions.

**Task 023 remains IN_PROGRESS.** This implementation creates reproducible corpus
candidates, exact source evidence, indexed leakage checks, token accounting and
bounded mixture plans. It does not release an adequate pretraining corpus, fit a
tokenizer, train a model or change chat serving.

The [candidate manifest](../corpus/pretraining-candidates/v1/8f55313b779ffa8c8beebc19cbb3761a354a0e219fdf794865882d4728dcf852/manifest.json)
and [scorecard](../corpus/pretraining-candidates/v1/8f55313b779ffa8c8beebc19cbb3761a354a0e219fdf794865882d4728dcf852/scorecard.json)
are immutable and checksum-bound to the exact source, tokenizer and processing
code. `trainingAllowed=false` applies throughout this candidate release.

## Current measured corpus

This section preserves the original task-023 candidate. The later
[source expansion](LLM_SOURCE_EXPANSION.v1.md) adds 24 exact-source-reviewed files
and creates a new immutable 62-record candidate with 42 train, seven validation,
five test and eight quarantine records. All old exclusions remain intact; the
expanded candidate also retains `trainingAllowed=false`.

| Input | Candidate records | Existing-tokenizer proxy tokens | Surviving records |
| --- | ---: | ---: | ---: |
| Selected FreeRTOS implementation/header files | 14 | 718,908 | 14 train |
| Selected Python developer documentation | 7 | 86,918 | 0; all 7 quarantined |
| Previously surviving owned-domain examples | 17 | 4,818 | 12 train, 5 test |
| Total | 38 | 810,644 | 26 train, 0 validation, 5 test |

The train split contains **722,234 proxy tokens**: 718,908 code, 2,031 electronics,
1,295 math and zero language tokens. One repository is one independent source
family; fourteen FreeRTOS files are not fourteen independent code families.
The five owned test records contribute 1,492 additional proxy tokens. All 50
task-022 exclusions remain excluded; they were not relabelled or imported again.

Two Python documents trigger the existing protected lexical-containment policy.
Their repository ancestry quarantines all seven selected documentation files.
The predicate matches shared word sets, not necessarily a copied sentence;
there is no semantic-copying claim. The exclusions remain in place. The eighth
selected Python file fails the inherited hidden/broken-text gate before raw
source persistence; its reason and checksum are retained. Linked documents,
reST includes, scripts and submodules are never fetched or executed.

These counts use the existing owned byte-BPE tokenizer v1.1.0, with no BOS/EOS
addition or special-token interpretation. Code and source markup remain in the
counted text; metadata and license receipts are not added as separate training
documents. They are exact counts for this representation, **not Gen2 tokenizer
counts, diverse general-language coverage, or a model-quality result**.

## Source acquisition and admission

The [source plan](../data_governance/pretraining/sources.v1.json) pins:

- Python developer guide commit `9d481ef402e0d112ae6e1c11a87c5df038e0e5f0`:
  eight named reST files proposed, seven retained. Its [pinned CC0 license](https://github.com/python/devguide/blob/9d481ef402e0d112ae6e1c11a87c5df038e0e5f0/LICENSE)
  is retained as evidence.
- FreeRTOS kernel commit `dbf70559b27d39c1fdb68dfb9a32140b6a6777a0`
  (observed tag V11.1.0): fourteen named C/header files. The [pinned MIT license](https://github.com/FreeRTOS/FreeRTOS-Kernel/blob/dbf70559b27d39c1fdb68dfb9a32140b6a6777a0/LICENSE.md)
  and every complete in-file MIT grant/copyright notice are retained.

The [v2 acquisition manifest](../corpus/acquisition/v2/80396fd5000abf7f4f7d62bd0ce895a63f0436ee19271a64d8f60fc381264bc3/manifest.json)
records 1,318,853 bytes obtained for the bounded selection and license review,
21 retained source files, two license files and one denied source file. This
number includes rejected bytes inspected in memory. Each selected file retains
its exact upstream URL, commit, checksum and license evidence. The acquirer
rejects redirects, unpinned references, path traversal, oversized responses,
secret patterns and unsupported text. There is no crawler or code execution.

The initial v1 acquisition over-rejected the FreeRTOS copyright reservation
notice. V2 checks the **complete MIT grant**, SPDX identifier and copyright
notice before treating that reservation within the licensed header as compatible
with the grant; restrictions elsewhere still require review. Tests reject
missing/altered grants, conflicting identifiers and added noncommercial terms.
The [initial v1 manifest](../corpus/acquisition/v1/0af19c48d917a962cb7c6395c42e70ecc299bafe2fc412269c45d1df951a5acc/manifest.json),
its stricter exclusions and its original verifier remain intact.

**Acquisition is candidate review, not source-use admission.** The new source
records retain origin, exact revision/checksums, license observations, privacy
scope, retention and deletion procedures, with training and redistribution
approval still false. Existing task-019 approvals do not automatically transfer
to these bytes, including the newly authored task-022 source. No owner or legal
approval is fabricated by a boolean or an open-source repository label. The
full selected-file review packet is in the candidate `sources.json` and acquired
license/source files. No OpenStax, Gutenberg, Wikipedia or unreviewed corpus was
downloaded during this task.

## Indexed leakage controls

The follow-up [precision review](LLM_CORPUS_LEAKAGE_PRECISION.v1.md) now separates
identity/literal/code signals from lexical-only matches without changing the
matcher or any existing exclusion. Its 56 authored controls retain all 32
protected positives and expose 16 unrelated word-overlap matches. Real corpus
matches remain unadjudicated; these controls do not authorize readmission.

`data_governance/pretraining/indexed.py` indexes raw/normalized identities,
content/code hashes, lineage keys, lexical tokens and protected literal anchors.
It prunes impossible pairs, then calls the unchanged task-021 comparator with
the same thresholds. The anchor lookup covers literal containment even when a
protected string has fewer than six distinct words. It does not remove board
facts, relax matching thresholds or substitute approximate nearest neighbours.

Only explicitly named protocol/provenance containers are omitted by the typed
payload adapter. Questions, answers, context and code remain checked. A prose
document containing code examples retains both prose and code protections.
Cross-source matches connect entire families before splitting. Existing reserved
assignments are preserved; conflicting reservations cause quarantine.

The [index evaluation](../evaluation/reports/llm-task-023-index-evaluation.json)
records identical match results to exhaustive scanning for 93 authored control
documents against 85 guards (71 protected matches, eight duplicate edges), and
for all 38 real candidate documents against 1,005 protected guards. A separate
100-family fixture produces 74 train, 16 validation and 10 test records, with
zero cross-split/protected collisions. Fixtures never become training data.

The bounded fixture benchmark measured roughly 335 documents/second for 6,000
preconstructed independent documents, with about 28.8 MB of traced allocations
for index construction and scanning. The benchmark excludes input feature
construction and uses disjoint vocabulary; it is **not** worst-case or
billion-token corpus capacity evidence. Common-word controls deliberately exceed
the comparison cap and fail closed. Real input scanning verifies 14,856 pairs
versus an exhaustive upper bound of 38,893. The included real records have zero
audited cross-split/protected collisions; validation remains empty.

## Mixture and release gates

`mixture.py` starts with the requested 40% language, 30% code, 20% electronics,
10% math hypothesis. Integer quotas use deterministic largest remainders so they
sum exactly to the exposure budget. Each document may be visited at most three
times; repeated passes add exposures, never unique tokens. Validation records,
duplicate IDs/bytes, unknown domains and invalid token counts are rejected.
An infeasible plan has no schedule and zero actual training exposures.

All three measured current-data plans fail: a 722,234-exposure plan, the existing
2-billion pilot hypothesis and the 20-billion base hypothesis. The language pool
is empty and the math/electronics pools are too small for the mixture. A
2-billion-exposure plan still lacks 800,000,000 language, 597,843,276 code,
399,993,907 electronics and 199,996,115 math tokens after the three-pass cap.
These are planning deficits, not approved budgets. No held-out model experiment
has established mixture quality.

The full-training readiness function reports missing source-use admission,
independent language/code family breadth, validation coverage, verified scaling
and mixture-pilot receipts, and Gen2 token counts. It refuses caller-supplied
approval booleans as replacements for future task-owned receipt verifiers. Gen2
token recounting belongs to task 024; scaling and mixture measurements belong
to tasks 031/032. These later full-training gates do not constitute successful
task-023 corpus acceptance.

The initial candidate manifest records an acceptance-order loop: task 023 demanded
a measured scaling budget while task 031 depends on task 023. The backlog now
distinguishes the admitted pilot/input corpus in task 023 from measured
production-corpus adequacy in tasks 031/032. The original measured-budget
requirement is preserved verbatim in task 032's planning record and explicitly
retained in its acceptance criteria. Dependencies are unchanged, and task 023
remains open for its incomplete input-corpus requirements.

## Reproduce and continue

From `Voltforge_AI`, offline after the explicit acquisition:

```powershell
rtk proxy .\.toolchains\gen1\Scripts\python.exe -B tools/acquire_pretraining_sources.py verify --release corpus/acquisition/v2/80396fd5000abf7f4f7d62bd0ce895a63f0436ee19271a64d8f60fc381264bc3
rtk proxy .\.toolchains\gen1\Scripts\python.exe -B tools/build_pretraining_corpus.py build
rtk proxy .\.toolchains\gen1\Scripts\python.exe -B tools/build_pretraining_corpus.py verify --release corpus/pretraining-candidates/v1/8f55313b779ffa8c8beebc19cbb3761a354a0e219fdf794865882d4728dcf852 --recompute
rtk proxy .\.toolchains\gen1\Scripts\python.exe -B tools/build_pretraining_corpus.py source-impact --release corpus/pretraining-candidates/v1/8f55313b779ffa8c8beebc19cbb3761a354a0e219fdf794865882d4728dcf852 --source-id vf-freertos-kernel-dbf70559
rtk proxy .\.toolchains\gen1\Scripts\python.exe -B -m pytest -q -p no:cacheprovider tests/test_pretraining_corpus.py
```

The [verified source-impact report](../evaluation/reports/llm-task-023-source-impact.json)
traces the FreeRTOS selection to all 14 candidate records and their exact hashes
and token counts. It reports no dependent training artifact in this candidate
workflow and performs no deletion.

`source-impact` reports affected records without deleting anything. The public
`require-use` command permits candidate review only; tokenizer fitting,
pretraining, SFT and retrieval evaluation remain denied. A changed source,
permission, matcher, tokenizer or policy requires a new immutable version.

Continue **LLM-TASK-023**, with these concrete remaining steps:

1. Complete the pilot/input-corpus requirements recorded in task 023, retaining
   the separate measured-production-budget acceptance in tasks 031/032.
2. Five new repositories now have recorded exact source-input decisions. Complete
   separate admission for the surviving inherited FreeRTOS/owned-domain sources;
   do not transfer old candidate approvals or readmit the excluded Python source.
3. Preserve the expanded technical English/C train-validation family separation
   and add independent math/electronics validation families. Broader language and
   realistic corpus adequacy still need evidence.
4. The independent lexical-precision review is complete. Before adopting a new
   matcher for fresh corpus data, add versioned precision/recall acceptance and
   review evidence; retain all current exclusions and protected bytes.
5. Measure real acquisition/normalization/tokenization/index resources at the
   intended corpus scale, enforce repetition/coverage gates, and publish an
   admitted corpus only when its applicable acceptance evidence is complete.
