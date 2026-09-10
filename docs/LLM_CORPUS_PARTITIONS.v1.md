# Gen2 family partitions and leakage prevention v1

LLM-TASK-021 adds reproducible family partitions, exact/near-duplicate checks, protected evaluation guards and token/exposure accounting on top of the immutable task-020 ingestion output. It changes no prior source, evaluation, ingestion, tokenizer or model artifact.

**The current 227 source-approved staging records do not yield an independent training pool under these checks. All 227 are quarantined for this new partition candidate release.** Source permission and task-020 exact held-out exclusions remain valid historical facts; task 021 applies the additional content/family checks required before training. No original file is deleted or relabelled.

## Immutable evidence and current result

- [Partition manifest](../corpus/partitions/v1/be29e68619449efec0348e5747944b2a1861b78b82a93642df923de2424813c6/manifest.json)
- [Partition and token scorecard](../corpus/partitions/v1/be29e68619449efec0348e5747944b2a1861b78b82a93642df923de2424813c6/scorecard.json)
- [Document decisions](../corpus/partitions/v1/be29e68619449efec0348e5747944b2a1861b78b82a93642df923de2424813c6/decisions.jsonl) and [protected matches](../corpus/partitions/v1/be29e68619449efec0348e5747944b2a1861b78b82a93642df923de2424813c6/protected-matches.jsonl)
- [Adversarial fixture evidence](../evaluation/reports/llm-task-021-leakage-fixtures-v3.json)
- [Task completion evidence](../evaluation/reports/llm-task-021-validation.json)

| Current input and outcome | Count |
| --- | ---: |
| Task-020 eligible staging documents inspected | 227 |
| Exact protected content matches | 114 |
| Direct protected lineage/family matches | 113 |
| Connected components, both protected | 2 |
| Pairwise duplicate/family edges, not independent examples | 20,937 |
| Train / validation / test documents retained | 0 / 0 / 0 |
| Quarantined document decisions | 227 |
| Existing input proxy tokens, owned tokenizer v1.1.0 | 112,418 |
| Post-partition eligible training proxy tokens | 0 |
| Gen2 released training tokens / actual exposures | 0 / 0 |

These matches include shared content and recipe families; the counts do not prove that every document is a verbatim copy of an acceptance question. Conservative component propagation prevents a related board, document, code or template variant from crossing the protected boundary. The report makes that policy explicit instead of weakening it to produce a nonempty training file.

Cross-split collision counters are zero for the current release because no documents survive. **That is a safe rejection result, not corpus adequacy.** Separate authored controls create 100 independent descriptor families with 80 train, 10 validation and 10 test records, verify zero collisions across populated partitions, and prove that an inserted family collision is rejected. These controls are test material and contribute no training data or token credit.

## Bound protection inputs

The registry verifies and fingerprints all 900 task-018 cases, including their scenario, prompt, turns, context and oracle content; all 26 frozen Gen1 evaluation cases; 39 historical tokenizer hold-out IDs; 23 unique recoverable held-out records from the retained synthetic releases; and all 17 extraction regression inputs plus their accepted normalized representations. Historical copies are deduplicated in the guard registry and receive no independent data credit.

Protected text is read offline to construct comparison features. The new partition artifacts store guard IDs, fingerprints, match reasons and counts, not evaluation text or answer bodies. Private data, teacher outputs and unapproved external sources cannot enter through this tool: its only production input is the checksum-verified task-020 source staging release, which is revalidated by fresh ingestion recomputation.

Missing historical v1.0.0 source bytes remain missing. Their known IDs are denied, but their unavailable text cannot be semantically reconstructed from present data. The frozen suites still need separate confidential evaluation custody before model training; checksum integrity is not confidentiality.

## Family grouping and duplicate checks

Every descriptor requires source-family, document-family and repository ancestry. Repository-document inputs remain grouped by repository. For an owned deterministic generator, its broad repository is permission provenance: the recipe/template, scenario/circuit family and original document ancestry are the indivisible statistical groups. Board variants stay within their template family rather than being randomly scattered across partitions. Missing ancestry fails closed.

The legacy generator did not reserve splits before rendering. A separate, checksum-bound interpretation derives conservative groups from verified task/context fields: firmware templates, status-LED repair programs, I2C/OLED circuit tasks, and coarse task families where more precise metadata does not exist. This does not retrofit false generation history. Future generators must reserve descriptors before rendering using `reserve_before_render()` and validate each family/split with `require_render_assignment()`. The reservation binds the seed, policy, descriptor set and assignments; relabelled or unreserved families are denied. Post-render content/leakage checks remain mandatory because a reservation alone does not establish originality or source approval.

Matching checks cover exact raw and normalized hashes, exact content segments, lexical paraphrases with an explicit stop-word/alias vocabulary, protected-text containment, and structural code fingerprints. Code fingerprints ignore comments and identifier names while retaining operators, control flow and string literals. Numeric code variants are grouped conservatively as a family. Code is excluded from prose word-set matching so a changed operator does not become a duplicate merely because it uses the same words. Highly similar prose or renamed-code copies retain one deterministic representative; a numeric-only code-family match groups variants without claiming that changed numeric examples are byte duplicates.

Connected components are formed before partition assignment. A protected match anywhere taints the whole connected component. Remaining components use SHA-256 with the fixed policy seed and 80/10/10 target weights. Actual ratios depend on whole-component sizes and are reported rather than forced by splitting a family. Input ordering cannot change the output. Shared lineage, raw/normalized duplicates, near text/code matches and protected collisions are audited again across the resulting partitions.

This is an explicit lexical/structural detector, not a guarantee against arbitrary semantic paraphrases. Positive and negative controls cover exact copies, paraphrases, comment/identifier changes, numeric code variants, board/template variants, embedded oracle markers and historical IDs. Operators, string contents, unrelated prose, changed source bytes, tampered outputs, missing ancestry and altered pre-render reservations have regression coverage.

## Release and use boundary

The versioned output contains `train.jsonl`, `validation.jsonl`, `test.jsonl`, document/component decisions, protected and duplicate matches, and a scorecard. The three current data shards are intentionally empty. Quarantined payloads are not copied into the new release. Every retained record in a future candidate release carries its original ingestion lineage, component ID, partition and seed.

The manifest binds the ingestion manifest, protected input hashes, policy and implementation. `verify --recompute` reconstructs decisions and every output byte from source inputs. Publication uses the task-020 immutable writer and OS lock; interrupted publication can resume identical files and publishes its manifest last. Changed inputs/policy/implementation require a new version, never edits to retained artifacts. A checksum-only verification does not replace full recomputation before consumption.

`require_partition_use()` performs full verification and denies tokenizer fitting, pretraining, SFT, retrieval evaluation and tuning with `TRAINING_CORPUS_RELEASE_REQUIRED`. Only explicit partition review is currently supported. A filename containing `train` does not grant training permission. Task 023 must issue a separate reviewed corpus release and consumption contract, retaining all evaluation exclusions; do not flip `trainingAllowed` in these artifacts.

Token counts use the unchanged owned v1.1.0 tokenizer as a planning proxy. Unique accepted document tokens are recorded separately for each partition. Hypothetical 1/3/10 passes multiply only training exposure counts and never unique-data counts; actual training exposures remain zero. Task 024 must measure the later Gen2 tokenizer's representation.

The pilot matcher evaluates all pairs and fails closed above 5,000 descriptors or 2,000,000 record/guard comparisons. It never silently samples or skips comparisons. This bound supports the present inventory, not a billion-token corpus. Task 023 must add a reviewed indexed implementation with measured recall/resource evidence for larger inputs before release. General-language/code acquisition and new source approvals remain separate work.

## Reproduce and continue

From `Voltforge_AI`:

```powershell
rtk proxy .\.toolchains\gen1\Scripts\python.exe -B tools/partition_llm_corpus.py verify --release corpus/partitions/v1/be29e68619449efec0348e5747944b2a1861b78b82a93642df923de2424813c6 --recompute
rtk proxy .\.toolchains\gen1\Scripts\python.exe -B tools/evaluate_corpus_splits.py --report evaluation/reports/NEW-split-fixtures.json
rtk proxy .\.toolchains\gen1\Scripts\python.exe -B -m pytest -q tests/test_corpus_splitting.py
```

**Next: LLM-TASK-022 — expand owned electronics and compiler-verified domain data.** Author new independent recipe/circuit/problem families, reserve partitions before variants or augmentation, preserve exact-board compiler/calculation receipts, and run this contamination policy against every new candidate. New board labels or paraphrases of the present recipes do not create independent examples. Add versioned source/ingestion adapters instead of modifying the frozen task-019/020/021 inputs and code. Task 023 remains responsible for the broad corpus, scalable release checks and usable token budget.
