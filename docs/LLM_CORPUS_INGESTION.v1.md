# Gen2 reproducible corpus ingestion v1

LLM-TASK-020 implements the offline ingestion step for the owned LLM foundation. It produces immutable raw and normalized staging shards from exact, permitted source bytes. It does not release a Gen2 training corpus, fit a tokenizer, train a model, or change chat serving.

The source inventory remains the authority for this version. Its four approved synthetic shards contain 250 documents; ingestion admits the 227 inventoried eligible payloads and quarantines the 23 historical held-out records before raw content is persisted. The twelve legacy datasets and seven proposed external sources are not inputs. The eight curated packs are upstream evidence, not additional copies of training data.

## Current immutable outputs

- [Source staging manifest](../corpus/ingestion/v1/4eb23ca6950fb9d7fa9fd87bfbfcbe29aadd92d008a1d9d1e45229231078948d/manifest.json): 227 raw / 227 normalized records, 23 content-free quarantine decisions.
- [Extraction fixture manifest](../corpus/extraction-fixtures/v1/34b8ea2db7081e29c315d69da02f0b2814682866a59d1f58a5ce98e2fb6833ca/manifest.json): 17 authored regression inputs; 8 accepted and 9 quarantined, including one exact normalized duplicate. Fixtures never receive training credit.
- [Sample extraction review](../evaluation/reports/llm-task-020-extraction-review.json): two samples from each source shard and all 17 fixture decisions, with checksums and preservation criteria.
- [Throughput and memory receipt](../evaluation/reports/llm-task-020-memory.json): fresh-process measurements at two input sizes.
- [Completion evidence](../evaluation/reports/llm-task-020-validation.json): final test counts, preservation checks and continuation checkpoint.

The content-addressed directory name binds the input plan, source permission receipts, original shard manifests, eligible record fingerprints, parser/policy code, Python/Unicode versions, pinned PDF parser and installed PDF parser code fingerprint. The final manifest binds every generated shard. Source revisions, document locators, raw hashes and source permission receipts accompany every normalized record. Source removal can trace affected documents using `lineage.sourceIds`; removal/deletion and dependent model retirement still require the governed lifecycle rather than editing immutable bytes.

## Processing and resume

1. Revalidate the frozen task-019 inventory, source permissions and exact input unit hashes. A caller-supplied source ID cannot authorize a different file. The production build command selects only the four currently inventoried approved shards.
2. Read JSONL a bounded line at a time. Hash oversized lines in bounded chunks, quarantine them, then continue at the next document. Plain text, Markdown, code and HTML documents use a 1 MiB input limit. PDFs use an 8 MiB input limit, a 100-page limit, a 1 MiB decompressed stream limit and a 20-second worker deadline.
3. Enforce known held-out/payload admission, strict decoding, broken/hidden text and secret-pattern checks. Reject rather than silently replace undecodable text. Scan HTML before boilerplate removal as well as extracted text. No rejected raw document body enters staging or the SQLite journal; quarantine logs contain hashes, lineage and reason codes. Synthetic credential sentinels exist only in the explicitly marked regression inputs.
4. Normalize NFC and line endings, retaining SI symbols, superscripts, tabs and meaningful code whitespace. HTML extraction removes navigation/footer/script regions, preserves preformatted code, retains MathML markup and TeX math scripts, and explicitly represents HTML superscripts/subscripts. PDF extraction uses fixed-width layout and removes only matching prose margins across at least three pages.
5. Keep exact normalized-text fingerprints in a disk-backed SQLite index. A committed document transaction contains the raw/normalized records or its content-free denial. Resume checks the input/parser plan and journal signatures, reconstructs the duplicate index, and skips already committed extraction. An OS-released writer lock prevents concurrent publication and permits restart after process death. Resume streams past prior input lines, so it is bounded but not a constant-time seek for large JSONL inputs.
6. Export complete, checksum-named raw, normalized and quarantine JSONL shards, then publish the manifest last. Target shard size is 64 records / 4 MiB; a single permitted document can exceed this target up to the separately bounded document/serialization size. Completed shards are never overwritten. A completed build verifies and returns `already-complete`; a partial export resumes by matching existing bytes. Unpublished `.pending-*` temporary files are ignored, carry no release credit, and may need reviewed cleanup after a crash.

Raw JSONL shard entries retain each accepted document's exact bytes as base64 plus lineage. Normalized records contain the extracted text and its hash. For the existing task-schema source, the representation is canonical `task`, `input`, `output` JSON, retaining firmware, citations, tool evidence and proposal constraints. It is still small structured synthetic data, not a broad language corpus. The current build grants no new token credit beyond task 019; task 024 must measure the new tokenizer representation.

## Extraction review and measured limits

The reviewed fixtures preserve `Ω`, `µ`, `°C`, superscripts, decomposed-accent NFC, UTF-16 and explicitly declared Windows-1252, HTML/MathML equations, TeX formulas, fenced-code blank lines and nested indentation. The digital PDF code fixture retains its formula and four-space indentation. A separate three-page fixture proves repeated-margin removal. Blank, malformed and encrypted PDFs, malformed code/math HTML, hidden characters, invalid encoding, oversized documents and secret-shaped text have automated quarantine coverage.

The eight source samples preserve their nested task/input/output values exactly, including multi-line firmware and evidence. This checks extraction fidelity; it does not re-certify electronics correctness or model capability. Both manifests reproduce byte-for-byte in new temporary output roots. Interrupted ingestion, publication interruption, journal tampering, changed source bytes, changed parser identity, output tampering and concurrent writers have regression coverage.

The PDF adapter uses the pinned [pypdf 6.17.0 extraction API](https://pypdf.readthedocs.io/en/6.17.0/user/extract-text.html). Its documentation explains that PDF text lacks reliable semantic structure and that image-only pages need OCR. This implementation quarantines pages without extractable text and flags extracted PDF documents for source-specific review. Multi-column order, complex visual equations and OCR accuracy are not established by these fixtures. File/stream/page/deadline limits are applied, but there is no hard aggregate PDF-worker RSS cap or complete hostile-PDF sandbox; arbitrary external PDF acquisition remains unadmitted and needs those controls plus source-specific review before a new production plan.

| Development measurement | 1,000 documents | 10,000 documents |
| --- | ---: | ---: |
| Input bytes | 1,059,000 | 10,590,000 |
| End-to-end seconds | 11.20 | 113.09 |
| Documents/second | 89.29 | 88.42 |
| Peak traced Python allocation, bytes | 2,159,297 | 2,143,428 |
| Parent peak working set, bytes | 81,272,832 | 81,510,400 |

These runs include normalization, durable per-document SQLite commits, deduplication, export and verification. The tenfold input increase did not cause linear allocation growth. Traced allocation excludes native/OS memory and PDF child processes; peak working set is an observation, not an enforced cap. These are local synthetic JSONL measurements, not server capacity evidence. Corpus-scale parallelism, batch-commit tuning and server measurements need separate evidence; workstation GPU capacity does not determine the model plan.

Secret detection is conservative pattern matching, not comprehensive privacy clearance. Source permission and private-data exclusion remain necessary. Exact duplicate removal does not establish semantic deduplication or family-level partitioning. The frozen evaluation suite remains unchanged, with confidential custody still required before training. The historical tokenizer v1.0.0 missing-byte and external backup/restore limitations remain open.

## Commands and next task

From `Voltforge_AI`, install the isolated build dependency without changing the runtime framework pin:

```powershell
rtk proxy uv pip install --python .toolchains/gen1/Scripts/python.exe --no-deps -r requirements-corpus.txt
rtk proxy .\.toolchains\gen1\Scripts\python.exe -B tools/ingest_llm_corpus.py build
rtk proxy .\.toolchains\gen1\Scripts\python.exe -B tools/ingest_llm_corpus.py fixtures
rtk proxy .\.toolchains\gen1\Scripts\python.exe -B tools/ingest_llm_corpus.py verify --release corpus/ingestion/v1/4eb23ca6950fb9d7fa9fd87bfbfcbe29aadd92d008a1d9d1e45229231078948d --recompute
rtk proxy .\.toolchains\gen1\Scripts\python.exe -B tools/review_ingestion_samples.py --source-release corpus/ingestion/v1/4eb23ca6950fb9d7fa9fd87bfbfcbe29aadd92d008a1d9d1e45229231078948d --fixture-release corpus/extraction-fixtures/v1/34b8ea2db7081e29c315d69da02f0b2814682866a59d1f58a5ce98e2fb6833ca --report evaluation/reports/NEW-ingestion-review.json
rtk proxy .\.toolchains\gen1\Scripts\python.exe -B tools/evaluate_ingestion_memory.py --report evaluation/reports/NEW-ingestion-memory.json
rtk proxy .\.toolchains\gen1\Scripts\python.exe -B -m pytest -q tests/test_corpus_ingestion.py
```

Use new receipt paths; evidence tools refuse replacement. Corpus download is a separate, currently unimplemented acquisition step that requires newly reviewed source/revision evidence. No builder or parser is imported by chat serving, and ingestion/recomputation are offline. The task-019 metadata research and this build dependency installation are distinct from corpus acquisition.

**Next: LLM-TASK-021 — prevent train/evaluation leakage and near-duplicate inflation.** Build source/document/family partitions, cross-split exact/near-duplicate checks and reproducible split manifests over this staging format. Keep `trainingAllowed: false` until the later corpus release and training gates pass.
