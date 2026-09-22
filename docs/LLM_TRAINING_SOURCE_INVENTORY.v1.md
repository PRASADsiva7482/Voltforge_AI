# Gen2 training-source inventory v1

LLM-TASK-019 inventories sources, records their current permitted uses, and measures the corpus gap before ingestion or training. It does not acquire an external training corpus, approve a Gen2 data release, train a tokenizer/model, or change any Gen1 approval or artifact.

The machine-readable [inventory](../data_governance/foundation/inventory.v1.json), [policy](../data_governance/foundation/policy.v1.json) and [manifest](../data_governance/foundation/manifest.v1.json) are the source of truth for this version. The [validation receipt](../evaluation/reports/llm-task-019-validation.json) records final checks and limitations.

## Verified inventory

| Scope | Current result | Meaning |
| --- | ---: | --- |
| Existing source definitions | 8 | Four currently pass existing training-source permission checks; four do not |
| Approved synthetic task shards | 4 / 250 records | Existing source, shard, generator, schema and preprocessing fingerprints verified |
| Existing training pool after exclusions | 227 records | 23 held-out records excluded; exact payload duplicates and task-018 leakage checked |
| Curated upstream fact packs | 8 / 61 records | Source permission and pack/schema checks pass; these facts are not added again to their derived synthetic training pool |
| Quarantined legacy datasets | 12 / 17,814 records | Zero training or runtime-retrieval credit; an owned generator does not establish the lineage of old outputs |
| Retained synthetic shard copies | 8 across 2 releases | Historical/current bytes verified; they contribute no additional unique-data credit |
| Proposed external sources | 7 | Permission research only; no content revision acquired or admitted |
| Explicitly excluded source classes | 4 | Private projects/chats, secrets, external teacher outputs, and evaluation material |
| Gen2 released training tokens | 0 | Normalization, family splits, release admission and the Gen2 tokenizer remain later work |

Discovery is bounded to `dataset.txt`, `model/artifacts/*.jsonl`, current synthetic shards, curated packs, their governance records and retained synthetic releases. It does not inspect private databases, conversation stores, credentials or arbitrary workspace text. Unknown files in those data roots get quarantine entries and zero token credit. Quarantined records are not copied into the inventory; only content-free fingerprints, existing manifest counts and denial reasons are retained.

The four approved existing source definitions are the domain generator, multitask generator, curated factual builder and verified synthetic pipeline. Their recorded authorship/permission basis is inherited from the existing source registry and checked against the exact source bytes. This is not a new external rights grant or a guarantee about authorship beyond that retained evidence. Every source receipt records origin, acquisition method, revision, checksum, license evidence, privacy classification, explicit training/retrieval/redistribution decisions, retention and deletion obligations. None gains redistribution rights merely because its source is available.

The existing tokenizer v1.1.0 lineage remains 227 training / 23 evaluation records. Evaluation IDs from every retained tokenizer version are excluded. Payload-equivalent renamed copies of retained held-out records also remain excluded. Historical v1.0.0 shard bytes are still unrecoverable at their declared fingerprints; current shards are never substituted. Full semantic leakage and family partitioning remain task 021, and missing historical bytes remain an explicit limitation.

## Tokens and the training gap

The measured existing pool contains **112,418 proxy tokens**, counted with the already-owned v1.1.0 byte-BPE tokenizer. Each record contributes canonical `task`, `input` and `output` JSON; record IDs and provenance headers are excluded. The encoder adds no BOS/EOS tokens. Structured context and answer payloads still contain protocol overhead: this count is not equivalent to 112,418 tokens of diverse natural-language prose.

These are exact counts for the stated existing encoder/representation and a planning proxy for Gen2. No new tokenizer is fitted. Gen2 token counts must be measured again after task 024; do not compare tokenizers without recording the representation and tokenizer fingerprints.

| Existing planning hypothesis | Planned token exposures | One-pass gap against the existing proxy pool |
| --- | ---: | ---: |
| Pilot lower range | 2,000,000,000 | 1,999,887,582 |
| Pilot upper range | 6,000,000,000 | 5,999,887,582 |
| Base lower range | 20,000,000,000 | 19,999,887,582 |
| Base upper range | 60,000,000,000 | 59,999,887,582 |

The JSON also gives the passes over this small pool needed to reach each exposure target and sensitivity at 1, 3 and 10 passes. Repetition increases exposures, never unique tokens, language breadth, or independent examples. These are arithmetic scenarios, not approved epochs or a capability forecast. The exposure ranges are task-017 planning hypotheses, subject to measured pilots; local workstation GPU availability is not a design limit.

For **Gen2 release readiness**, the available count is zero and the gap is the full planned range. Existing source permission and lexical exclusion checks cannot substitute for a released, normalized, deduplicated, family-split corpus with an owned tokenizer. The 227 structured task records are useful inputs for engineering and later instruction-data review; they do not provide a broad base-pretraining corpus.

## Proposed external sources and recorded permission evidence

The [proposal catalog](../data_governance/foundation/proposed-sources.v1.json) keeps all seven sources unadmitted. Each proposal has a concrete origin, intended acquisition process, candidate reference, permission observation, blocking reason and obligations. Unacquired content has `revision: null` and `contentSha256: null`; a URL, tag or webpage hash is not a corpus checksum.

| Proposed source | Observed primary evidence | Required next work |
| --- | --- | --- |
| English Wikipedia text dumps | Wikimedia describes text reuse under attribution/share-alike licenses and content/edition exceptions. [Terms, section 7](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use#7._Licensing_of_Content) | Select a dated dump; record page revisions, contributor attribution, exceptions and source-use/removal review |
| Selected Project Gutenberg English books | Terms distinguish work-specific copyright status and do not establish rights outside the US. [Publisher license](https://www.gutenberg.org/policy/license.html) | Review each title/edition and applicable jurisdiction; no collection-wide permission assumed |
| OpenStax University Physics Volume 2 | Current page states CC BY-NC-SA terms and specifically requires permission for LLM ingestion/training. [Publisher preface](https://openstax.org/books/university-physics-volume-2/pages/preface) | Keep excluded until source-specific permission covers the planned uses; no book extraction or training |
| FreeRTOS kernel C source | Tagged kernel license is MIT. [V11.1.0 license](https://raw.githubusercontent.com/FreeRTOS/FreeRTOS-Kernel/V11.1.0/LICENSE.md) | Resolve an exact commit/archive, audit individual files/exceptions and preserve required notices |
| Arduino AVR core | Representative `wiring.c` at 1.8.6 declares LGPL 2.1 or later. [Pinned source header](https://raw.githubusercontent.com/arduino/ArduinoCore-avr/1.8.6/cores/arduino/wiring.c) | Review all selected file licenses and planned use obligations; installed compiler dependencies do not become training data |
| AVR-LibC | Maintainer license describes modified BSD terms with per-file details. [Maintainer license](https://raw.githubusercontent.com/avrdudes/avr-libc/main/LICENSE) | Select an immutable revision and review source/documentation licenses file by file |
| GCC 14.2.0 compiler manual | Manual notice specifies GFDL 1.3 or later, an invariant section and cover-text obligations. [Versioned manual](https://gcc.gnu.org/onlinedocs/gcc-14.2.0/gcc/) | Review extraction/use obligations and acquire a complete fingerprinted manual snapshot; compiler execution rights are separate |

The [permission observation receipt](../data_governance/foundation/external-evidence.v1.json) hashes seven bounded publisher/maintainer responses and the authored summaries. Raw page bodies and training text were not retained. Evidence capture was an explicit online metadata operation; inventory generation and verification are offline. The observations establish what was reviewed, not new source-use approval or an immutable archive of remote pages. A mutable upstream page must be reviewed again at acquisition; a changed response needs a new receipt.

The code sources are concrete candidates for file-level review. General language and mathematical exposition remain major acquisition gaps. OpenStax is a permission-dependent proposal, not an automatically available math corpus. Continue owned deterministic mathematics/electronics authoring and source review through tasks 022/023; do not fill gaps with private conversations, external model answers, unreviewed scrapes or held-out fixtures.

## Use checks, removal and handoff

`data_governance.foundation.inventory.require_source_permission(source_id, usage)` is a source-level prerequisite for task-020 staging. It verifies the frozen inventory and current source bytes; unknown sources, unapproved uses, private/evaluation/teacher kinds and stale evidence fail closed. Its receipt explicitly states that a separate Gen2 corpus release is required and training is not authorized. Task 020 must bind each ingested document to a permitted source revision and task 021 must enforce family/split/leakage checks before release.

Source removal begins with a lineage impact report and blocked affected use. Existing registered Gen1 sources use `tools/source_removal_impact.py`; future Gen2 releases need document-to-source and artifact lineage. Reviewed deletion, retraining or removal of dependent releases follows the applicable retained obligation. This inventory performs no deletion, retirement, registry operation, rights rewrite or relabelling of retained data. Backup/restore and confidential evaluation custody gates remain open where previously recorded.

Changed sources, permissions, tokenizer, counting logic or discovered units need a new immutable inventory version. `freeze` refuses to overwrite v1; `verify --recompute` proves that the retained inputs and current code reproduce the exact inventory. Git attributes preserve the bound bytes across checkouts.

From `Voltforge_AI`:

```powershell
rtk proxy .\.toolchains\gen1\Scripts\python.exe -B tools/inventory_llm_sources.py verify --recompute
rtk proxy .\.toolchains\gen1\Scripts\python.exe -B tools/inventory_llm_sources.py draft --report evaluation/reports/NEW-source-inventory.json
rtk proxy .\.toolchains\gen1\Scripts\python.exe -B tools/audit_data_governance.py --check
rtk proxy .\.toolchains\gen1\Scripts\python.exe -B -m pytest -q tests/test_foundation_source_inventory.py
```

The evidence capture tool and `freeze` were one-time operations for v1. No external corpus acquisition, model download, runtime change or source approval is implicit in rerunning a read-only check. **Next: LLM-TASK-020 — reproducible corpus ingestion and normalization**, with only admitted source revisions eligible for staging.
