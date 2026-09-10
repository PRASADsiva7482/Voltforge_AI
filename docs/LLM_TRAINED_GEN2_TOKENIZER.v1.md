# Task 024 owned corpus-trained tokenizer release

The selected **vfdlm-g2-byte-bpe 0.1.0** tokenizer has **16,384
entries**. It was fitted from raw bytes of the task-023 admitted training split:
42 documents, 1,475,271 bytes. It imports no pretrained vocabulary, merges or
weights. Both requested candidate targets were reached from the same exact
training data. All old Gen1 tokenizers and Gen2 fixture files remain immutable.

[Release manifest](../model/tokenizers/gen2/releases/f2bf8a23c7d251252bf120ea3262875152e3dede6a4549e3ea147bc867a09d65/manifest.json) |
[Candidate comparison](../model/tokenizers/gen2/comparisons/v1/d16d202707d79ee8dd5f13493b655a56838ebceeb745aeff86376ca12f477d40/report.json) |
[Gen2 corpus counts](../corpus/token-counts/gen2/v1/e94d9f2c168dda354fa5c76ab8decabd252d66002c99c300eccdf4e490b5614f/report.json) |
[Real model/context integration](../model/tokenizers/gen2/integration/v1/f17eae152f367f847fa8da33830b77317067374a19def449f76c9fcaa6bc991c/report.json)

## Completed task plan

1. Add an evidence-verifying adapter for task 023's exact admitted corpus;
   recompute source/split admission before merge fitting and enforce byte limits.
2. Fit both 16,384 and 32,768 ceilings in fresh processes, retaining exact train
   lineage, code, policy, vocabulary, merges and resource measurements.
3. Freeze both candidate artifacts before reading permitted validation data;
   compare per-domain compression, fertility, exact round trips, SI/Unicode,
   long-code cost, literal markers and codec throughput with the 3,072 baseline.
4. Apply the predeclared vocabulary selection rule, publish a write-once 0.1.0
   version record and immutable release, and recount the admitted corpus.
5. Feed compiled IDs into real randomly initialized decoder embeddings and
   logits with the same tokenizer/template descriptor. Check mismatches before
   forward execution, and preserve all historical release evidence.

## Candidate comparison and selection

| Target entries | Actual entries | Train tokens | Validation tokens |
| ---: | ---: | ---: | ---: |
| 16,384 | 16,384 | 139,542 | 19,348 |
| 32,768 | 32,768 | 95,147 | 18,098 |

The policy selects 32k only when its equal-domain validation token reduction
over 16k is at least 8%, with no domain regressing by more than 10%. Otherwise
it retains 16k. The measured macro reduction is
**7.29%**;
the selected target is **16,384**.
This criterion selects a vocabulary for this input corpus, not an optimal
model architecture or a demonstrated language-quality result. Validation text
is measured only after both merge sets are frozen; it never enters fitting.
Test and sealed acceptance data do not select the tokenizer. Source/leakage
verifiers still inspect their protected references as required by corpus gates.

| Public measurement category | Gen1 3,072 tokens | Gen2 16,384 tokens | Gen2 32,768 tokens |
| --- | ---: | ---: | ---: |
| code | 43 | 23 | 21 |
| long-code | 7395 | 5697 | 5613 |
| markers | 38 | 37 | 37 |
| si | 50 | 46 | 46 |
| text | 36 | 21 | 20 |
| unicode | 70 | 69 | 69 |

These public implementation measurements supplement actual train/validation
counts. Whitespace fertility is a proxy. The historical baseline had different
fitting data and is diagnostic. The report records three-iteration codec
throughput; those host timings establish no server serving capacity.

## Token contract, framing and model integration

The existing 16 special IDs, all 256 byte values, no-normalization UTF-8 with
surrogatepass, and `vfdlm-g2-chat-v1` template version 1.0.0 are retained.
Literal `<|assistant|>` and other marker-looking content use byte tokens;
only trusted role framing inserts structural IDs. Exact content marker costs
are measured rather than approximated or rewritten. All byte values, SI text,
Unicode, surrogate cases and incremental decoding are covered by checks.

`context_compiler/gen2_release.py` reuses the frozen ID compiler and passes its
exact binding to `model/gen2_tokenizer_training/model_boundary.py`. The latter
constructs a real decoder from the existing owned Gen1 tensor mathematics under
a new Gen2 tokenizer binding. The integration profile has
**533,600 actual parameters**, embeddings of
shape **[16384, 32]**, and logits of shape
**[1, 72, 16384]**. Cached final-token logits match the
full-prefix result. A clean process imports pinned PyTorch
**2.8.0+cpu** before tensor allocation.

This proves model/context consumption of the selected version, beyond a
dictionary-only interface check. The finite random-weight integration profile
does not complete task 025's configurable architecture/profile acceptance or
claim learned 4k context. No model optimizer step, checkpoint load/save or
application activation occurs. Existing application routing remains unchanged.

## Corpus recount and resource limits

The selected tokenizer counts **139,542 unique train
tokens**, **19,348 validation tokens** and
**708 test tokens**. Train/validation counts reuse
the selected comparison's exact checked records; test text is encoded only
after final vocabulary registration, with no selection feedback. Quarantines
remain excluded. These are token positions in deduplicated documents, not
distinct vocabulary types. Actual model-training exposures remain zero.

Each candidate receipt records measured fit time and process peak memory.
Limits were declared before fitting: at most 96 train documents, 2 MiB train
bytes, 300 fit seconds, 2 GiB peak process memory, 32 MiB vocabulary bytes and
minimum pair frequency two. Source lineage and byte/token counts are immutable.
Production data/compute budget selection remains in tasks 031/032.

## Release and retention boundary

The 0.1.0 record approves this immutable offline tokenizer artifact for later
training-input integration. It does not approve a model training run, serving,
deployment or external-retention acceptance. The inherited VFAI-FU-015 policy
is an accepted-for-later Gen1 synthetic-history follow-up; the foundation model
release policy requires backup/restore in integrated-assistant acceptance.
Owner-approved encrypted external backup/restore requirements remain intact
for deployment in tasks 041/054/055. No owner signature, backup or restore
receipt is fabricated, and the historical retention policy is not changed.

The authoritative task-024 definition requires the immutable tokenizer plus
model/context version consumption. Those are the acceptance criteria here;
deployment-only gates do not turn an offline tokenizer fit into a serving
approval. The existing byte-codec fixture runtime gate stays closed.

## Commands

From `Voltforge_AI`:

```powershell
rtk proxy .toolchains\gen1\Scripts\python.exe -B tools/train_gen2_tokenizer.py verify --release model/tokenizers/gen2/releases/f2bf8a23c7d251252bf120ea3262875152e3dede6a4549e3ea147bc867a09d65 --recompute
rtk proxy .toolchains\gen1\Scripts\python.exe -B tools/train_gen2_tokenizer.py integration --release model/tokenizers/gen2/releases/f2bf8a23c7d251252bf120ea3262875152e3dede6a4549e3ea147bc867a09d65
rtk proxy .toolchains\gen1\Scripts\python.exe -B -m pytest -q tests/test_trained_gen2_tokenizer.py tests/test_gen2_tokenizer.py
```

New candidate revisions require new immutable namespaces/policies; do not rerun
fits into the accepted candidate pair or overwrite the 0.1.0 version record.
Changing bytes, source permissions, split ownership, merge rules, special IDs,
template or model vocabulary dimensions fails verification. Source revocation
continues to block the tokenizer's corpus lineage through task 023's verifier.

The release also binds the exact reviewed `model/gen1/config.py`,
`model/gen1/model.py`, legacy codec/compiler, and corpus-verifier bytes used
for its evidence. Later architecture work should extend separate versioned
modules; changing these dependencies in place invalidates this release's
verification and requires a new immutable evidence revision.
