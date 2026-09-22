# Corpus leakage precision review

Task **LLM-TASK-023**, continuation after tokenizer preparation. This completes
the independent precision-review substep. Task 023's corpus release and task
024's production tokenizer release remain **IN_PROGRESS**.

## Finding

The frozen comparator treats sufficient unordered word overlap with a protected
prompt as a blocking match. A long document can contain all those words in
different places, without reproducing the protected sentence or event. The
whole repository then inherits the document's quarantine. A word-set match
therefore needs careful interpretation; it is not proof that a source copied a
protected answer.

The new read-only `data_governance/leakage_review/` module reports two kinds of
existing blocking evidence:

- Identity, exact literal and code-family matches retain their mandatory
  exclusion under the current policy. This classification does not infer intent.
- Lexical-only matches require review. Reports provide distinct-word counts,
  Jaccard, protected-word coverage and the shortest window containing the shared
  words in any order. A short window is not itself a copied phrase.

Reports contain no protected prompt text, matched word lists or word positions.
The checker calls the existing matcher, verifies its exact decisions and never
overrides them. Both categories remain blocked; a no-match result also grants
no source-use permission or corpus admission.

## Independent controls

The authored suite has **56 cases from 10 families**. Eight narrative families
have exact copies, normalized copies, embedded copies, glossary overlap and
unrelated-topic controls. Separate code and identity families exercise renamed
code, numeric changes and unconditional identity exclusions. Two narrative
paraphrases demonstrate why a lexical match cannot simply be ignored.

| Authored label | Current matcher blocks | Current matcher does not match |
| --- | ---: | ---: |
| Protected positive | 32 | 0 |
| Unrelated negative | 16 | 8 |

All 32 positive controls remain blocked. Thirty have identity/literal/code
signals, and two paraphrases require lexical review. All 16 negative matches
are lexical-only glossary controls; none are automatically readmitted.

The current matcher's precision is 32/48 (66.7%) and recall is 32/32 (100%)
**on these constructed controls only**. These figures are not an estimate of
corpus-wide precision, semantic recall or model quality. Variants share families;
56 cases do not represent 56 independent sources. Controls never enter training.

## Real corpus impact

The existing 38-document candidate is checked against the same 1,005 protected
guards. The review recomputes partitions and requires exact equality with the
immutable `decisions.jsonl`; all historical exclusions remain in force.

The real report distinguishes direct matches from family-propagated quarantine,
binds the protected registry and corpus hashes, and publishes no protected text.
Both direct document matches are lexical-only. Repository ancestry propagates
quarantine to five additional documents, keeping all seven excluded.
Real lexical matches remain **unadjudicated**. The independent glossary examples
do not establish that the Python documentation's matches are false positives.
The existing corpus retains 26 train, zero validation, five test and seven
quarantined records, with 722,234 existing-tokenizer proxy train tokens.

This review changes no matcher threshold, corpus shard, approved use, tokenizer,
model or serving route. Source admission, independent language and validation
coverage, real corpus scale measurements and corpus release acceptance remain
open. The current input corpus still cannot fit a production tokenizer.

## Verification and continuation

```powershell
.toolchains\gen1\Scripts\python.exe -B tools/review_corpus_leakage.py controls
.toolchains\gen1\Scripts\python.exe -B tools/review_corpus_leakage.py build
.toolchains\gen1\Scripts\python.exe -B tools/review_corpus_leakage.py verify --release corpus/leakage-reviews/v1/<content-hash> --recompute
.toolchains\gen1\Scripts\python.exe -B -m pytest -q tests/test_corpus_leakage_review.py
```

The integration tests require the immutable review produced by `build`. Review
artifacts bind source code, policy, corpus, protected registry and evidence
checksums. They are write-once, manifest-last and reproducible offline. Bounds
cover match count, lexical pairs, input characters, word tokens, word-set work
and sliding-window work; exhaustion raises an error and never permits admission.

Continue task 023 with exact source-use admission and new independent corpus
families. Before adopting a changed matcher for new data, freeze its proposed
criteria and test literal, identifier, code-family and paraphrase recall as well
as unrelated word-overlap precision. Use a fresh corpus version, retain the
current quarantine history and require applicable source/quality/split evidence.
This diagnostic report is not permission to exempt old quarantined documents.
