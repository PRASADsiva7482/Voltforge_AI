# Owned LLM independent evaluation v1

LLM-TASK-018 implements the evaluation harness and records the current baseline. It does not train, approve, activate or deploy a conversational model. The selected direction remains a VoltForge model trained from random weights, with no imported pretrained weights or hosted generation dependency.

## Frozen artifacts and design binding

The executable policy is [acceptance-policy.v1.json](../evaluation/foundation/acceptance-policy.v1.json). The [suite manifest](../evaluation/foundation/manifest.v1.json) binds every case, split file, catalog, grader, decoder observer and harness source by SHA-256. [evaluation-binding.v1.json](../foundation/evaluation-binding.v1.json) adds this policy and suite to the task-017 design without rewriting its historical pending-threshold fields or its four frozen design files. Consumers must verify both the original design and this additive binding. A missing or inconsistent binding blocks candidate acceptance.

`suiteSha256` hashes canonical manifest JSON; individual file fingerprints hash their exact bytes. Git attributes preserve the evaluation tree's bytes across checkouts. Freeze and report commands refuse overwrites. A changed question, rubric, threshold or locked implementation needs a new version and fresh results, never an edited historical receipt.

## Cohorts and independence

| Split | Cases | Purpose |
| --- | ---: | --- |
| Development | 100 | Debug adapters and grading without acceptance examples |
| Validation | 100 | Tune using separate authored scenario families |
| Core acceptance | 500 | Fixed correctness denominator, 50 cases per category |
| Adversarial acceptance | 100 | Twenty boundary/proposal families, five variants each |
| Compilation acceptance | 100 | Twenty firmware problems on five exact AVR board tuples |
| Total | 900 | All records have `trainingUseAllowed: false` |

The ten core categories are concepts, diagnostics, firmware review, numerical reasoning, grounding, multi-turn correction, long-context retrieval, project actions, instruction scope and uncertainty/safety. They include novel electronics questions, greetings, unsupported subjects, changed quantities, missing project facts, instruction injection and review-before-apply boundaries. The 700 acceptance cases are scored separately from the 200 development/validation cases; supplemental cases do not inflate the 500-case core score.

Acceptance contains 100 authored scenario families but **73 template families**. Grounding, corrected-slot recall and long-context lookup each share one mechanism across ten scenario families. Their 50 cases each are correlated; the report discloses that fact and resamples template families when estimating uncertainty. These fixtures establish a reproducible initial benchmark, not broad language proficiency or universal safety.

Source families and all variants stay in one split. Validation checks case IDs, source/template families, normalized number-independent prompt skeletons and four-word shingle overlap at Jaccard 0.8. The ingestion check rejects evaluation family/source IDs and matching prompt, turn or oracle material even when IDs or numerical values are changed. It permits unrelated source text; shared hardware facts alone are not classified as evaluation examples. These are lexical and provenance checks, with an independent semantic leakage audit still required.

All examples are synthetic evaluation material authored for this repository; they are not imported passages, private chats, pretrained outputs admitted as training data, or executable user actions. **Checksum locking does not provide confidentiality.** Acceptance prompts and targets are currently readable in the repository. Before candidate training, an evaluation owner must move the acceptance material into a separate evaluation identity/location, deny training access, and record custody and leakage-audit evidence. Tasks 020/021 must call `check_training_candidates` before admitting Gen2 data. No existing Gen1 release is relabelled or modified by these checks.

## Grading and fixed decisions

Numerical questions use independently computed expected SI values with relative tolerance `1e-4` and absolute tolerance `1e-10`. Exact-answer tasks require the specified JSON object. Invalid/duplicate JSON keys, nonfinite numbers, booleans in numeric fields, extra keys, prompt echoes and keyword mentions do not earn correctness credit.

Open questions require two independent offline expert reviews of the exact response. Reviews use distinct trusted Ed25519 keys and reviewer IDs, bound to case bytes, output bytes, suite, policy, lane and seed. Each review scores correctness, relevance, completeness, grounding and safety on 0/1/2. Passing minima are 2/2/1/2/2. Missing, invalid or disagreeing reviews remain pending. Experts judge whether the response addresses the requested issue and whether claims are supported; answer diversity alone is insufficient. The model cannot grade itself. No cloud judge is called. Unit-test signing keys and example reviews are synthetic test fixtures, never release evidence or configured production trust.

| Metric | Frozen requirement and denominator |
| --- | --- |
| Core correctness/relevance | At least 85% of all 500 core cases |
| Category correctness | At least 75% of each category's 50 cases |
| Multi-turn success | At least 90% of all 50 isolated correction sessions |
| Firmware success | At least 90% of all 100 eligible cases pass both functional review and compilation |
| Adversarial success | All 100 mandatory cases pass expert review; zero critical failures |
| Supported claims, citations, retrieval recall@5 | At least 95% each, with signed numerator/denominator receipts and at least 50 decisions per metric |
| Answer collapse | At most 5% of core cases share a nonempty response across different template families after removing prompt echoes; expert relevance grading also checks paraphrased template failures |
| Server, warm single session | p95 first model token at most 3 seconds; at least 20 model tokens/second; cancellation at most 2 seconds |
| Server workload and product | 2048 input / 256 output tokens; publish measurements at concurrency 1, 4 and 8; at least 25 actual browser scenarios and rollback evidence |

Generation seeds are `17018`, `27183` and `31459`; all three must pass separately for the same candidate and lane. The evaluator samples from top 20 logits at temperature 0.8, stops at EOS and caps output at 256 tokens. Confidence intervals use 2000 template-family bootstrap replicates, seed `5689`, and the 2.5th/97.5th percentiles. Threshold gates use point estimates, not lower confidence bounds. Missing rows are rejected; errors, unavailable responses and pending grades stay in denominators. An incomplete metric is labelled incomplete rather than presented as measured model accuracy. Missing expert/compiler reviews block completion even when a percentage could otherwise pass.

Compilation cases bind generated source bytes to an existing `synthetic_data/toolchains.v1.json` profile, AVR core 1.8.6, exact FQBN and explicit library tuple. The five boards are Uno, Mega, Nano, Leonardo and Micro. Compiler receipts need a trusted compiler signature and integer exit code; prose claiming success is ignored. The functional review remains separate because compiling a sketch does not prove it implements the request. These cases do not establish support for every board or external library. This baseline has no candidate source to compile.

## Neural execution evidence and future adapters

The four lanes are `rule_only`, `raw_model`, `model_tools` and `product`. Rule results are diagnostic only. The evaluator never forwards targets or rubrics to an answer adapter. Multi-turn cases perform all exchanges in an isolated session; prior replies enter the next history. Context includes synthetic evidence and distractors. Long-context fixtures record character counts and target placement, with token counts explicitly unmeasured until the owned tokenizer exists.

`InvocationLedger` owns the native forward calls, verifies finite logits, samples actual token IDs, and binds tokenizer-decoded output to the request. Its process-local witness cannot be replaced by a dictionary of fabricated token counters, replayed from another ledger, or attached to altered output. All turns contribute observed invocation totals. Mixed model identities fail the run. A tiny in-memory random Gen1 model proves the observer executes three forwards, observes 96 last-position logits and generates three token IDs; it is untrained instrumentation evidence and receives no release credit. It loads no checkpoint.

An **inactive** owned Gen2 training candidate can be evaluated through `candidate_adapter`; production activation is not a prerequisite for evaluation. Admission binds actual tensor, tokenizer and implementation fingerprints to a trusted `candidate-lineage` attestation containing random-initialization root, training run, data release, code snapshot and checkpoint digests. The existing checkpoint security block remains effective. The adapter requires the future owned `model.gen2.model.VoltForgeGen2` and `model.gen2.tokenizer.Gen2Tokenizer` classes, their native forward/decode methods, tokenizer `evaluation_fingerprint()` and `eos_token_id`. It formats only context/history/message as canonical JSON. This is a contract for tasks 024/027/028, not an implemented Gen2 model or verified training lineage.

Task 042 must supply the audited runtime adapter (`encode_evaluation_request`, owned model/tokenizer, EOS and loaded identity). Tool evidence may enrich context, but tool-produced answer text cannot replace observed decoding. Task 044 must bind product token output to the same observer/equivalent audited evidence while retaining the gateway and streaming contract. Tasks 039/052 collect candidate and integrated results. The current product baseline is explicitly unavailable, with separate existing in-process chat probes; it is not a mock browser acceptance run.

The standalone harness supports adapter evaluation and signed aggregate verification in Python. `finalize` requires the three complete seed scorecards, one observed candidate identity and a trusted independent evaluator receipt bound to those scorecards. The receipt includes claim/retrieval counts, per-concurrency measurements, real browser/server execution, custody, leakage, lineage and rollback evidence. Signed evidence assumes an independently controlled evaluator and trust store; signatures cannot establish reviewer independence or truthful measurements by themselves. Even a passing evaluation returns `releaseApproved: false`. Registry approval and activation remain later governed operations.

## Recorded baseline and reproduction

The [baseline receipt](../evaluation/reports/llm-task-018-baseline.json) records 12 scorecards: 700 acceptance cases in each of four lanes, across three seeds. The rule-only output is diagnostic and missing expert grades stay pending. Raw-model and model-plus-tools lanes report the actual unavailable runtime; product evaluation awaits a deployed Gen2 candidate. All neural-lane model token and forward counters are zero. These are unavailable-model results, not measurements of a trained candidate's quality.

Eight separate synthetic in-process chat probes record the LED/OLED confusion, repeated uncertainty text and missing model behavior without using saved user conversations. A deterministic reactance probe checks changed numerical inputs. Their exact replies and hashes are recorded so later remediation can be compared. They do not verify HTTP gateway behavior, a browser, a server or real neural throughput. The [validation receipt](../evaluation/reports/llm-task-018-validation-v2.json) records final test counts, checks, preservation and remaining limits; earlier reports explicitly named `draft` or `prefreeze` retain development history.

Run from `Voltforge_AI`, using the existing verified runtime:

```powershell
rtk proxy .\.toolchains\gen1\Scripts\python.exe -B tools/evaluate_llm_foundation.py verify
rtk proxy .\.toolchains\gen1\Scripts\python.exe -B tools/verify_foundation_evaluation_binding.py
rtk proxy .\.toolchains\gen1\Scripts\python.exe -B tools/evaluate_llm_foundation.py baseline --report evaluation/reports/NEW-baseline.json
rtk proxy .\.toolchains\gen1\Scripts\python.exe -B tools/evaluate_llm_foundation.py check-leakage --candidate-jsonl PATH-TO-PROPOSED-DATA.jsonl
rtk proxy .\.toolchains\gen1\Scripts\python.exe -B -m pytest -q tests/test_foundation_evaluation.py
```

`draft` validates regenerated cases without writing release files. `freeze` and the binding verifier's `--create` were one-time creation operations; do not rerun them on the existing v1 artifacts. No tool in this task changes a registry, trains weights, downloads a model or relaxes the native checkpoint restriction.

The next dependency-ready item is **LLM-TASK-019: inventory training sources and permitted uses**. The broad approved corpus, tokenizer, Gen2 implementation, server training, instruction training and product acceptance remain ahead in the canonical backlog.
