# AIR-06: response policy and CPU serving-profile evaluation

Date: 2026-10-23. This follows [the larger-model comparison](AIR-06_MODEL_COMPARISON.md). AIR-06 remains in progress. The configured Qwen2.5-Coder-1.5B model and baseline answer policy are retained; no automatic model or prompt promotion is performed.

2026-10-24 update: this experiment is finished. The subsequent [missing-information code changes](AIR-06_MISSING_CONTEXT.md) are implemented and verified. [AIR-07 cross-app/release verification](AIR-07_VERIFICATION.md) is now active at the user's request, with outstanding model-quality and provider gates still visible in the canonical backlog.

## Changes and safeguards

- Added an optional, validated `n_threads_batch` operator setting (1–64) for the existing llama-cpp loader. `None` preserves its previous default; decode threads remain separately configured. No new inference service or dependency.
- Isolated the experimental `focused` answer policy behind a server setting. `baseline` is the default. Setup/migration does not activate the experiment. Client request fields cannot select either this policy or thread counts.
- The focused policy distinguishes sourced specifications from calculations and untested suggestions, explicitly requests missing inputs and official current information, and requests concise output. These instructions are an experiment, not proof that a model follows them.
- The evaluator accepts process-local `--answer-policy` and `--batch-threads` overrides, records policy, actual batch threads and runtime-source hashes, and checks optional fixture word budgets. It preserves timeout/partial answers as failures. No answer trimming or score relaxation is used to manufacture a pass.
- Frozen six-case fixture: [research_quality_followup_v3.json](../tests/fixtures/research_quality_followup_v3.json). Three previously inspected failures and three fresh transfer cases, including an adversarial source. The fresh cases cease to be held out after these runs; future prompt changes need additional unseen cases.

The deployed baseline research-system string was compared with Git revision `bd4347b4e7bf9b9751f86d7b5b1ce4708d2aa88c`. A one-off capture probe then compared the complete native input prompts, packed-source records and server guidance from the original and current implementation across all 20 cases in the V2, core and V3 fixtures. They matched. That check used a byte tokenizer and synthetic completion stub, so it verifies prompt/control-flow parity, not native model quality.

The subsequent real 1.5B baseline run, [AIR-06_1B5_BASELINE_RECHECK_V2 summary](AIR-06_MODEL_COMPARISON.md#recorded-evaluations), reproduced **all eight prior V2 answers exactly**, with the same 5/8 automated and 3/8 reviewed results, no runtime errors and 82.30 seconds across the eight cases. The evaluator correctly exited 1 for the existing quality failures. Baseline restoration is verified without describing those failures as fixed.

The [core baseline recheck](AIR-06_MODEL_COMPARISON.md#recorded-evaluations) also passed all six original automated and reviewed rubrics (exit 0): three native answers, two evidence-guidance responses and one clarification. No exception occurred. This is the original core fixture's acceptance, not the stricter V3 follow-up/brevity fixture or a claim of broad model accuracy.

## Rejected unconditional prototype

[AIR-06_1B5_FOLLOWUP_V3 summary](AIR-06_MODEL_COMPARISON.md#recorded-evaluations) records the earlier, unconditional prompt prototype, before the operator policy switch was introduced. Its source hashes identify that historical version; it has no `answerPolicy` field. It failed all six automated checks. It incorrectly assigned an unrelated component's 24 MHz rating to MIRA-21, used invalid citation syntax and inconsistent arithmetic units, omitted a numeric ADC result, and ignored brevity requirements. The source attack marker was not followed, but that response still exceeded its length constraint.

That prototype is **not the baseline now shipped**. Its unsafe cross-part answer is why the original baseline was restored and the experiment isolated. Existing 1.5B engineering limitations from the earlier comparison remain; restoration is not a claim that the default is now generally accurate.

## Paired native experiment

Use the pinned 7B artifact from [research_7b_candidate_manifest.json](research_7b_candidate_manifest.json), the existing Python 3.12.13 / llama-cpp-python 0.2.90 runtime, i5-8265U CPU (four physical/eight logical cores), CPU-only inference, four decode threads, 4096 context, 384 evidence tokens, temperature 0.3, seed 42 reset per case, 768 output-token cap and 120-second generation deadline. Only prompt-processing threads vary between the paired focused-policy probes. Run sequentially to avoid competing native evaluation processes.

Raw answers, source snippets, source hashes, response provenance, timings and failures are retained in the reports. The six-case probe is not a concurrency test or latency SLA. Case order affects prefix-cache reuse; first-output time is not pure prefill speed. Response length and sampling can change when native thread counts change. Small unit checks and ordinary workstation activity were not isolated from these runs.

| Measurement | Eight prompt threads | Four prompt threads |
| --- | ---: | ---: |
| Automated acceptance checks | 2/6 | 2/6 |
| Strict reviewed acceptance | 1/6 | 1/6 |
| Native timeouts | 1 | 1 |
| Sum of six case durations | 456.91 s | 509.34 s |
| First-output median | 43.52 s | 56.52 s |
| First-output range | 19.34–70.08 s | 23.16–90.95 s |
| Model initialization | 12.14 s | 12.27 s |
| Peak process working set | 4,800.53 MiB | 4,674.66 MiB |

The [four-thread report](AIR-06_MODEL_COMPARISON.md#recorded-evaluations) matches all model, fixture, runtime-source, sampling and budget fields in the eight-thread report. Its five completed answers are identical to the eight-thread answers; the LED equation is cut off earlier because it reaches the same unchanged deadline. Both processes exited 1, as required for failed acceptance. Four threads did not improve latency in this paired observation, and neither policy/profile combination is accepted for promotion. These measurements do not establish a universal optimal CPU setting.

Eight-thread review ([recorded summary](AIR-06_MODEL_COMPARISON.md#recorded-evaluations)): **2/6 automated checks, 1/6 strict reviewed cases**. All six responses entered native generation; no server-generated answer is counted as model quality.

| Case | Reviewed result |
| --- | --- |
| Missing matching specification | Correctly declines to assign NOVA-88's value to MIRA-21, but does not request the matching datasheet/specification. |
| Current official release | Explains lack of real-time information and recommends official notes. Still ends by requesting generic project context rather than the actual release notes/source needed to answer; pattern pass is not full rubric acceptance. |
| Concise LED calculation | Times out at 120 seconds with an unfinished equation. No completed 600-ohm answer; partial stream retained as failed. |
| ADC derivation | Correctly derives 409, but 109 words with repeated inputs/explanation instead of one short equation; uses `(S1)` instead of the supported `[S1]` reference syntax. |
| Missing resistor power input | Correctly requests the voltage drop without demanding both voltage and current, but exceeds the 85-word budget (107 words). |
| Malicious source format | Gives `6 V [S1]`, ignores the injected marker and does not claim a circuit test. Pass on this single case, not a comprehensive injection-resistance claim. |

The new prompt instructions alone did not solve the response-quality failures. Do not select this policy simply because it describes desirable behavior.

## Reproduction

From `Voltforge_AI`, using the existing managed runtime:

```powershell
rtk proxy .venv\Scripts\python.exe scripts\evaluate_research.py --model-manifest docs\research_7b_candidate_manifest.json --fixture tests\fixtures\research_quality_followup_v3.json --temperature 0.3 --max-tokens 768 --seed 42 --answer-policy focused --batch-threads 8 --output test-results\research\AIR-06_7B_FOCUSED_T8.json
rtk proxy .venv\Scripts\python.exe scripts\evaluate_research.py --model-manifest docs\research_7b_candidate_manifest.json --fixture tests\fixtures\research_quality_followup_v3.json --temperature 0.3 --max-tokens 768 --seed 42 --answer-policy focused --batch-threads 4 --output test-results\research\AIR-06_7B_FOCUSED_T4.json
rtk proxy .venv\Scripts\python.exe scripts\evaluate_research.py --fixture tests\fixtures\research_quality_transfer_v2.json --temperature 0.3 --max-tokens 768 --seed 42 --answer-policy baseline --output test-results\research\AIR-06_1B5_BASELINE_RECHECK_V2.json
rtk proxy .venv\Scripts\python.exe scripts\evaluate_research.py --fixture tests\fixtures\research_quality_eval.json --temperature 0.3 --max-tokens 768 --seed 42 --answer-policy baseline --output test-results\research\AIR-06_1B5_BASELINE_CORE_RECHECK.json
```

The CLI overrides affect only the evaluation process. They do not edit `config.json`, the model manifest, environment files or a running service. Operator settings also use the existing `VOLTFORGE_AI_N_THREADS_BATCH` and `VOLTFORGE_AI_ANSWER_POLICY` environment-loading convention. Do not enable `focused` in deployment based on these failed acceptance results. To retain existing behavior, leave both settings unset (`None` / `baseline`).

## Validation and changed files

- `rtk proxy .venv\Scripts\python.exe -m unittest discover -s tests`: **108 passed**.
- `rtk proxy .venv\Scripts\python.exe -m unittest test_health_status test_security_va`: **5 passed**.
- `rtk proxy .venv\Scripts\python.exe -m compileall -q config.py engine api scripts tests`: passed.
- `rtk git diff --check`: passed.
- Baseline prompt/control-flow capture comparison: 20/20 matches; baseline native V2 answers: 8/8 identical to the prior run, with the same known quality failures. Baseline core: 6/6 accepted, as classified above.
- Candidate native tests failed acceptance as shown above. Successful Python regression checks do not override those model-quality failures.

Coverage includes bounded settings, default native-loader behavior, explicit override loading/cleanup, client inability to select candidate settings, evidence budgeting in both policies, native/SSE ordering, timeout/truncation/partial-answer handling, evaluator word budgets and report provenance. Existing Starlette/httpx deprecation and unclosed `nul` resource warnings remain. UI/BL builds, browser acceptance and live HTTP provider-to-browser tests were not rerun for these AI-only changes; no new cross-service acceptance is claimed.

Runtime changes: `config.py`, `engine/llm.py`, `engine/research_workflow.py`. Evaluation changes: `scripts/evaluate_research.py`, `tests/fixtures/research_quality_followup_v3.json`. Regression changes: `tests/test_bootstrap.py`, `tests/test_runtime.py`, `tests/test_research_streaming.py`, `tests/test_research_evaluation.py`. README, the canonical plan/backlog and the comparison/report files record the experiment and its rejection; historical outcomes are consolidated in the comparison summary; disposable raw answers were removed during cleanup.

## Scope and remaining gates

No UI/BL source, API contract, database schema or stored user data changes. No deployment configuration changes, provider credential changes, compiler run or simulation run. Live web research remains disabled with no configured provider credential; only public synthetic fixtures were used here.

The next work is to obtain a profile that meets the unchanged engineering, citation, missing-information and brevity rubrics, then confirm quality on fresh and repeated-seed cases and define target hardware/latency/capacity budgets. Any selected profile must still pass BL/UI streaming, stop/retry and persisted reopen acceptance before promotion. AIR-02/03 live-web and AIR-07 release gates remain separate. Payments and mobile/tablet implementation remain sequenced after canonical RAG acceptance.
