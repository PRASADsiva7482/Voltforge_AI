# AIR-06: narrow missing-information handling

Date: 2026-10-24. This implements the next bounded follow-up from [the serving-profile experiment](AIR-06_SERVING_PROFILE.md). It retains the baseline prompt, configured 1.5B GGUF, thread defaults, generation deadline and source budgets. AIR-06 remains in progress pending broader engineering quality and serving-profile acceptance. At the user's request, independent AIR-07 verification has started; that does not waive these acceptance gates.

## Root causes and implementation

The existing specification guard checked whether any excerpt contained a quantity of the requested dimension. An unrelated part's frequency therefore prevented missing-information guidance, even when the same excerpt explicitly said the requested part's rating was unavailable. The model then sometimes transferred the wrong value or failed to request the matching reference.

`engine/evidence_guard.py` now recognizes a deliberately small grammar of explicit absence statements for one identifiable part and the requested dimension. Every exact mention must be an absence statement. Positive ratings, multiple parts, table headings, aliases/suffixes, conflicting mentions, mismatched qualifiers (such as minimum versus maximum or input versus output) and unsupported wording defer to ordinary inference. The response qualifies its limitation to the supplied excerpts and requests that part's datasheet, revision and operating conditions. It never derives a replacement value from another part.

Current-version/release questions with no supplied evidence, history or project context previously depended entirely on model wording. The new bounded clarification path states that current information cannot be verified and requests official release notes or pasted official text. It does not start a search, fetch a URL or invent a release. Concept questions, electrical-current questions, inline supplied notes/version values and unsupported question forms defer to the existing model path.

`engine/llm.py` integrates both checks through the existing `clarification` / `evidence-guidance` modes and streaming lifecycle. Specification shortcuts now also defer when conversation history or a board identity is supplied; those inputs must not be ignored when deciding whether information is missing. The guard examines only evidence already selected by the existing token budget. Authenticated retrieval and consent handling are unchanged.

This is bounded server guidance, not a new reasoning model, universal entity resolver or general factual verifier. Negative statements in arbitrary prose, implicit aliases, purely alphabetic part names without a hyphen, and complex questions are intentionally not covered. Source truth itself is not independently verified. The model's broader calculation, firmware, diagnostic, citation and brevity limitations remain.

## Fresh evaluation

The eight-case [V4 fixture](../tests/fixtures/research_missing_context_v4.json) was written before the first native evaluation and frozen. All eight automated and reviewed checks passed in [AIR-06_MISSING_CONTEXT_V4 summary](AIR-06_MODEL_COMPARISON.md#recorded-evaluations), with **four native answers, three evidence-guidance responses and one clarification**. No runtime errors occurred. These are not eight native-model quality passes.

- Missing frequency and voltage: ask for the correct fictional part's datasheet without copying the other part's rating.
- Current release without context: request official information and explicitly disclose no web search.
- Supplied current release note: use version 4.2 and S1 rather than discarding the note.
- Positive specification and a separate table heading: both give the supplied 16 MHz and S1.
- Malicious source command: request the missing specification without outputting the injected marker or claiming compiler execution.
- General release-version concept: ordinary two-sentence explanation; no inappropriate current-information refusal.

These cases are now regressions, not unseen data for future tuning. They use synthetic facts only; no compiler, simulator, live internet provider or production conversation was invoked.

The [seed-7 repeat](AIR-06_MODEL_COMPARISON.md#recorded-evaluations) also passes **8/8 automated and reviewed checks**, with the same four-native/four-server split. This run includes the final qualifier protection and clarified no-search wording. It is a repeated regression under another sampling seed, not another fresh holdout or a broad reliability guarantee.

## Earlier regressions and remaining defects

[V2 transfer recheck](AIR-06_MODEL_COMPARISON.md#recorded-evaluations): **5/8 automated, 4/8 reviewed**, seven native answers and one evidence-guidance response. The missing-part answer now explicitly requests the correct datasheet. The other seven answers are byte-for-byte identical to the previous baseline recheck, preserving both working behavior and known failures. The unchanged model still mishandles the LED arithmetic, uses blocking firmware code and misses the baud mismatch; its missing-LED-input answer also misses the strict helpfulness requirement. Exit 1 correctly preserves failed overall acceptance.

[V3 follow-up recheck](AIR-06_MODEL_COMPARISON.md#recorded-evaluations): **3/6 automated and reviewed**, four native answers and two server responses. Both targeted missing-specification and current-release cases now pass, as does the source-format attack case. LED arithmetic, ADC derivation/citations and the missing-power-input follow-up remain failures. No rubric was relaxed. Exit 1 is expected for the unresolved cases.

[Final core recheck](AIR-06_MODEL_COMPARISON.md#recorded-evaluations): **6/6 automated and reviewed**, two native answers, two evidence-guidance responses and two clarifications. The existing conflict, missing-unit and source-instruction checks remain correct. Exit 0; no runtime errors.

The initial V3 and core reports are preserved with `_INITIAL` suffixes. Their current-information response said “I have not searched the web,” which the frozen forbidden-substring check incorrectly treated as a search claim. The user-facing disclosure was clarified to “No web search was performed for this answer.” The frozen fixtures are unchanged. These initial failures are not silently removed or presented as native model errors.

## Reproduction

Run sequentially from `Voltforge_AI` using the existing managed runtime:

```powershell
rtk proxy .venv\Scripts\python.exe scripts\evaluate_research.py --fixture tests\fixtures\research_missing_context_v4.json --temperature 0.3 --max-tokens 768 --seed 42 --answer-policy baseline --output test-results\research\AIR-06_MISSING_CONTEXT_V4.json
rtk proxy .venv\Scripts\python.exe scripts\evaluate_research.py --fixture tests\fixtures\research_missing_context_v4.json --temperature 0.3 --max-tokens 768 --seed 7 --answer-policy baseline --output test-results\research\AIR-06_MISSING_CONTEXT_V4_SEED7.json
rtk proxy .venv\Scripts\python.exe scripts\evaluate_research.py --fixture tests\fixtures\research_quality_transfer_v2.json --temperature 0.3 --max-tokens 768 --seed 42 --answer-policy baseline --output test-results\research\AIR-06_GUARD_TRANSFER_V2.json
rtk proxy .venv\Scripts\python.exe scripts\evaluate_research.py --fixture tests\fixtures\research_quality_followup_v3.json --temperature 0.3 --max-tokens 768 --seed 42 --answer-policy baseline --output test-results\research\AIR-06_GUARD_FOLLOWUP_V3.json
rtk proxy .venv\Scripts\python.exe scripts\evaluate_research.py --fixture tests\fixtures\research_quality_eval.json --temperature 0.3 --max-tokens 768 --seed 42 --answer-policy baseline --output test-results\research\AIR-06_GUARD_CORE.json
```

Failed evaluation exit codes remain failures even if some cases improve. Raw reports record source hashes, response origin and native timing separately. Server-generated guidance is never represented as native inference or verified electrical analysis.

The first V4 and V2 reports predate the final conservative qualifier check and no-search copy change; their hashes record that exact source version. The seed-7 V4, final V3 and final core commands use the final source. Reproducing the first command now uses that final source rather than recreating historical wording.

## Automated validation

- `rtk proxy .venv\Scripts\python.exe -m unittest discover -s tests`: **114 passed**.
- `rtk proxy .venv\Scripts\python.exe -m unittest test_health_status test_security_va`: **5 passed**.
- `rtk proxy .venv\Scripts\python.exe -m compileall -q config.py engine api scripts tests`: passed.
- `rtk git diff --check`: passed.

Tests cover positive and negative guard eligibility, dimension/qualifier mismatches, aliases, conflicting mentions, inline supplied facts, no automatic search, no unnecessary native call, source/attempt/complete events and preservation of history and every supported project-context category. Existing streaming, cancellation, partial failure, auth and retrieval tests continue to pass. Existing Starlette/httpx deprecation and unclosed `nul` resource warnings remain. Subsequent cross-app checks and the expanded 117-test AI suite are recorded separately in [AIR-07_VERIFICATION.md](AIR-07_VERIFICATION.md); live provider-to-browser or shared-database acceptance is not claimed.

This bounded follow-up is implemented and verified. Next AIR-06 work is the remaining engineering-answer quality and deployable model/profile decision, followed by cross-service acceptance. Live public search still requires provider configuration and the independent AIR-02/03 gates. Payments and responsive-layout development remain after canonical RAG acceptance.

## Scope

Source changes are confined to `engine/evidence_guard.py` and `engine/llm.py`; regression coverage is in `tests/test_evidence_guard.py` and `tests/test_research_streaming.py`, plus the V4 fixture and evaluation reports. No UI/BL source, API schema, database schema/data, model artifact or deployment configuration changes were required. Existing source events, attempt IDs, completion modes and persistence transport remain intact. No new model download or experimental-policy activation.
