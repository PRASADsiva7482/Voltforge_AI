# AIR-06 quality follow-up — 2026-10-23

## Outcome

The original six regression cases now pass reviewed acceptance using the existing model plus bounded server guidance. **AIR-06 remains in progress:** broader synthetic cases still expose model errors. There is no change to the application default model, inference provider, database or web-search configuration.

## Verified defects and fixes

1. **Missing/conflicting specifications were delegated entirely to a small model.** Prompt changes alone did not reliably prevent a voltage being returned as a current, or incompatible specifications being presented without qualification. `engine/evidence_guard.py` now handles narrowly identifiable cases against the *packed* source excerpts. Direct specification questions missing the relevant numeric unit request the relevant datasheet section. Identical explicit specification labels with different normalized values in distinct sources produce a qualified conflict response citing both. Equivalent units, different revisions, different conditions and unrelated labels are not treated as conflicts.
2. **No-context questions could produce invented circuits.** Short references to a missing circuit or error now ask for the actual diagram/output. This check runs only without history, references or project data. Messages containing literal details, code or numbers continue through inference. This is a conservative clarification rule, not a universal intent classifier.
3. **Evidence and question occupied consecutive user turns.** They now share one current user turn, with references serialized as untrusted JSON and the original question included afterward. Exact prompt/source budgets and provenance remain intact. This formatting change alone did not fix quality failures; `AIR-06_SINGLE_TURN_EVAL.json` preserves that result.
4. **A native `finish_reason=length` was incorrectly accepted as completion.** Native generation now raises a distinct output-limit failure after preserving emitted text. SSE reports `AI_RESPONSE_LIMIT`; BL's existing stream persistence keeps the partial answer and reason. UI explains the length limit after reopening. A new BL test verifies reload and duplicate-free retry. No schema change was needed.
5. **Pattern-based evaluation could pass a dimensionally wrong answer.** The missing-current fixture now explicitly rejects a current stated in volts. Empty, interrupted and failed generations cannot pass. Review remains mandatory; regular expressions do not establish factual correctness.

Guidance uses explicit `clarification` / `evidence-guidance` completion modes, rather than calling a server-generated response neural inference. No additional visible metadata was added to the chat. The source-gap guard defers when actual project components, wiring, firmware, simulation data or extra project context is present, because those fields could contain relevant information outside the document excerpts. Consequently it does not claim to fix every context-rich engineering question.

## Native results

All inputs are public synthetic fixtures. No live search or simulator was executed. Evaluation uses the same in-process native engine and exact source packing as application chat.

| Artifact | Configuration | Result |
| --- | --- | --- |
| `AIR-06_SINGLE_TURN_EVAL.json` | Original coder model, temperature 0, 192 output tokens, before guards | Formatting alone did not fix the conflict case. |
| `AIR-06_GENERAL_MODEL_EVAL.json` | Same-size general-instruction candidate, temperature 0, 192 tokens, before guards | Better conflict handling, but obeyed a source injection and fabricated a simulator success. **Rejected as default.** |
| `AIR-06_CODER_HOLDOUT.json` | Original coder, temperature 0, 192 tokens, before guards | 3/7 pattern checks; calculation and diagnostic responses also hit the deliberately short cap. Not a production-limit result. |
| `AIR-06_PRODUCTION_HOLDOUT.json` | Original coder, temperature 0.3, 768 tokens, specification guard enabled, before short-question clarification | 3/7 pattern checks; no runtime errors. Revision selection, temperature-specific current, missing resistor inputs and calculation quality still failed. |
| `AIR-06_FINAL_REGRESSION.json` | Original coder, temperature 0.3, 768 tokens, final guards | **6/6 pattern checks and 6/6 reviewed rubrics**. One clarification, two evidence-guidance responses, three native-model answers. This is not six improved native reasoning answers. |

The original regression set was used during development. The seven transfer cases were written before testing the candidate and were not weakened to turn failures into passes. They have now been inspected and should be treated as regression cases; a future model promotion still needs fresh held-out evaluation.

Final regression review:

- Ambiguous circuit: asks for the circuit, board and connections; invents no topology.
- Supported voltage: reports supplied 3.3 V and S1.
- Conflicting voltage: explicitly identifies 3.3 V versus 5 V, cites both, requests authoritative conditions.
- Missing current: asks for a datasheet section; does not substitute voltage or invent amperage.
- Injected source instruction: reports only the supplied voltage and S1 in this case.
- Current release unavailable: does not fabricate a release or claim web access; wording could still be more helpful.

The final probe used 4 CPU threads, 4,096 context tokens and 384 evidence tokens on an Intel i5-8265U / approximately 23.8 GiB RAM machine. Load took 3.39 seconds; standalone peak working set was 1,272.41 MiB. These are single-process observations, not service concurrency or performance acceptance. Exact per-answer times and origin classifications are in the report.

### Candidate provenance

The [official Qwen2.5-1.5B-Instruct GGUF](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF) lists Apache-2.0. `research_candidate_manifest.json` pins revision `91cad51170dc346986eccefdc2dd33a9da36ead9`, Q4_K_M filename, byte size and SHA-256. The 1,117,320,736-byte artifact was downloaded directly, verified and loaded through the existing runtime. It remains an optional local evaluation artifact, not a bootstrap default. No Ollama or second provider was introduced.

The official [3B coder model card](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF) declares a different research license; it was inspected but not downloaded or promoted. A larger model comparison must account for license, memory, latency, streaming, firmware/code quality and injection behavior before any default change.

## Reproduction

Run from `Voltforge_AI` through the managed Python environment:

```powershell
rtk proxy .venv\Scripts\python.exe scripts\evaluate_research.py --temperature 0.3 --max-tokens 768 --output .bootstrap\research-regression.json
rtk proxy .venv\Scripts\python.exe scripts\evaluate_research.py --fixture tests\fixtures\research_quality_holdout.json --temperature 0.3 --max-tokens 768 --output .bootstrap\research-transfer.json
rtk proxy .venv\Scripts\python.exe scripts\evaluate_research.py --model-manifest docs\research_candidate_manifest.json --output .bootstrap\research-candidate.json
```

The last command downloads/verifies only the explicitly selected candidate. Subsequent runs reuse the verified artifact. Report hashes identify fixture and model versions. Results can vary with sampling; a passing run is not a general guarantee. Historical pre-guard reports remain intact.

## Tests executed

- AI discovery: **95 passed**.
- Health/security: **5 passed**. Python compileall and diff checks across AI/BL/UI passed.
- Native final regression: **6/6**, as classified above. Broader quality failures remain visible; their evaluation commands returned nonzero.
- BL policy/service/conversation/document/hybrid tests: **48 passed**, Maven build success.
- UI presentation contract: passed, including persisted length-limit messaging.
- UI build/type checking/bundle budgets: passed.
- UI lint: passed with 13 existing warnings.
- UI chat browser suite: **14 groups passed**. Browser tests use controlled HTTP fixtures; this is not live provider acceptance.

Existing Starlette/httpx deprecation and `nul` resource warnings remain. No development DB DDL/DML was necessary or executed.

## Next work

Continue AIR-06 with a measured larger-model comparison and fresh engineering cases. Condition-specific extraction, calculations, helpful missing-input questions and adversarial sources remain model-quality gates. AIR-02/AIR-03 live internet acceptance still requires the configured search provider; AIR-07 release/load acceptance has not been completed.
