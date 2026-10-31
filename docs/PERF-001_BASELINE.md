# PERF-001 local baseline - 2026-10-27

**Status: complete for the authorized local/document-only baseline scope.** It
does not qualify the configured model for release, the RAG release gate, a real
identity session, live web research, or production capacity.

## Profile and method

| Field | Value |
| --- | --- |
| Host | Windows 11, Intel Core i5-8265U, four cores / eight logical processors, 23.8 GiB RAM |
| AI admission | One active native request, zero queued requests; bounded rejection on overload |
| Model/configuration baseline | Qwen2.5-Coder-1.5B-Instruct Q4_K_M; four threads; 4,096 context tokens; 768 output-token limit |
| Browser | Installed headless Chrome, 1366 x 768 unless the artifact says otherwise |
| Scope | Local/document-only development; synthetic identity/H2 fixtures where noted |

The model row identifies the measured configuration only. `DEC-02` remains open
for PLAT-000 release-model selection. Cold and warm measurements are kept
separate, and small samples report their maximum alongside percentile values.

## Captured evidence

| Workload | Samples and result | What it measures | Limits |
| --- | --- | --- | --- |
| Chat, end-to-end local stack | Five sequential requests: browser first-text p50 132.3 ms, reported p95 134.9 ms, max 15,783.8 ms; saved confirmation p50 955.2 ms, max 16,609.5 ms | Browser transport, Spring admission/retrieval/persistence, FastAPI and actual local GGUF generation | One cold then four warm short general-electronics prompts; synthetic identity/H2; no live search. At n=5 the p95 excludes the maximum. |
| Native stages in that chat run | Queue-wait max 0.0335 ms; native first-text p50 78.5 ms, max 15,617.6 ms; completion p50 890.0 ms, max 16,444.3 ms; prefill max 15,601.5 ms | Separates admission, document retrieval, native prefill and generation from UI timing | Prefix reuse makes warm prefill deltas zero; this is not a long-chat or capacity result. |
| Fixture chat delta to visible DOM | Five fixture streams x 32 chunks x 256 bytes: latest p50 48.0 ms, p95 48.7 ms, max 66.2 ms across 160 chunks; prior independent run p95 62.7 ms, max 74.0 ms | SSE token delivery to visible assistant DOM content, followed by two animation frames | Fixture-only; includes test-controller scheduling and excludes network, server, retrieval, model, production identity, pixel paint and hardware-input latency. |
| Project-list data boundary | `BackendRuntimeCompatibilityTest` passed with six records whose canvas and component fields are each 256 KiB; paging, ownership and search return summaries without hydrating those document fields | Server-side list/paging boundary used by the UI's 24-summary project page | An integrity/query-shape test, not route timing. Browser fixture route timing remains deferred. |
| AI admission | Targeted queue rejection, timeout, cancellation and slot-recovery test passed; bounded aggregate/redaction test passed | Local one-active/zero-queued profile and bounded overload behavior | The interpreter used for this check lacks `llama_cpp`, so native-token tests were not used as acceptance evidence. |
| Document retrieval | 20-document corpus: load 0.609 s, index 0.469 s, query p95 69.0 ms; 32-query batch 8.985 s; peak process memory 88.31 MiB | Local lexical/semantic/hybrid retrieval work | Fixed benchmark corpus and local process, not an authorized project corpus or web search. |
| Public route usability | One configured-endpoint homepage sample: heading visible 1,011 ms, LCP 1,940 ms, one 54 ms long task | Production-preview browser route bootstrap | A single lab sample with no real account sign-in; not field Web Vitals. |
| Editor store-to-frame baseline | 60 runtime updates: p50 19.3 ms, p95 28.6 ms; 60 clock/AVR/diagnostic/viewport/PCB updates: p95 17.4-17.7 ms | React commit counters and two-animation-frame scheduling in the actual editor components | Dev + StrictMode, 25-node fixture, no solver load; scheduling proxy, not paint/input latency. |
| 512-pad ratline routing | Cold p95 19.0 ms; warm p95 33.2 ms; warm max 54.0 ms; median immediate heap delta 7.1 MiB | Deterministic routing algorithm on the declared host | Algorithm-only: no browser rendering, simulation, server work or controlled host load. |
| Bundle budget | Entry 82.1/150 kB, initial static JS 495.5/550 kB, 12/16 chunks, largest component factory 289/450 kB | Production build size guard | No route-by-route transfer trace. |
| AVR execution | Existing deadline, ordering, lifecycle, browser timing, telemetry and fidelity contract passed | Simulation scheduling contract | No physical-device or full editor interaction measurement. |

Artifacts are retained in the working tree at
`AIF-007_DIAGNOSTIC_SAMPLE.json`, `test-results/retrieval/latest.json`,
`../../Voltforge_UI/test-results/auth-availability/vfopt-ui-001-auth-availability.json`,
`../../Voltforge_UI/docs/reports/perf-001-chat-delta.json`,
`../../Voltforge_UI/docs/reports/perf-001-render-before.json`, and
`../../Voltforge_UI/docs/reports/vfopt-ui-026-ratline-before.json`.

## Baseline findings and ranked candidates

1. **P0 - editor runtime subscription isolation.** The 60-update editor probe
   committed `CircuitEditorPage` 60 times for runtime feedback. Its current p95
   is 28.6 ms, but the page-wide work scales with editor complexity. The same
   full after-suite also failed its runtime-isolation assertion. First inspect
   the page's runtime-store subscription and split/select the rendered state;
   retain the existing editor correctness and render-isolation checks.
2. **P0 - simulation lifecycle regression.** In the full after-suite, the
   simulation start/pause/step/resume/stop check and the unmount cleanup check
   timed out. This is a correctness/lifecycle blocker found while measuring,
   rather than a claim about frame speed. Reproduce and repair it before using
   the full editor suite as an optimization acceptance gate.
3. **P1 - cold native prompt evaluation.** The cold prefill maximum of 15.6 s
   dominates the cold end-to-end outlier while queueing and persistence remain
   small. Compare fixed prompts, context sizes and candidate models after
   `DEC-02`; do not conceal it with warm-prefix reuse or change model defaults
   during PERF-001.
4. **P1 - large-board routing tail.** A 512-pad warm run reached 54.0 ms, over
   one 60 Hz frame. Profile allocation and candidate pruning on realistic board
   geometries before changing routing semantics. The existing browser/GPU
   follow-up evidence remains required.
5. **P2 - route and chat breadth.** The fixture chat delta probe is below the
   proposed 100 ms interaction guideline, and the bundle guard passes. There
   is still no repeated authenticated projects route, large project list, long
   Markdown response, networked stream, or pixel-paint trace. Add these traces
   before setting an application UI budget. The project-list server boundary is
   protected by page-size and summary-only compatibility coverage, but it is
   not a browser route-performance measurement.

## Follow-up after PERF-001

- Repeat declared cold/warm route runs and collect memory, long-task and
  network/route-bundle evidence for public, authenticated projects and editor
  workloads. A real issuer is not available, so authenticated measurements must
  remain fixture-labelled until that gate is available.
- Measure long chat history, long Markdown/code responses, networked
  delta-to-paint, representative project documents, large project lists and
  bounded overload/recovery through the actual BL/AI path.
- Resolve `DEC-02` with quality, latency and capacity evidence for the release
  model in PLAT-000. Do not convert this configuration baseline into a model
  decision.
- Implement OPT-FB-012, then re-run the full editor browser suite. Preserve
  the current artifacts as the before state.

The scoped PERF closeout records reproducible local evidence, explicit limits,
and ranked follow-up. Release-specific budgets remain intentionally deferred to
the accepted model and production profile.

## Reproduction commands

```powershell
# Voltforge_AI
rtk proxy .venv\Scripts\python.exe scripts\benchmark_retrieval.py

# Voltforge_UI
rtk npm run build
rtk npm run test:auth-availability
rtk npm run test:chat-perf
rtk npm run test:editor-render -- --before --report-prefix=perf-001
rtk npm run test:ratline-benchmark -- --before
rtk npm run test:avr-budget
rtk python -m pytest tests/test_telemetry.py::TelemetryTests::test_queue_rejection_timeout_cancellation_and_slot_recovery tests/test_telemetry.py::TelemetryTests::test_bounded_thread_safe_aggregates_and_label_redaction -q

# Voltforge_BL (uses the installed Maven distribution when the wrapper is unavailable)
rtk powershell -NoProfile -Command "& 'C:\Users\vsiva\.m2\wrapper\dists\apache-maven-3.9.15\0226a00282e400185496f3b60ec5a3f029cbdc6893912937d4876d57695224e1\bin\mvn.cmd' -q '-Dtest=BackendRuntimeCompatibilityTest' test"
```
