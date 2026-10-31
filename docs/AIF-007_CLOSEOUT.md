# AIF-007 closeout — 2026-10-27

**Complete for the authorized local/document-only development scope.** This
closes telemetry implementation and its acceptance checks, not PLAT-000 or any
production release gate. Next item: **PERF-001**.

## Acceptance evidence

| Requirement | Evidence |
| --- | --- |
| Attribute latency without private content or metric labels | Browser delivery, BL admission/retrieval/turn preparation/checkpoint, native queue/first-text/completion and native prompt-evaluation counters; combined artifact below |
| Bounded long-running metrics | Fixed stage/event/outcome labels, 256 timing samples per window, 256 recent stream outcomes, five-minute fault/rate expiry; concurrent Python recording, 1,000-operation/label and duplicate-terminal tests |
| Real tokens distinguished from status | Empty token payloads, keepalives and status do not set first-token timing; legacy/canonical events and SSE errors verified |
| Authorized diagnostics, no reasoning | Administrator-only health details/components and metrics; anonymous/ordinary/admin HTTP matrix; content-free aggregates, no transcript/hidden-reasoning export |
| Sanitized end-to-end diagnostic | Five Chromium production-transport/contract/timing-module → Spring controller/store/gateway → FastAPI → actual GGUF model → saved/reopen turns |
| Fault activation and recovery | Real H2 constraint failure rolls back checkpoint with no saved success; persistence signal activates; remove constraint and same turn saves. Known model/search errors, capacity rejection, rate/latency conditions and expiry tested |

Operational DEC-07 is resolved by the user's **Use this local aggregate policy**
answer. No user/project identifiers, prompts, code, keys or output text are stored
in diagnostic aggregates. No external exporter, persistent trace store or global
browser debugging API was added. Restart clears server aggregates; reload clears
browser aggregates. Billing retention remains open under DEC-07 for PAY-002.

## Combined measured example

[Sanitized report](AIF-007_DIAGNOSTIC_SAMPLE.json) is the final 2026-10-27 run.
It contains counts/timings/profile only, not request IDs or message content.

- This PC; Qwen2.5-Coder-1.5B-Instruct Q4_K_M, four threads, context 4096,
  output limit 768, one active native request and zero queued requests.
- Five sequential repetitions of one short general electronics question: one
  cold model prompt and four warm/cache-reusing prompts. All five saved and reopened.
- Browser first-text median **132.3 ms**, maximum **15,783.8 ms**.
- Browser saved-confirmation median **955.2 ms**, maximum **16,609.5 ms**.
- Native prompt evaluation maximum **15,601.519 ms**; warm prompt-counter
  increments were zero with prefix reuse. Zero does not imply no generation work.
- Native queue wait maximum **0.0335 ms**. The cold delay is predominantly native
  prompt evaluation, rather than queue saturation or transcript persistence.
- Observed gateway failure rate **0/5**, all signals inactive on the successful run.

This is a small local diagnostic workload, not a performance/SLO qualification.
The existing window percentiles use the lower indexed order statistic
`floor((n-1)*p)`; at n=5 the reported p95 omits the maximum. **Use the maximum and
sample count for the cold outlier**, not that small-window p95 as a tail guarantee.
Broader workloads, cache variability and render/paint profiles belong to PERF-001.

Identity decoding and H2 are controlled test fixtures. The browser runs the actual
transport/contract/timing modules in an isolated page, not a new full-product
real-account sign-in test. The existing 16 product-browser groups cover panel
wiring, stop, failed/missing-save behavior and timing ordering. Embeddings and web
search are disabled in this probe; existing document tests cover that boundary.
No live provider, real-issuer, production database, physical-device or production
capacity acceptance is inferred. Runtime defaults were not changed by the probe.

## Prefill correction

Earlier notes incorrectly stated that the installed binding did not expose pure
prefill. Local source inspection found `llama_cpp.llama_get_timings` in installed
0.2.90. `native_prefill` now measures the delta of cumulative `t_p_eval_ms` while
the native owner lock is held, with verbose mode off. It excludes queue wait,
tokenization and token sampling. No timing reset or model-state mutation is used.
Unavailable or decreasing/reset counters produce no fabricated sample. Five real
native samples and a deterministic delta/reset test verify this adapter.

The native stages now have four fixed labels and at most 24 stage/outcome
combinations. Each model invocation is a sample, including internal assessment
calls; these are not user-visible token counts or billable usage.

## Alert policy and measured rationale

All signals are administrator-readable local diagnostic warnings. They trigger
no automatic restart, retry, billing action or external notification.

| Signal | Condition | Basis / interpretation |
| --- | --- | --- |
| admissionSaturation | User/project capacity rejection in last 300 s | Excludes authentication/unknown codes; existing single-active overload qualification |
| modelUnavailable / searchUnavailable | Fixed known error codes in last 300 s | Known unavailable states, not inferred from message text |
| persistenceFailure | Checkpoint transaction throws, within last 300 s | Actual H2 rollback/recovery test plus stage exception propagation |
| streamErrorRate | At least 5 eligible completed streams, failure/timeout fraction >=20%, last 300 s/latest 256 | Local measured baseline 0/5; warning detects one failure among five. This is an explicit sensitivity choice, not statistically established field reliability |
| slowStream | Gateway stream duration >=60,000 ms in last 300 s | More than twice the prior six-workload 26.422 s maximum, rounded up; current 16.61 s cold path remains below it |

Actual user cancellation/rejection is excluded from the stream failure-rate
denominator. Cancellation caused by consuming an already observed provider
complete/error retains that provider outcome. Duplicate finish cannot add samples.
The slow-stream signal evaluates when the stream finishes; existing stream timeout
handling remains responsible for in-flight stalls. These local-profile defaults
must be reassessed for different hardware/topology and workloads in PERF-001.

## Verification and reproduction

- **53 backend tests passed**, no failures/errors/skips: telemetry, admission,
  gateway, conversation and the explicitly enabled real-native integration test.
- **167 AI tests passed**, including native prefill adapter, concurrency, buffer
  bounds and existing streaming/consent regressions.
- UI timing/SSE runner: **191 chunk-boundary cases plus timing checks passed**.
  UI lint: no errors, 13 existing warnings. Previous product browser **16 groups**
  and build/bundle checks remain valid; no product UI changed in this closeout.
- All three repositories pass whitespace checks. Known existing test warnings
  (including the Python `nul` resource warning) do not indicate passed live gates.

From Voltforge_BL (sibling UI and AI repositories and installed model required):

```powershell
rtk proxy .\mvnw.cmd "-Dtest=AiTelemetryTest,AiRequestAdmissionTest,AiServiceImplTest,ConversationIntegrationTest,TelemetryNativeIntegrationTest" "-Dvoltforge.telemetry.native=true" test
```

The native integration is opt-in: it starts loopback-only services on selected
ports, uses a fresh service token, enforces the exact test browser origin, and
stops its processes. `scripts/telemetry_probe_server.py` exposes an authenticated
aggregate endpoint **only in this test host**; production `main:app` never imports
it. The Node probe opens an isolated page and does not invoke real-issuer sign-in.
Reports/logs are under BL `target/aif-007-*`; only the sanitized JSON is copied
into documentation. The first combined attempt failed at test-origin admission;
the exact test CORS origin was configured, not bypassed in production.

For recovery, stop the probe processes or restart the affected application to
clear aggregate memory. Reverting instrumentation must retain the existing
administrator access restrictions and must not change saved-message authority,
native admission, consent or error propagation.
