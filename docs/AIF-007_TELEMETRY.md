# AIF-007 telemetry implementation record

Status: **complete**, 2026-10-27. See [authoritative closeout](AIF-007_CLOSEOUT.md) and [measured diagnostic artifact](AIF-007_DIAGNOSTIC_SAMPLE.json). The increment history below is retained; the closeout supersedes its earlier limitations and pending verification notes. Local/document-only development authorization
applies; this is not production or live-provider acceptance.

## Implemented increment

The existing Spring `AiTelemetry` now recognizes the actual `token` event as well
as legacy `delta`. First-token latency measures arrival of the first nonzero-byte
token/delta event at the gateway, relative to gateway observation start. It is
not a tokenizer count, native prefill time, browser paint time or billing usage.
Status, keepalive, source and terminal events do not set that timestamp.

Event labels remain a fixed allowlist. Admission labels now use the exact three
codes emitted by `AiRequestAdmission`, with all other strings collapsed to
`other`. The previous prefix regex allowed arbitrary labels and could retain
identifiers embedded in otherwise valid-looking error codes. Unknown/null
outcomes also collapse to `other`. Existing timing windows retain at most 256
samples; repeated completion cannot add another sample or release twice.

No new storage, exporter, endpoint or correlation identifier has been introduced.
Existing process-local counters last until restart; latency windows evict by
sample count, not time.

### Approved operational policy — 2026-10-27

The user selected **Use this local aggregate policy**: process-local aggregates,
latest 256 timing samples, no transcripts or user/project IDs, no external export,
and administrator-only diagnostic details. DEC-07 is resolved for AIF-007 only;
its billing-record/event/usage retention scope remains open for PAY-002.

Restart clears counters and timing windows; eviction removes older individual
timing samples. There is no per-request trace store or user-indexed diagnostic
record to delete. Do not add identifiers or cross-process trace storage to obtain
correlation. Compare aggregate stage windows with their documented boundaries.
Persisted conversations and billing data are outside this operational policy.

Spring health details and components now require `ADMIN`. The public root health
endpoint still returns overall status. Component health and metrics endpoints
require the administrator role through the existing JWT role conversion. Native
Python aggregates remain internal with no newly exposed HTTP access. Configuration
overrides that loosen these settings are outside the approved policy.

The existing `admissionSaturation` signal now means a user/project capacity
rejection occurred within the last 300 seconds. Authentication and unknown-code
rejections do not trigger it. This is a bounded recent-event signal, not current
queue occupancy, an error-rate threshold, a measured SLO or an external alert.
Expiry requires no background task, extra event buffer or persistent storage.

### Backend stage measurements

`snapshot.stages` contains fixed-enum `latencyMs` windows and success/failure
counters. Each window retains at most 256 samples; there are four stage labels
and at most eight outcome labels. No supplier input, output or exception text is
retained. Exceptions propagate unchanged after recording a failed measurement.

| Stage | Actual interval | Interpretation |
| --- | --- | --- |
| `admission` | `AiRequestAdmission.acquireStreaming` call | Nonblocking BL admission, including rejection; not the native queue wait |
| `document_retrieval` | Authorized `DocumentService.retrieve` call during `ConversationStore.begin` | Includes authorization/retrieval work and no-document fast paths; not public web search |
| `turn_preparation` | Proxied `ConversationStore.begin` call from the HTTP controller | Includes initial transactional commit, authorization, retrieval and memory work; replay also enters this boundary. Not database-only timing. |
| `persistence` | Proxied `ConversationStore.checkpoint` call from `ConversationStream` | Includes transaction commit for periodic and terminal checkpoints; excludes initial turn creation |

Gateway first-token and stream-total windows overlap these intervals. They cannot
be summed into request latency. Document retrieval precedes gateway observation;
terminal persistence may occur after its cancellation/completion. A replay skips
retrieval and checkpointing, so it does not manufacture additional stage samples.
Native and browser delivery intervals are measured separately below. Browser
paint is not inferred from event-delivery timing.

### Browser delivery and fault signals (2026-10-27)

`chatTiming.ts` records page-local aggregate durations from just before the chat
transport call to validated `accepted`, first nonempty text, and authoritative
`saved` events. Total duration closes on saved confirmation or request cleanup.
It includes authentication refresh, transport, server work and response parsing;
it excludes initial conversation creation, pre-send validation and browser paint.
No render-performance claim is made from these event-handler measurements.

Four windows retain at most 256 durations; five fixed outcome counters distinguish
completed, failed, interrupted, cancelled and unconfirmed. Complete/EOF without
saved cannot record success. Reload destroys browser aggregates. No localStorage,
network export, global debugging hook or user-facing diagnostic view is added.
The source module snapshot is used by development tests, not served as an
application diagnostic endpoint. Server diagnostics remain administrator-only.

The gateway now recognizes default-event JSON types, excludes empty token text
from first-token timing, and classifies an SSE error as a failed stream even when
the HTTP connection finishes normally. Fixed known codes activate recent
`modelUnavailable` and `searchUnavailable` signals. Checkpoint exceptions activate
`persistenceFailure` while preserving the original exception. Unknown codes are
ignored, and each of these signals expires after 300 seconds. These are recent
event indicators, not current availability, measured error rates or external
alert delivery. Initial preparation failures are not automatically classified as
database failures because that interval also includes authorization/retrieval.

Sanitized controlled browser example: the live HTTP fixture produces one sample
each for accepted, firstText, saved and total, with outcome completed=1. Assertions
verify accepted <= firstText <= saved and no response text in the snapshot.
The same suite verifies cancelled=1 after Stop, and unconfirmed=1 with zero saved
samples for each malformed/missing-save case. These are browser-to-controlled-API
results; they are not a simultaneous real-account/native-model full-stack trace.

### Native runtime measurements (2026-10-27)

`LocalRuntime.telemetry.snapshot()` now provides internal, process-local fixed-enum
aggregates. No new API, health payload, log exporter or diagnostic access path is
exposed. The three timing windows retain the latest 256 samples each; outcomes
have at most 18 stage/outcome combinations. Snapshot copies are isolated from
internal state and concurrent recording is protected by a lock. Nonfinite values
and arbitrary labels are discarded, and durations are clamped to 0–600,000 ms.

| Stage | Actual interval and caveats |
| --- | --- |
| `queue_wait` | Entry into native admission through lock acquisition or admission failure. Rejection, cancellation and queue timeout are separate outcomes. Excludes generation and downstream token delivery. Applies to sync and streamed calls. |
| `native_first_text` | Start of a native completion invocation through its first nonempty text chunk. Includes prompt tokenization, prompt processing and first-token sampling; **not pure prefill**. Empty completions do not create a first-text sample. |
| `native_completion` | Same invocation through iterator closure. Includes downstream consumption/backpressure while the generator is suspended. Success, cancellation, timeout, output truncation and failure are distinct outcomes. |

Each model call is a sample, including internal evidence-assessment calls. These
are not necessarily user-visible tokens, one sample per chat, or billable usage.
Deterministic clarification paths that do not call the model produce no native
completion sample. The current binding provides no isolated prefill timing hook;
that decomposition remains unavailable and must not be inferred by subtraction.
No per-request correlation or cross-process total can be reconstructed from these
aggregates. Native runtime ownership, admission limits and cancellation behavior
are unchanged.

## Verification

Delivery/fault increment, 2026-10-27: **48 targeted backend tests passed**, zero
failures/errors/skips. Covers initial preparation through HTTP, research fault
signaling, bounded/expiring fault state, unchanged exception propagation, legacy
event classification and SSE-error failure outcome. UI timing tests and 191 SSE
split-boundary cases passed; all **16 browser groups** passed with new timing,
cancellation and missing-save assertions. TypeScript/build/bundle budget passed
(495.5 kB initial static JS against 550 kB); lint has 13 existing warnings and no
errors. Whitespace checks passed in all three repositories.

Approved-policy increment, 2026-10-27: the four targeted Maven suites passed
**46 tests**, zero failures/errors/skips. HTTP checks cover root health,
component health and metrics for anonymous, ordinary JWT and administrator JWT
identities, including absence of telemetry details for non-administrators.
Controlled JWT decoding and H2 are used; this is not live-issuer acceptance.
A deterministic-clock test verifies alert activation, exact expiry, renewed
capacity rejection, and exclusion of authentication/unknown rejection codes.
BL/AI whitespace checks and scoped DEC-07/backlog gate validation passed.

2026-10-27: the four targeted Maven suites passed **44 tests**, with zero
failures, errors or skips. This includes the AIF-002 gateway/conversation
regressions. Whitespace checks passed in UI, BL and AI repositories.

Run from `Voltforge_BL`:

```powershell
rtk proxy .\mvnw.cmd "-Dtest=AiTelemetryTest,AiRequestAdmissionTest,AiServiceImplTest,ConversationIntegrationTest" test
```

The new regression cases cover actual token and legacy delta timing, status-only
streams, zero-byte events, duplicate finish, post-finish events, 1,000 distinct
untrusted labels (including valid-looking `AI_PRIVATE_USER_*` strings), null
outcomes and 1,000 requests exceeding the 256-sample window.
Stage tests additionally check bounded samples and unchanged result/exception
propagation without private values. The conversation integration case checks real
retrieval/checkpoint instrumentation, saved replay without extra samples, and
absence of user, conversation, attempt and transcript values in the snapshot.

Native increment: `rtk proxy .venv\Scripts\python.exe -m unittest discover -s tests`
passed **166 tests** on 2026-10-27. New tests cover concurrent bounded recording,
snapshot isolation, label/content redaction, rejection/timeout/cancellation and
slot recovery, empty/nonempty completion timing, generator closure, truncation,
and propagation of original failures. Controlled models exercise the real wrapper;
this is not a new native-model capacity or live-provider qualification. An initial
test used a queue timeout below the validated minimum; corrected to 0.1 seconds.
The full suite retains an existing unclosed `nul` resource warning. Whitespace
checks passed.

## Closeout

All item acceptance checks are closed within the approved development scope; see [AIF-007_CLOSEOUT.md](AIF-007_CLOSEOUT.md). Native prefill is now measured using the installed native timing counter. Five real-native cross-app turns, transactional rollback/recovery and measured local warning thresholds are verified. Real-account, live-search and production gates remain unchanged. Next: PERF-001.
