# AIR-06/07 local closeout — 2026-10-26

Development sequencing update (2026-10-26): the user explicitly approved starting platform work against the local/document-only baseline with live search, tunnel-free identity and production release excluded. AIR statuses remain unchanged. See [the current platform sequence](PLATFORM_DELIVERY_PLAN.md) and [completed AIF-001 mapping](AIF-001_FLOW_AND_OWNERSHIP.md). This supersedes prior statements that all platform development must wait; it does not close any RAG release gate.

The known V6 within-source contradiction is corrected, and the user-selected local
one-active-request capacity profile passes. Full web-RAG release acceptance remains
open: live search access and real tunnel-free session restoration are unavailable.
This record does not promote either AIR item or waive PLAT-000.

## Engineering result

The existing timing-capability boundary now vetoes explicit negative and conditional
`millis` claims before native assessment. The parser repeats the check so an affirmative
model label cannot override that source text. It returns an uncertain, cited
clarification, not a claim that the hardware is unsupported. Unrelated negative
statements about compilation or another API do not trigger the veto.

This is a bounded English-language limitation check, not general semantic proof.
Unrecognized paraphrases, qualifications across sentences and arbitrary capabilities
still rely on the existing model. References comparing revisions can conservatively
require clarification even when a human could resolve them. No model, catalog,
provider, prompt/schema variant, database or service dependency was added.

| Frozen suite | Pattern and manual rubric result | Scope |
| --- | --- | --- |
| V2 | 8/8 | Existing engineering transfer |
| V3 | 6/6 | Missing inputs, arithmetic and source injection |
| V5 | 6/6 | Four native sketches, two cited clarifications; pin/history preserved |
| V6 | 4/4 | All four failures/negative cases clarify without code or native inference |
| New V7, seeds 42 and 7 | 7/7 on each | Five limitations plus two valid registered-Uno/shared-code controls |

Settings: current baseline model, temperature 0.3, 768 combined output tokens,
seed 42 except the stated seed-7 repeat. Valid V7 outputs retain pin 6 and 825 ms,
and label themselves uncompiled. New output was reviewed, but compiler/emulator,
physical hardware and authenticated browser persistence were not rerun in this
increment. Previous compile/browser records remain historical evidence.

The reusable V7 fixture is `tests/fixtures/research_firmware_limits_v7.json`.
Full discovery: **161 tests passed**; health/security: **5 passed**. Existing
Starlette/httpx deprecation and unclosed `nul` warnings remain.

## Local capacity result

User decision: qualify this PC for **one active AI request with bounded overload
rejection**. Tested profile: one process, zero waiting requests, four threads,
4096 context tokens, configured 768-token response limit, 120-second generation
deadline. The verifier uses a private temporary token and loopback port; it does
not modify normal service configuration. To deploy this exact profile, set
`VOLTFORGE_AI_MAX_PENDING_REQUESTS=0`; the application default remains two waiting
requests, which this run does not qualify.

Hardware: Intel Core i5-8265U, four cores/eight logical processors, 25,569,435,648
bytes installed RAM. Model SHA256:
`cc324af070c2ecbfd324a30884d2f951a7ff756aba85cb811a6ec436933bb046`.

Before running, the local probe used these limits: readiness within 90 seconds,
response within the configured generation deadline plus five seconds, overload
rejection within two seconds, health response within one second, and summed
process-tree peak working sets within 4096 MiB. These are development acceptance
limits, not a production SLA or a claim that 26-second answers meet product UX goals.

| Measurement | Observed |
| --- | ---: |
| Fresh service readiness | 4.500 s |
| First nonempty output, initial request | 14.016 s |
| Competing request rejected | 0.063 s |
| Maximum of five health requests during generation | 0.016 s |
| Completed response after disconnect | 1.940 s |
| Three general-chat response times | 2.438 / 17.093 / 15.906 s |
| Three document-backed firmware response times | 25.656 / 26.422 / 22.484 s |
| Document first-output times | 18.718 / 19.094 / 16.859 s |
| Maximum summed process-tree peak working sets | 1346.86 MiB |

Authentication, body limit, busy rejection, native incremental output, disconnect
recovery, six sequential completed workloads and memory limits passed. Firmware
workloads must actually emit code containing the required timing API/interval;
a clarification response cannot count as a successful capacity workload.
General-chat timing varies with the preceding prompt/cache state. No percentile
or concurrency extrapolation is justified from six samples. This is an isolated AI
HTTP result, not BL/UI throughput, long-chat/large-document, multi-process, sustained
thermal-soak or physical-device acceptance. Disconnect-to-completed-response time
is not a measurement of exact cancellation latency.

Two harness failures were corrected before the accepted run: the synthetic
document payload lacked its required attempt ID; initial memory sampling measured
only the managed Python launcher. Memory now includes its actual native child.
Regression coverage prevents smoke checks from passing capacity mode, rejects
incomplete capacity checks and clarification-only firmware, validates the document
request contract, and checks launcher-plus-child memory accounting. Windows cleanup
targets only the owned service process tree. Failed runs do not count as acceptance.

Reproduce from `Voltforge_AI`:

```powershell
rtk proxy .venv\Scripts\python.exe scripts\verify_service.py --single-request-capacity --output test-results\research\closeout-capacity.json
rtk proxy .venv\Scripts\python.exe scripts\evaluate_research.py --fixture tests\fixtures\research_firmware_limits_v7.json --output test-results\research\closeout-v7.json --temperature 0.3 --max-tokens 768 --seed 42
rtk proxy .venv\Scripts\python.exe -m unittest discover -s tests
rtk proxy .venv\Scripts\python.exe -m unittest test_health_status test_security_va
```

## Remaining release gates and handoff

1. Live consented search → evidence → stream → saved citations → reopen still
   requires provider access. Rechecked this increment: public HTTPS fetch passed;
   live search returned `WEB_DISABLED`. The user has already said no key is available.
2. UI and BL still configure the ngrok issuer. Real session restoration without
   its interstitial requires a reachable intended identity endpoint and a real
   reload/session-expiry/deep-link acceptance run. Controlled auth fixtures do not
   close this gate.
3. Accept the declared supported quality/deployment scope and run the final selected
   profile through the authenticated BL/UI persistence path before release. Do not
   repeat rejected model experiments or treat this bounded check as general firmware
   reliability. Any broader deployment needs its own workload/capacity qualification.
   The existing live runner was inspected: its username/password are absent from
   this process environment and the three applications' standard `.env.local`
   files. No credentials were printed and no new account or authentication bypass
   was created. This prevents a fresh authenticated cross-app run in this session.

The original RAG-first sequence remains in force unless the user explicitly permits
platform development against this local/document-only baseline. Such permission
would change development sequencing, not make live search, SSO or production release
complete. No payments or responsive implementation was started by this increment.
