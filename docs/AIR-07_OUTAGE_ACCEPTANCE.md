# AIR-07: AI-process outage and retry acceptance

Date: 2026-10-24. Scope: real desktop browser, existing identity provider, local BL and native AI, configured shared development MySQL. This increment does not close AIR-07 or accept model quality.

## Implementation

The existing `Voltforge_UI/scripts/run-chat-live-tests.mjs` now accepts `--ai-outage`. It retains the original 13 real-browser checks and requires six additional checks: owned AI process stopped, failure persisted, actionable failure visible, model ready after restart, retry using the same saved message IDs, and final owned-process cleanup.

`scripts/owned-live-service.mjs` launches the repository's managed Python interpreter directly, without a shell or external model app. It refuses an occupied port and stops only its own child process. Startup and shutdown are bounded. Health readiness uses the existing authenticated model endpoint. Login credentials are excluded from the AI child environment; service authentication stays in process environment, never command arguments or reports.

The new scenario stops AI before sending a fourth synthetic question through the actual composer. It requires a saved user message and failed assistant message, visible failure status and Retry action, and no active Stop control. It restarts the actual model, retries through the UI, and verifies the same message IDs, one user/assistant pair, completed nonempty text, and an unchanged earlier transcript. Only the conversation created by this run is stopped/deleted during cleanup.

No mock authentication, API responses, SSE, or model output are used. This tests unavailability before generation; it does not certify process death during a partially generated response, backend-process crashes, or multi-instance recovery.

## Reproduction

Inspect the configured development database first using `Voltforge_BL/scripts/audit_schema.py`. Start the existing local BL at port 2001 with schema validation and migrations disabled for this acceptance run; point its AI gateway at `http://127.0.0.1:2002/voltForge-ai`. Leave AI port 2002 free. Supply the matching service token in `VOLTFORGE_AI_API_TOKEN` and real-login credentials in `VF_CHAT_LIVE_USERNAME` / `VF_CHAT_LIVE_PASSWORD` through a secure process environment. Alternatively, pass `--headed` to log in manually. Never put credentials in source, CLI arguments, reports or browser storage snapshots.

From `Voltforge_UI`:

```powershell
rtk proxy npm.cmd run test:chat-live-report
rtk proxy npm.cmd run test:chat-live-service
rtk proxy npm.cmd run test:chat-live -- --start-ui --allow-development-writes --ai-outage
```

The harness starts/stops AI itself; it does not terminate an already running AI service. `--start-ui` launches the unmodified application on localhost:3000. The outage mode requires the configured local BL URL. Reports go to `test-results/chat-live/report-outage.json`, separate from the earlier baseline report. Missing evidence, cleanup failure, unavailable identity or native runtime produce a nonzero exit, never a passing result.

## Verification status

**Final live rerun passed all 19 required checks, exit 0.** [Sanitized evidence](AIR-07_VERIFICATION.md#recorded-acceptance-runs) records the original 13 checks plus all six process-outage checks. Four synthetic sends produced eight messages; retry added no duplicate user/assistant pair. The one test conversation and its messages were removed through the authenticated API. This verifies existing application recovery behavior; no production UI/BL/AI service change was necessary for this scenario.

The first run passed the actual outage, persistence and duplicate-free retry assertions and deleted its test conversation, but remained blocked because its immediate final port probe raced Windows socket teardown. The helper now waits up to five seconds for socket release after its child exits, without terminating another process. The blocked report is retained locally as `report-outage-teardown-race.json` and is not counted as a pass.

Process-control tests passed for occupied-port protection, duplicate startup, repeated actual child stop/restart, launch failure, readiness timeout and cleanup. Report tests require all outage-specific checks in addition to the original checks, reject stale/incomplete success, and redact exception messages. UI lint passed with the existing 13 warnings; script syntax checks passed.

Pre/post database audits match all 12 audited table counts: three conversations and 40 messages remain, alongside the original users/projects/documents. Six schema scripts are verified, all 14 checked foreign-key relationships have zero orphans, and schema visibility was verified from a fresh connection. No DDL was needed. Matching aggregate counts verify test cleanup, not checksums of existing records; normal login/profile synchronization remains unchanged. Earlier acceptance prose incorrectly said 13 tables; it was corrected to the 12 entries present in its original JSON evidence without changing measurements.

After the final run, ports 2001, 2002 and 3000 had no listeners. Temporary BL, native AI and UI processes were stopped; the user's identity service was untouched. Build/type and BL/AI regression suites were not rerun for this test-harness-only increment; their earlier results remain in the prior acceptance reports.

The known ngrok third-party-cookie iframe interstitial still requires explicit sign-in after reload. Project/two-user real-issuer browser authorization, live search/citations, engineering-quality acceptance and deployment capacity remain separate gates.
