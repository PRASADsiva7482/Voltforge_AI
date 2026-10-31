# AIR-07: release verification in progress

Latest local closeout (2026-10-26): **V6 4/4**, fresh V7 **7/7 at two seeds**, preserved V5/V2/V3, **161 AI + 5 health/security tests**, and the user-selected **one-active/zero-queue local HTTP capacity profile passed**. See [scope, measurements and remaining gates](AIR-06_07_LOCAL_CLOSEOUT.md). Live search, tunnel-free session restoration and final selected-profile cross-app release acceptance remain open. This supersedes older failure/capacity snapshots below; historical results are retained, not rerun or relabelled.

Date: 2026-10-24. The user requested moving to the next work item. Independent release checks now run alongside the unresolved AIR-06 model-quality and AIR-02/03 live-provider gates. This is not release approval, model promotion or a claim that AIR-06's failed engineering cases passed.

## Implemented and executed

Extended the existing `scripts/verify_service.py` with an atomic `--output` report. It records actual nonempty native SSE output before completion, readiness and HTTP responsiveness, authentication, oversized-body rejection, busy rejection, disconnect cancellation followed by successful generation. Failures/interruption replace stale success and retain completed checks without storing error text, credentials or private conversation content. A report cannot pass unless every required isolated HTTP check completed. Added three report integrity tests in `tests/test_service_verification.py`.

Real local model/HTTP result: [AIR-07_NATIVE_HTTP summary](AIR-07_VERIFICATION.md#recorded-acceptance-runs), all six checks passed. A temporary loopback service and temporary random service token were used; the process was terminated afterward. No production session or stored conversation was used. The probe sets 48 output tokens and zero queued requests, so it is not a production-load or capacity benchmark.

| Observed metric | Result |
| --- | ---: |
| Model/service readiness | 7.844 s |
| First start event | < 0.001 s (rounded to 0.000) |
| First nonempty native token | 11.875 s |
| Maximum of five health calls during generation | 0.016 s |
| Completed response after disconnect | 1.39 s |

These are single-run observations on the current local CPU. Completion after disconnect proves recovery in this probe; its response duration is not an isolated measurement of cancellation latency. No p95, multi-user throughput, thermal stability, agreed SLA or production model-quality guarantee is claimed.

## Cross-app results

| Check | Observed result | Scope |
| --- | --- | --- |
| AI unit/integration discovery | 117 passed | Includes guard, retrieval/consent, streaming and report-integrity checks |
| AI health/security | 5 passed | Test-client authentication and readiness checks |
| BL targeted integration/unit tests | 48 passed, Maven build success | Research policy, gateway, conversations, documents and hybrid retrieval; H2 test databases and provider/JWT doubles |
| UI production build | Passed | TypeScript, Vite and bundle budgets; initial static JS 495.5 kB / 550 kB |
| UI lint | Passed with 13 existing warnings | Unused symbols and existing fixture export warning |
| UI AI presentation contract | Passed | Run states, safe errors, source attribution and deduplication |
| UI chat browser suite | 14 groups passed | Actual Chromium with controlled auth/API fixtures and real local HTTP chunk transport |
| Public HTTPS retrieval | Passed | Fixed Arduino page: 417,533 bytes, 2,173 extracted characters |
| Live provider search | **Not verified: WEB_DISABLED** | Configured search disabled; provider credential absent; command returned nonzero |

Browser coverage includes history/reload, rename/search/delete, duplicate prevention, context consent, safe Markdown/code copy, dark/light and mobile layouts, pagination/scroll position, incremental rendering, stop/retry, persisted error presentation, source links and document upload/version/delete interactions. Browser fixture storage is not the shared VPN database and is not a real Keycloak login.

Existing Starlette/httpx deprecation, unclosed `nul` resource warnings and UI lint warnings remain. Backend test logs include expected simulated gateway failures. No shared development DB DDL/DML, stored user-data changes, deployment changes or new model downloads were performed.

## Commands executed

From `Voltforge_AI`:

```powershell
rtk proxy .venv\Scripts\python.exe scripts\verify_service.py --output test-results\research\AIR-07_NATIVE_HTTP.json
rtk proxy .venv\Scripts\python.exe -m unittest discover -s tests
rtk proxy .venv\Scripts\python.exe -m unittest test_health_status test_security_va
rtk proxy .venv\Scripts\python.exe scripts\verify_web_research.py --live-search
rtk proxy .venv\Scripts\python.exe -m compileall -q config.py engine api scripts tests
rtk git diff --check
```

From `Voltforge_BL`:

```powershell
rtk proxy .\mvnw.cmd "-Dtest=AiResearchPolicyTest,AiServiceImplTest,ConversationIntegrationTest,DocumentIntegrationTest,DocumentHybridIntegrationTest" test
```

From `Voltforge_UI`:

```powershell
rtk proxy npm.cmd run build
rtk proxy npm.cmd run lint
rtk proxy npm.cmd run test:ai-presentation
rtk proxy npm.cmd run test:chat-browser
```

## Open acceptance gates

2026-10-24 development follow-up: [database/ownership/native cancellation acceptance](AIR-07_DEVELOPMENT_ACCEPTANCE.md) now verifies a real shared-DB and BL-to-native-AI path with synthetic users, HTTP ownership checks, consent filtering, partial-response cancellation, duplicate-free retry, stale-attempt rejection and source/embedding cleanup. JWT decoding remains a test double; browser/real issuer/provider/load acceptance is still open. The earlier statement that no DB DML occurred applies to the initial verification above, not this explicitly documented synthetic-data follow-up.

2026-10-24 subsequent [real-issuer browser acceptance](AIR-07_BROWSER_ACCEPTANCE.md) passed 13 checks for general chat, including actual Keycloak login, native incremental rendering, persistence, Stop/retry, a new login and browser-offline recovery. Its synthetic conversation was deleted and database counts returned to baseline. This supersedes the earlier general-chat browser/login gap, but does not certify project/two-user browser isolation, provider search or capacity. The ngrok third-party-cookie iframe warning required explicit sign-in after reload; seamless SSO is not claimed.

1. AIR-06 engineering quality: remaining arithmetic/ADC, firmware, serial diagnosis and helpfulness failures are recorded in [AIR-06_MISSING_CONTEXT.md](AIR-06_MISSING_CONTEXT.md). The narrower missing-information changes are complete, but those model failures are not fixed by passing transport tests.
2. Live search: configure the existing provider in the service environment, then verify a consented public query through the authenticated UI → BL → AI path, persist citations and reopen. Model/bootstrap code cannot create a provider subscription or credential.
3. Verify seamless session restoration on an identity deployment without the ngrok interstitial. Abrupt process death during partial generation now passes the recovery checks below. Real-issuer two-user isolation passes alongside general chat, owner project/document flows and pre-generation outage recovery; see AIR-07_CROSS_USER_ACCEPTANCE.md. These browser identity checks supersede the earlier synthetic-JWT limitation for the covered ownership paths.
4. Define target deployment hardware/concurrency and verify representative cold/warm workloads, bounded cancellation and resource budgets. The isolated HTTP observation is a baseline, not capacity sign-off.

AIR-07 is the active verification item. Dependent platform layers, payments and mobile/tablet implementation remain gated by canonical RAG acceptance. No unresolved gate was silently converted to complete.

2026-10-24 [AI-process outage acceptance](AIR-07_OUTAGE_ACCEPTANCE.md) passed 19 real-browser checks with real issuer, shared development MySQL and native AI: stopped service produces a persisted failed response, restart/retry completes the same message IDs, older transcript remains unchanged, and synthetic data/process cleanup succeeds. Unit tests cover the owned-process lifecycle and report guards; lint passed with 13 existing warnings. No production service or schema changes were required. All 12 audited table counts match the baseline, six schema scripts are verified and 14 FK checks report zero orphans.

2026-10-24 [owner project/document browser acceptance](AIR-07_PROJECT_DOCUMENT_ACCEPTANCE.md) passed 27 live checks: project association, sharing-off payload omission, server consent enforcement, explicit document retrieval, saved citations/source reopen, immutable old version, anonymous denial and confirmed deletion without transcript loss. Fixed a reproduced pre-SSE JSON content-negotiation failure in the BL exception handler; added safe encoder diagnostics and rate-aware acceptance cleanup. The initial embedding failure did not recur and its cause remains unestablished. No schema changes; final counts match across 12 tables with six scripts and 14 zero-orphan FK checks verified. Real-issuer cross-user verification still needs a second account.

2026-10-24 subsequent [real-account isolation acceptance](AIR-07_CROSS_USER_ACCEPTANCE.md) passed 36 live checks after creating the user-authorized second account. Actual fresh Keycloak login, private conversation/project/document denial, positive own-conversation access and unchanged owner data are verified. Found and fixed AI's legacy timestamp revision contract: UI chat/validator/proposals and BL now use the existing documentRevision token. 48 BL tests, 14 UI browser groups, proposal/report/process tests, build/type/bundle/package and lint passed (13 existing warnings). No DDL; 11 non-user table counts match baseline, users intentionally increased 7 to 8, six schema scripts and 14 zero-orphan checks verified. Account retained; credentials are outside repository evidence. The earlier second-account blocker is closed. Other release gates above remain open.

## Recorded acceptance runs

Raw per-run JSON reports were removed on 2026-10-24 at the user request. Checked outcomes and scope remain here and in the linked acceptance notes. Runtime reports stay in ignored test-results until reviewed and cleaned up. Deleting an artifact does not close incomplete gates.

| Run | Result | Checks | Scope | Release accepted |
|---|---|---|---|---|
| AIR-07_BROWSER_ACCEPTANCE | passed | 13/13 | real_issuer_browser_chat | False |
| AIR-07_CROSS_USER_ACCEPTANCE | passed | 36/36 | real_issuer_browser_chat | False |
| AIR-07_DEVELOPMENT_ACCEPTANCE | passed | 17/17 | development_shared_mysql_bl_native_ai | False |
| AIR-07_NATIVE_HTTP | passed | 6/6 | isolated_ai_http | False |
| AIR-07_OUTAGE_ACCEPTANCE | passed | 19/19 | real_issuer_browser_chat | False |
| AIR-07_PROJECT_DOCUMENT_ACCEPTANCE | passed | 27/27 | real_issuer_browser_chat | False |

Original native HTTP observations: first token 11.875 s; maximum of five health calls 0.016 s. This was an isolated probe, not a capacity SLA. Owner acceptance: 27 checks; cross-user acceptance: 36 checks and 48 backend regressions. Cross-user DB cleanup retained only the requested additional user (7 to 8); all 11 other table counts returned to baseline, six schema scripts verified, 14 foreign-key checks had zero orphans. No DDL was required.

## Mid-generation crash recovery

Verified 2026-10-25 (Asia/Calcutta), using real Keycloak login, the local UI/BL/native AI and the configured shared development MySQL database. All **21 live checks passed**; this closes the abrupt-process-death scenario, not overall RAG acceptance.

The existing browser runner now supports `--midstream-crash`. It starts an owned AI process, waits for persisted partial output, force-kills that process, verifies terminal persistence and UI recovery, reloads through authentication, restarts AI and retries. It never kills an unrelated process occupying the port.

- 90 characters persisted before the kill; the terminal failed response retained 155 characters.
- Observed kill-to-terminal time: 141 ms in this run, not a latency guarantee.
- Reload restored the failed partial answer. Retry completed with the same eight message IDs and four user messages; the earlier six messages were unchanged and the retried error metadata was cleared.
- Baseline incremental streaming, completion/reload/new-login persistence, Stop/retry and offline recovery also passed.
- The synthetic conversation was deleted and all owned local services on ports 2001, 2002 and 3000 were stopped.

Two earlier attempts exposed harness defects: raw Markdown was compared literally to rendered text, and retry polling raced the accepted retry while cleanup reused an expired initial login. Assertions now separate persisted raw text from rendered Markdown, await the retry HTTP acceptance and use the refreshed owner session for cleanup. One abandoned synthetic fixture was recovered transactionally only after validating its ID, creation time, lack of project and exact test-message shape; its parent/messages were confirmed absent on a fresh connection. No production chat behavior change was necessary for this scenario.

### Validation and database result

- Existing live report guards pass, including failure when any crash-specific check is missing.
- Owned-process tests pass, including forced stop and restart.
- AI research/service-verification tests: **14 passed**.
- UI lint: no errors, 13 existing warnings. Browser runner syntax check passes.
- Full UI build and BL unit suite were not rerun for this harness/output-path increment.
- Post-run audit at 2026-10-24T23:52:19Z: all 12 table counts equal baseline; conversations 3, messages 40, memories 3, projects 5, code files 5, components 71, users 8, project shares and four document tables 0.
- Six schema scripts verified; all 14 foreign-key orphan checks zero; no audit problems; fresh-connection visibility verified. No DDL was applied.

Reproduce with the existing authenticated development environment and owned-service commands configured by the runner:

```powershell
node scripts/run-chat-live-tests.mjs --midstream-crash
```

Run from Voltforge_UI. Credentials belong in the existing environment, never the command or evidence files. Runtime reports go to ignored test-results/chat-live/report-midstream.json. Retain concise outcomes here, then remove disposable reports, screenshots and logs; keep reusable regression scripts, fixtures, manifests and canonical backlogs.

Cleanup completed: 31 generated JSON reports removed from docs after outcome consolidation, plus 54 disposable local artifacts (38 UI test outputs and 16 backend logs/audits). Reusable regression tests and required fixtures remain.

## Sign-in availability recovery, 2026-10-25

The next AIR-07 session check reproduced a sign-in regression in Chromium: a health response identifying another host with the same realm still triggered authorization navigation. AuthProvider swallowed every availability error, session.ts accepted HTML as a successful probe, and its issuer check accepted a matching realm suffix on another host.

The existing login/signup flow now awaits a successful JSON availability response with the exact configured issuer. Failed, malformed, timed-out or mismatched responses retain the current page and retry action. This is a navigation/readiness guard; backend JWT validation remains the authentication boundary. No token storage, silent-SSO redirect fallback or database change was introduced.

The existing browser suite was refreshed for current landing-page labels, runtime-configured origins and the existing eight-second session deadline. Added regressions cover same-realm/different-host rejection, HTML and malformed JSON, private deep-link restoration on reload, recovery after a failed probe and duplicate-click suppression. Reports now use ignored test-results/auth-availability rather than docs/reports.

Validation:
- Before the fix, the same-realm/different-host browser case failed because the page navigated to authorization instead of showing the retryable error.
- Production Chromium suite: **25/25 passed**, including all existing public/anonymous, callback/PKCE, sync failure/timeout, late-token, refresh and invalid-state cases.
- Development-mode targeted same-realm rejection and retry/recovery cases passed.
- UI build, TypeScript and bundle budget passed; lint passed with 13 existing warnings; runner syntax and chat-live report guards passed.
- BL IdentityAvailabilityServiceTest and IdentityHealthSecurityTest: **10 passed**, using the existing controlled HTTP/security fixtures.
- No AI runtime change or shared-database DDL/DML. No new report/test source files; existing scripts were extended and disposable outputs removed after review.

The authentication browser successes use the real installed adapter with controlled identity/token/API responses. The configured-endpoint scenario only verifies public-page availability. These results do **not** establish real-account seamless SSO through the current ngrok iframe interstitial; that deployment gate remains open.

Live-search configuration was rechecked: web_search_enabled=false and no Brave credential in effective AI settings. The provider must be configured before authenticated live search/citation acceptance can run. AIR-07 remains in_progress; engineering quality, live search, tunnel-free session restoration and deployment capacity remain required. All 38 downstream platform items remain planned.

## Engineering-quality prerequisite follow-up, 2026-10-25

[Bounded LED arithmetic](AIR-06_MODEL_COMPARISON.md#deterministic-led-arithmetic-follow-up) fixes the explicit single-LED resistance case without promoting the model. Four new numeric cases and their repeat pass; existing V2 is 6/8 patterns and 5/8 reviewed, V3 is 4/6. AI 123 plus health/security 5 passed. Broader engineering failures remain. No UI/BL or database changes and no new browser/persistence acceptance. User confirmed no provider key is available, so live-search acceptance stays pending. AIR-07 remains in_progress.

2026-10-25 subsequent [explicit ADC transfer calculation](AIR-06_MODEL_COMPARISON.md#explicit-adc-transfer-calculation-follow-up) fixes the original 409-code case using exact arithmetic, not model promotion. Six new cases pass (five calculations, one tie clarification); unchanged V3 improves to 5/6 reviewed, with missing-power-input guidance still failing. AI 128 and health/security 5 passed. No UI/BL/DB changes or new cross-app acceptance; remaining release gates stay open.

2026-10-25 [explicit missing-input follow-up](AIR-06_MODEL_COMPARISON.md#explicit-missing-input-follow-up) now requests either resistor voltage/current or only the absent LED inputs for supported complete records. V3 passes 6/6 reviewed; five fresh explicit-input cases pass as application clarifications. V2 improves to 6/8 reviewed but firmware and serial diagnosis still fail. AI 133 and health/security 5 passed, plus compile/whitespace checks. Temporary evaluation inputs/results removed. No UI/BL/DB changes or new live browser/provider acceptance. AIR-06/AIR-07 stay in_progress; broader native quality, provider, real session and capacity gates remain open.

2026-10-25 [explicit serial configuration follow-up](AIR-06_MODEL_COMPARISON.md#explicit-serial-configuration-follow-up) compares supplied baud settings with conditional guidance and no execution claim. The source-loss/history/project-context exclusions remain intact. V2 improves to 7/8 reviewed, retaining the nonblocking firmware/citation failure; three fresh configuration cases pass as evidence guidance. AI 138 plus health/security 5 passed, with compile/whitespace checks. No UI/BL/DB changes or new live-provider/browser acceptance; temporary files removed. Overall AIR-06/AIR-07 gates remain open.

2026-10-25 [nonblocking firmware and streamed references](AIR-06_MODEL_COMPARISON.md#nonblocking-firmware-and-streamed-reference-follow-up): explicit millis-based LED code requests receive task-specific bounded guidance; unsupported citation IDs are now filtered before SSE deltas. Unchanged V2 passes 8/8 patterns/review, with four native/four application-guided answers. Original plus two new intervals pass both seeds 42 and 7; three synthetic sketches compile through the configured development compiler and pass three-transition timing checks in the installed AVR emulator. AI 142 and health/security 5 pass. This is not real authenticated Playground, physical hardware, general firmware reliability or capacity acceptance. No UI/BL/DB changes; temporary artifacts removed. Fresh adversarial transfer, live provider, tunnel-free session and capacity gates remain open.

2026-10-26 [firmware capability/context transfer](AIR-06_MODEL_COMPARISON.md#firmware-capability-and-context-transfer): evaluator now supports validated synthetic history/project inputs. Frozen six-case native matrix passes **4/6 reviewed**, failing unsupported and contradictory timer capabilities. A conditional-prompt experiment regressed history/code rewriting and was reverted; runtime hashes match the baseline. Four valid sketches compiled and passed three-transition pin/timing checks in AVR emulation. **144 AI and 5 health/security tests passed**, with post-revert targeted checks, compileall and whitespace validation. Retained one reusable regression fixture; temporary reports/HEX/harness removed. No UI/BL/DB changes or fresh browser/live-provider acceptance. Next is capability-aware response selection before firmware synthesis, not repeated model benchmarking. Overall RAG and downstream platform gates remain open.

2026-10-26 subsequent [bounded timing-evidence precheck](AIR-06_MODEL_COMPARISON.md#bounded-timing-evidence-precheck): same-model per-source assessment now precedes explicit research-backed LED/millis synthesis. Unknown decisions give cited, literal clarification without code/proposals; stage JSON remains internal. Shared admission/deadline/cancellation and bounded combined output are verified. V5 passes 6/6 at seeds 42/7 (four native sketches and two application-composed clarifications); V2 8/8 and V3 6/6 pass again. **Fresh V6 fails 3/4**: missing timer evidence, a within-source contradiction and a different target are incorrectly accepted. Broad capability/RAG acceptance remains open. Five synthetic sketches compiled and passed AVR pin/timing checks; 152 AI and five health/security tests passed, with final 37 targeted checks. The assessment adds about 4–8 seconds in observed runs; no capacity acceptance implied. No UI/BL/DB or new real-account/provider verification. Temporary artifacts removed; V6 retained for the next distinct, bounded correction.

2026-10-26 [API and catalog-target evidence checks](AIR-06_MODEL_COMPARISON.md#api-and-target-evidence-checks): two of three V6 failures corrected. Missing timing API and mismatched/unknown board references now clarify before native inference; catalog variants are distinguished without a duplicate registry. V6 is **3/4**, still failing the within-source contradiction. V5 6/6 and V2 8/8 pass review; an explicit Uno target preserves shared-code pin 6 and 825 ms. Two failed contradiction-extraction variants were removed, leaving the original native prompt/schema. **155 AI and five health/security tests pass**, plus compileall. V3 and compiler/emulator results above are historical, not new execution evidence. No UI/BL/DB, live search, real-account or capacity verification. AIR-06/AIR-07 remain in_progress and all platform items planned; next independent work is contradiction/qualification handling with a distinct bounded hypothesis.
