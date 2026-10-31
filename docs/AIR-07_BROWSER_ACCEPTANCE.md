# AIR-07: real issuer and browser acceptance

Date: **2026-10-24**. Scope: the existing desktop AI Workspace, real configured Keycloak issuer, local BL and native AI, and the configured development MySQL database. This increment is separate from model-quality, internet-search and deployment-capacity acceptance.

## Repeatable command

Added to the existing UI repository:

- `scripts/run-chat-live-tests.mjs`: real browser/API acceptance, with no authentication transform, JWT decoder double, route fulfillment or fabricated SSE.
- `scripts/live-chat-report.mjs`: atomic reports that require all named checks; failures replace earlier success and exclude credentials, URLs and transcript content.
- `scripts/run-chat-live-report-tests.mjs`: report completeness, stale-success replacement, error redaction and service-URL boundary tests.
- `package.json`: `test:chat-live` and `test:chat-live-report` commands. Ordinary tests do not automatically run against the development database.

Start the existing BL and AI services with matching private service authentication. Verify the target is the intended development environment before opting in to test writes. Supply the existing account credentials through `VF_CHAT_LIVE_USERNAME` and `VF_CHAT_LIVE_PASSWORD` in a secure process environment; never commit them or place them in CLI arguments. Alternatively, use `--headed` and complete the login manually in the browser. No password is included in these instructions or the evidence report.

From `Voltforge_UI`:

```powershell
rtk proxy npm.cmd run test:chat-live-report
rtk proxy npm.cmd run test:chat-live -- --start-ui --allow-development-writes
```

`--start-ui` starts the existing, unmodified Vite application at `http://localhost:3000`. Omit it to use an already running UI; `VF_CHAT_LIVE_URL` selects its URL. Remote services require HTTPS; plain HTTP is accepted only for loopback. The existing runtime configuration supplies the BL and identity URLs. Credentials are submitted only to the configured realm. A new browser context performs the second login without copied cookies, stored tokens or injected auth state.

The report is written to `Voltforge_UI/test-results/chat-live/report.json`, and the command exits nonzero for failed or blocked acceptance. Missing credentials/issuer availability cannot produce a passing result. The report explicitly keeps release, live search, project isolation and model-quality acceptance false: this scenario does not certify them.

## Scenario and data boundaries

The run opens the existing New Chat dialog, leaves the optional project unselected, and records only the ID returned by that run's successful create request. It checks visible incremental native output before completion, completed server persistence, reload restoration, the actual Stop button, retry with the same saved message IDs, and restoration after a fresh login. A final browser-offline send must leave the server transcript unchanged; reconnecting and retrying must add only one user/assistant pair.

API reads use the actual browser's authenticated token, retained only in process memory. They are spaced out to respect the existing request limit. Cleanup stops and deletes only the conversation confirmed created by this run, then checks that it is unavailable. It retains the original authenticated context for cleanup even if the second login fails. The test does not delete accounts, projects or pre-existing conversations, and does not upload project data. It does not save browser storage, traces or screenshots containing account/history data.

## Development identity limitation

The public identity endpoint initially timed out. After the user restarted it, non-browser discovery worked, but browser discovery received ngrok's `ERR_NGROK_6024` HTML warning rather than JSON. The test now recognizes that exact warning only on the configured `*.ngrok-free.dev` host. Its discovery request uses ngrok's warning-skip header; top-level browser login clicks the development warning's Visit Site button before normal Keycloak login. This does not bypass authentication, change issuer validation or mock an identity response.

The actual cross-site Keycloak third-party-cookie iframe also received this tunnel warning and timed out. Reload therefore required explicit sign-in before restoring the saved conversation. The report records `reloadRequiredReauthentication` and `identityIframeTunnelWarningObserved`; it must not be presented as seamless SSO restoration. No tokens were persisted to work around this. A stable identity deployment without the tunnel interstitial still needs session-restoration verification.

## Harness corrections and recovery

The first browser attempts identified test/setup issues, not model-quality successes: the New Chat action opens the existing project-selection dialog before creating a conversation; the Windows Java launcher can spawn a child JVM, so temporary services must be terminated using their actual process; and frequent transcript polling hit the unchanged 60-requests/minute BL limit. The harness now follows the dialog and polls at 1.5-second intervals. The final isolated service launcher uses the actual JDK executable.

One rate-limited attempt could not perform API cleanup. Its exact synthetic conversation was removed transactionally only after verifying its ID, absent project association and both exact synthetic user prompts; its messages cascaded away. No broad matching/deletion or customer-content changes were performed. Later attempts retain truthful cleanup status rather than claiming cleanup on an error.

## Results

**Final live browser scenario passed all 13 required checks**, including cleanup; exit 0. [Sanitized evidence](AIR-07_VERIFICATION.md#recorded-acceptance-runs) records real issuer sign-in, visible incremental native output, completed persistence, reload restoration with explicit reauthentication, actual Stop/retry with unchanged message IDs, fresh-login restoration, browser-offline failure with no server write, and duplicate-free retry after reconnecting. Three synthetic sends produced six saved messages before the one test conversation was deleted. This is one desktop Chromium run, not a load or cross-browser certification.

The final read-only database audit matched the pre-run aggregate counts across all 12 tables: seven users, five projects, three conversations and 40 messages remained, with zero document/embedding rows and no orphaned foreign keys. All six schema scripts were verified. No schema changes were required. Account sign-in/synchronization follows the existing app behavior; equal row counts are not a checksum claim about every profile field. Local BL, AI and Vite acceptance processes were stopped, and ports 2001, 2002 and 3000 were no longer listening. The user's identity service was left running.

Earlier partial runs and harness failures are not counted as passes. The final result retains the explicit reload/identity-tunnel limitation. No credentials, raw chat messages or authenticated browser state were saved in the checked-in evidence.

UI production build/type checks and bundle budgets passed. UI lint passed with the same 13 existing warnings. BL packaging passed with tests explicitly skipped for that packaging command. The report guard tests passed; the earlier backend/native acceptance suites remain separately documented in [AIR-07_DEVELOPMENT_ACCEPTANCE.md](AIR-07_DEVELOPMENT_ACCEPTANCE.md).

Subsequent [AI-process outage acceptance](AIR-07_OUTAGE_ACCEPTANCE.md) verifies pre-generation service failure and duplicate-free retry after restarting the native model. Remaining scope includes real-issuer two-user/project/document browser checks, live public-search/citation acceptance, hardware/concurrency measurements and AIR-06 engineering-quality failures. No payment or responsive-platform implementation is unblocked by these increments alone.
