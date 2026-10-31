# AIR-07: Real-login project and document acceptance

Date: 2026-10-24. Scope: one real authenticated owner, the existing browser UI, local BL/native AI, and configured shared development MySQL. Cross-user real-issuer isolation is a separate gate and is not inferred from owner or anonymous checks.

## Scenario

The opt-in `Voltforge_UI/scripts/run-chat-live-tests.mjs --project-documents` reuses the existing actual-login/general-chat checks, then calls `scripts/live-project-documents.mjs`. It creates one synthetic private Arduino Uno project through the authenticated project API, with an empty canvas and a clearly marked firmware fixture. The New Chat dialog associates a conversation with the returned project ID. No existing user project is edited or sent to AI.

The scenario verifies:

- The saved project retains its board, private flag and firmware, and its associated conversation exposes the correct editor link.
- An ordinary UI message with sharing disabled contains no synthetic private fixture data or selected documents.
- A small unique synthetic TXT document is uploaded through the existing UI, associated with the project, and initially unselected.
- The actual backend rejects a project-document request without project-sharing consent; no user/pending message pair is saved for that rejected request.
- Explicit UI sharing and document selection send the saved fixture and selected document, with external web research off.
- The native response and document references persist; the owner can reopen the referenced source through the actual source-reader dialog.
- Anonymous requests cannot read the private project, conversation or document source.
- Reload retains the project association and saved messages/references, while sharing and document selection require renewed consent.
- Uploading a second document version preserves the earlier cited version. Deletion requires confirmation, makes the old source endpoint unavailable and preserves the historical chat.
- Cleanup deletes only the project, conversation and document confirmed created by this invocation. Database audits independently check aggregate counts and foreign keys.

The existing report guard requires all original checks plus the project/document checks. `projectOwnerVerified` may become true only after that scenario passes; `projectIsolationVerified`, live-search acceptance, model-quality acceptance and release acceptance remain false. No fabricated responses or authentication/API interception are used. Browser request inspection verifies UI-to-BL context consent; server rejection verifies document gating. This scenario does not claim to capture the complete private BL-to-AI request body.

## Running the checks

Inspect the intended development environment using `Voltforge_BL/scripts/audit_schema.py` before enabling writes. Use the existing local BL on port 2001, pointed at the native AI endpoint on port 2002, with schema validation and migrations disabled for the acceptance run. Leave port 2002 free; the harness starts and stops only its own AI process. Supply matching service authentication through `VOLTFORGE_AI_API_TOKEN` and actual login through the secure process environment or `--headed` manual login, as described in [browser acceptance](AIR-07_BROWSER_ACCEPTANCE.md). Do not put credentials in source or evidence.

From `Voltforge_UI`:

```powershell
rtk proxy npm.cmd run test:chat-live-report
rtk proxy npm.cmd run test:chat-live -- --start-ui --allow-development-writes --project-documents
```

The sanitized report is `test-results/chat-live/report-project-documents.json`. Unavailable dependencies, failed assertions and cleanup failures cannot produce a passing report. The fixture's part number and rating are explicitly synthetic; generated answers are not engineering/model-quality certification.

## Results

**Final live rerun passed all 27 required checks, exit 0.** [Sanitized evidence](AIR-07_VERIFICATION.md#recorded-acceptance-runs) records real login, native chat readiness, project association, context consent, actual document ingestion/retrieval, persisted references, source reopening/version preservation, anonymous access denial and source deletion with preserved history. The final run created one project, one document with two versions and two conversations; all were cleaned up through their authenticated APIs.

The initial run verified project association, sharing-off payload omission, project upload and server-side document consent rejection, then encountered `AI_EMBEDDING_UNAVAILABLE`. Repeated polling after the rejected send exhausted the existing request limit and blocked API cleanup. Only the exact synthetic records were recovered transactionally after checking the unique project/firmware marker, document contents and all synthetic user prompts. The original 40 messages remained. The incomplete report is retained locally as `report-project-documents-initial.json`; it is not a passing result.

That run exposed a production error-reporting defect: `GlobalExceptionHandler` returned an `ApiResponse` without an explicit JSON content type. When an SSE client requested only `text/event-stream`, pre-stream failures could trigger `HttpMediaTypeNotAcceptableException` instead of delivering the meaningful error. A new regression test reproduced the failure before the fix. Error responses now explicitly identify JSON, preserving the HTTP status and error code. Five SSE preflight cases pass, together with the existing conversation/document/gateway tests: 54 backend tests passed. Packaging passed separately with tests skipped for that packaging command.

`DocumentEmbeddingClient` now records only the failing stage and safe exception category/HTTP status, never upstream bodies, document text, credentials, URLs or raw exception messages. An HTTP contract test with a local fixture server confirms the encoder metadata response is accepted, and a separate direct native-runtime probe confirmed the configured embedding model is available with 384 dimensions. The initial embedding failure did not recur in the final live run. Its cause was not established: neither the error-reporting fix nor the successful rerun is presented as an encoder-logic fix. The added diagnostics support investigation if it returns.

The live harness now checks the send response before polling for persisted output, verifies native embedding readiness before creating test records, polls less frequently, and retries only explicit 429 responses with bounded waits. It does not retry ambiguous writes, authorization errors or provider failures, and it leaves the application rate limit unchanged. Report-guard/rate-limit tests, script syntax checks and UI lint passed; lint retains 13 existing warnings. The known identity-tunnel iframe warning and explicit sign-in requirement after reload remain unchanged.

Because the JSON error handler is shared, the existing project/runtime/controller regressions were also executed: `BackendRuntimeCompatibilityTest`, `ProjectServiceRevisionTest` and `AiControllerTest` passed another 23 tests. Total backend validation for this increment is **77 passed, zero failures/errors/skips**, with suite counts recorded in the JSON evidence. These regression suites use their existing isolated databases/provider fixtures; they are distinct from the 27-check actual-login/shared-DB/native-runtime browser run.

Pre/post shared development audits match all 12 audited table counts, including three original conversations and 40 messages. All six schema scripts are verified, all 14 checked FK relationships have zero orphans, and a fresh connection sees the schema. No DDL was performed. Equal aggregate counts establish cleanup, not checksums of every existing record; normal identity/profile synchronization remains the application's existing behavior. Ports 2001, 2002 and 3000 had no listeners after temporary services stopped. The user's identity service was left running.

The existing two-user service tests use a JWT decoder test double. A second real development login is still required to close the distinct real-issuer cross-user browser gate. The current identity tunnel still requires explicit sign-in after reload; seamless SSO, live public search, engineering-quality and load/capacity acceptance remain open. UI source/build and AI suites were not rerun for this increment; frontend changes are acceptance scripts, and no AI runtime logic changed.
