# AIR-07: development database, ownership and native cancellation acceptance

Date: **2026-10-24**. This completes a bounded verification increment within AIR-07. It does not complete AIR-07, AIR-06 model quality or the provider-dependent web acceptance gates, and does not unblock the platform backlog.

## Implementation

Extended the existing opt-in [DevelopmentDocumentSmokeTest](../../Voltforge_BL/src/test/java/in/voltforge/api/DevelopmentDocumentSmokeTest.java) with `liveProjectIsolationConsentCancellationRetryAndSourceCleanup`. It reuses the existing BL controllers, conversation/document services, MySQL repositories, embedding client and native AI gateway. No second conversation implementation, provider integration or database schema was added.

The test creates two synthetic users, one private synthetic project, one short synthetic document and one conversation. It never selects an existing user's project or reads an existing transcript. Cleanup targets only the IDs created by this test. The local AI process uses a temporary service credential and loopback port; its configuration is process-local and the process is terminated afterward. The selected model and generation limits remain unchanged.

## What passed

- The configured database was inspected as `dev/development` before test writes. All six versioned schema scripts were verified, all 14 foreign-key orphan checks returned zero, and tables were visible from a fresh connection.
- A second synthetic user could not read, rename, delete, stop or send to the owner's conversation through the actual BL HTTP controllers. Unauthenticated history returned 401; cross-user operations returned 404. Foreign project attachment was rejected.
- Cross-user document retrieval, source reading and deletion were rejected. Owner source reading worked before deletion.
- With sharing off, BL removed supplied project ID, board, firmware, components, wires, netlist, canvas, simulation/serial data, files, diagnostics, project memory and forged evidence from the outgoing request. This verifies BL request assembly, not browser packet capture. The selected project document was rejected without consent, and that rejected send did not add messages.
- With sharing enabled for the owned project, authorized evidence and real native embeddings were available. The existing gateway streamed an actual model response.
- Cancelling the BL reactive subscriber after a native token interrupted upstream execution and left a nonempty partial assistant message persisted as `interrupted`. This exercises stream-disconnect cancellation, not a physical browser Stop-button click.
- Retrying used the same user/assistant message IDs and a new attempt ID. A late checkpoint from the cancelled attempt could not overwrite the new attempt. The retry completed with persisted document citations; replay created no duplicate messages.
- An authenticated BL HTTP history request reopened the saved response. The four messages comprised the earlier stopped sharing-off turn and the document turn; retry did not create another pair.
- Owner document deletion made its source URL unavailable and removed derived embeddings, while the saved transcript remained. Test cleanup removed the synthetic conversation, project and both accounts.

The live test passed: **1 test, 0 failures, 0 errors, 0 skips**. Its reported test-class duration was 37.14 seconds, including Spring startup; this is not inference latency or a capacity/SLA measurement. The consolidated acceptance summary is [AIR-07_DEVELOPMENT_ACCEPTANCE summary](AIR-07_VERIFICATION.md#recorded-acceptance-runs).

Related BL regressions were rerun: **48 tests passed, zero failures/errors/skips**, covering `AiResearchPolicyTest`, `AiServiceImplTest`, `ConversationIntegrationTest`, `DocumentIntegrationTest` and `DocumentHybridIntegrationTest`. Those use H2/provider doubles and are reported separately from the single live development test. The Maven test lifecycle compiled the updated test source. No UI or AI runtime source changed in this increment; their historical build/test results were not presented as freshly rerun. Existing Spring/debug/Swagger notices and expected simulated gateway-error logging remain.

```powershell
rtk proxy .\mvnw.cmd '-Dtest=AiResearchPolicyTest,AiServiceImplTest,ConversationIntegrationTest,DocumentIntegrationTest,DocumentHybridIntegrationTest' test
```

## Database preservation

Read-only schema audits before and after the run had identical aggregate row counts across all 13 inspected tables and no orphaned foreign keys. Existing counts included seven users, five projects, three conversations, 40 messages and three memories. Document/chunk/version/embedding tables returned to zero. Counts confirm cleanup and are not a checksum-based certification of every existing row's contents.

No DDL or migration was necessary. Only this test's synthetic DML was executed; no existing user data was intentionally updated or deleted. Aggregate audit results are recorded above. Disposable local audit files were removed after verification; no credentials or private conversation contents are retained in these notes.

## Reproduction and scope

Start the existing native AI service with the configured private service credential available to BL. The test defaults to its normal local URL; `VF_DOCUMENT_AI_URL` can point to an isolated local instance. Keep credentials in the existing secure environment, never in command arguments or evidence. From the BL repository, opt in for this invocation only:

```powershell
$previousDocumentSmoke = $env:VF_DOCUMENT_DEV_SMOKE
try {
    $env:VF_DOCUMENT_DEV_SMOKE = 'true'
    rtk proxy .\mvnw.cmd '-Dtest=DevelopmentDocumentSmokeTest#liveProjectIsolationConsentCancellationRetryAndSourceCleanup' test
} finally {
    $env:VF_DOCUMENT_DEV_SMOKE = $previousDocumentSmoke
}
```

The test uses `ddl-auto=validate` and disables Flyway. Run the existing read-only `scripts/audit_schema.py` before and after, using the configured development environment. Do not repoint this opt-in test to production. It has no dependency on the older PDF download used by the separate datasheet smoke test.

**Authentication limitation:** Spring's JWT decoder is a test double assigning two synthetic identities. Server authorization, HTTP routing, the database, embeddings and native generation are real. This does not verify Keycloak token validation, login, browser UI, deployment proxy behavior, live internet search or a new login session. Context-off assembly was inspected at the BL service boundary. The test does not certify engineering correctness of the generated answer.

Remaining AIR-07 work: real issuer/browser acceptance, explicit Stop-button/outage recovery across the full deployed path, configured live-search/citation acceptance, agreed hardware/concurrency measurements and the independent AIR-06 quality failures. Platform layers, payments and responsive implementation remain after those gates.
