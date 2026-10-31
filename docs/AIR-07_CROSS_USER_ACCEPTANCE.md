# AIR-07: real-account isolation acceptance

Verified on 2026-10-24. The final run passed **36 live checks** through the actual UI, public Keycloak issuer, local BL/native AI and configured shared development MySQL. See [redacted evidence](AIR-07_VERIFICATION.md#recorded-acceptance-runs). AIR-07 remains in progress for the other release gates.

## Identity and isolation

A dedicated account was created through the existing Keycloak registration form at the user's request. Fresh browser sign-in succeeded with a different authenticated subject. No JWT decoder, cookie injection, API routes or AI output were mocked. The account and its application profile are retained for the user; credentials are delivered separately and are not stored in this repository or evidence.

- Opening the owner's synthetic conversation in the second browser showed an error and no messages. Reads, history, rename, deletion, Stop and streaming-send requests returned 404.
- Private project reads, edits, deletion and attaching it to a new conversation returned 403.
- Project document listing returned 403. Source excerpts, deletion and replacement uploads returned 404.
- The owner's fixture IDs were absent from the second account's conversation, project and document lists. The second account could create and reopen its own conversation, providing a positive permission control.
- Persisted owner snapshots confirmed the project revision, circuit, firmware, transcript and original source were unchanged after the denied operations.
- The owner flow also verified sharing-off payload omission, server-enforced consent, native document retrieval, saved citations, reopen, old-version preservation and confirmed source deletion without transcript loss.

## Defect found and fixed

The second attempt failed with HTTP 409 `AI_PROJECT_REVISION_STALE` on an unchanged project. Project saves already used `documentRevision`, but AI chat, validation and proposal checks still used `updatedAt`, and the server compared that string with `LocalDateTime.toString()`. Timestamp serialization/precision is not a stable revision contract.

AI consumers and `ProjectServiceImpl.getProjectRevision` now use the existing opaque `documentRevision` token. Two assertions reproduced the contract mismatch before the fix. The integration test accepts the saved revision and verifies a genuine edit rejects the old revision without adding messages. Existing timestamp-bound tabs/proposals must reload/regenerate after deploying UI and BL together. No new schema or AI provider implementation is needed.

The first acceptance attempt also stopped at an integrity assertion that compared the create response with a later database read. The harness now compares persisted snapshots immediately before and after the denied operations, excluding intentional view counters. Both incomplete attempts cleaned up their synthetic content; neither was reported as accepted.

## Changed files

- UI: `AiChatPanel.tsx`, `AiValidatorPanel.tsx`, and the chat browser regression fixture/assertion.
- BL: `ProjectServiceImpl.java`, `ProjectServiceRevisionTest.java`, `ConversationIntegrationTest.java`, and `docs/PROJECT_REVISIONS.md`.
- Acceptance: new `scripts/live-cross-user.mjs`; extended `run-chat-live-tests.mjs`, `live-project-documents.mjs`, `live-chat-report.mjs` and report guard tests.
- AI docs: this report, evidence JSON, canonical verification and both backlog gate notes. Native AI implementation is unchanged in this increment.

## Verification and cleanup

- 36 real-stack browser checks passed, including native incremental answers, durable reload/new login, Stop/retry and browser-offline recovery.
- 48 targeted BL tests passed: conversation 24, project revision 4, AI controller 6, preflight errors 5, document 5 and hybrid retrieval 4.
- 14 UI browser regression groups, AI proposal contract tests, live report guard tests and owned-process lifecycle tests passed.
- UI TypeScript/build/bundle budgets and lint passed; lint has 13 existing warnings. BL package passed after tests, with tests skipped for packaging only.
- Shared DB audit verified six schema scripts and 14 zero-orphan relationships. All 11 non-user table counts returned to baseline; users increased from 7 to 8 because the requested account is retained. No DDL was executed. Cleanup removed only confirmed run-created content. Temporary local UI/BL/AI processes were stopped.

To repeat, configure `VF_CHAT_LIVE_USERNAME`, `VF_CHAT_LIVE_PASSWORD`, `VF_CHAT_LIVE_SECOND_USERNAME`, `VF_CHAT_LIVE_SECOND_PASSWORD` and the matching local AI service token through the test environment, then run from UI:

```text
rtk proxy npm run test:chat-live -- --start-ui --allow-development-writes --cross-user
```

The harness requires the configured local development BL and owns its temporary native AI process. Never place credentials in source, reports or committed environment files.

## Remaining gates

Subsequent [mid-generation crash recovery](AIR-07_VERIFICATION.md#mid-generation-crash-recovery) passed partial persistence, interruption, reload and duplicate-free retry. Seamless session restoration through an identity deployment without the ngrok iframe warning, live search-provider acceptance, engineering-answer quality and agreed hardware/concurrency budgets remain open. Cross-user acceptance does not imply those gates have passed or authorize starting the 38 downstream platform items.
