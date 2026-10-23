# AIR-01: trusted research contract

Date: 2026-10-19. Scope: contract, consent, capability reporting and fail-closed behavior across UI, BL and AI.

## Implemented

- Added bounded `research` input: `mode` (`off`, `auto`, `on`), separately reviewed `publicQuery`, `consentToExternalQuery`, and up to ten document references. The absence of research remains ordinary local chat. Off cannot carry an external query or consent. Auto/On require both a nonblank public query and explicit consent.
- Web and document adapters do not exist yet, so both services reject requests for them before generation. No silent substitution of an unresearched answer. Document IDs, including another user's IDs, cannot cause any document access in this stage.
- Removed client authority over `retrievedEvidence` from the direct-chat path: BL ignores it during JSON deserialization and never forwards it even when set programmatically. Python rejects nonempty externally supplied evidence. Persistent conversation requests already stripped the field; this protection is retained.
- Added a bounded server-produced source schema, distinguishing search snippets, fetched pages and document excerpts. It requires source provenance and timezone-aware retrieval timestamps. This is a data contract, **not** an SSRF-safe URL fetcher; fetching safeguards belong to AIR-02.
- Research input remains independent of project sharing when BL rebuilds a context-disabled request. Existing authenticated ownership, server-loaded history and preference handling are preserved.
- Bound conversation requests to the existing persisted generation attempt ID. Direct-chat requests receive a server-generated attempt ID; browser JSON cannot set it.
- Added service-authenticated AI `/research/capabilities`, proxied through authenticated BL `/api/v1/ai/research/capabilities`. Both capabilities accurately report unavailable/unconfigured. No provider credentials or private data are returned.
- Added a compact Web and documents disclosure inside existing Chat options. It reports availability, supports retry when the capability check fails, and distinguishes project sharing from external-query permission. Normal chat remains usable when this endpoint fails.

No enabled research switch is presented while the feature is unsupported. Selection/preference persistence will be connected in AIR-03 when real execution is available; this avoids storing a preference for a nonworking control. The shared types reserve disabled, unavailable, available and temporarily-failing capability states for the upcoming adapters.

## Source files changed in this step

AI:
- `api/research.py`, `api/schemas.py`, `api/routes.py`
- `tests/test_research_contract.py`

BL:
- `ai/dto/AiResearchRequest.java`, `ai/dto/AiChatRequest.java`
- `ai/gateway/AiResearchPolicy.java`, `ai/gateway/AiGatewayPolicy.java`
- `ai/conversation/ConversationStore.java`
- `ai/controller/AiController.java`, `ai/service/AiService.java`, `ai/service/impl/AiServiceImpl.java`
- `AiResearchPolicyTest.java`, `AiServiceImplTest.java`, `ConversationIntegrationTest.java`

UI:
- `src/features/ai/research.ts`, `ResearchOptions.tsx`, `AiChatPanel.tsx`
- `src/api/services.ts`, `src/types/domain.ts`
- `scripts/run-chat-browser-tests.mjs`

All BL main paths are relative to `src/main/java/in/voltforge/api`; test files retain their existing test package paths. Preexisting worktree edits were preserved.

## Verification

- AI: `.venv\Scripts\python.exe -m unittest discover -s tests`: **47 passed**. Includes real ASGI routing/authentication and model/network test doubles; this is not live web/model verification.
- BL: `mvnw.cmd --batch-mode --no-transfer-progress -Dtest=AiResearchPolicyTest,AiServiceImplTest,ConversationIntegrationTest test`: **37 passed**. Includes real HTTP controller checks, H2 persistence, ownership checks and a local HTTP upstream fixture. Found and fixed omitted-boolean deserialization in the new record; consent now safely defaults false.
- BL final focused rerun after direct-chat attempt-ID binding: `mvnw.cmd --batch-mode --no-transfer-progress -Dtest=AiServiceImplTest test`: **11 passed**, Maven BUILD SUCCESS. PowerShell's redirected native-warning handling returned a nonzero wrapper exit despite Maven success; the Maven log and Surefire results establish the test result.
- UI: `npm run build`: **passed**, including TypeScript and bundle-budget checks.
- UI: `npm run lint`: **passed**, with 13 existing warnings in unrelated files.
- UI: `npm run test:chat-browser`: **12 groups passed**, including research availability failure/retry, responsive dark/light layouts, streaming, reload, memory, copy, history, project sharing and stop/retry. The browser suite uses fixture APIs and a real local chunked HTTP stream, not the configured live AI provider.
- All three repositories: `git diff --check`: passed at verification time.

Nonblocking existing warnings include the FastAPI/Starlette test-client deprecation, Python's unclosed `nul` handle, and Java/Lombok deprecation warnings. The BL integration run also warns about its default MVC asynchronous executor; capacity work remains part of AIR-07 rather than being silently called production-verified.

## Database and deployment

No development or production database schema/data changes were required or applied. BL integration tests use isolated H2 storage. No model artifact download or provider subscription was introduced. Running BL/AI processes must load the updated source/artifact to expose the new endpoint; this stage does not claim they were restarted or that live web research works.

Next: AIR-02 search-provider adapter and safe public-page retrieval. Live acceptance requires a configured provider credential/quota; credentials must not be pasted into source or reported in logs.
