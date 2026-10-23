# AIR-03: researched streaming and saved source references

Date: 2026-10-22. Scope: existing UI, BL and AI applications.

**Implementation is in place. Live authenticated search-to-model-to-database-to-browser acceptance remains pending; AIR-03 stays `in_progress`.** AIR-02's missing provider credential is not bypassed. Development web search remains disabled. No credentials were printed or configuration secrets changed.

## Implemented flow

1. Chat options retains the quiet default interface. Off uses ordinary local chat. On requires a configured adapter plus a public query and explicit per-turn approval. Editing the query revokes consent; accepting a turn consumes it. Reload remembers the user-scoped mode only, not the query or consent. Auto is explicitly unavailable pending AIR-06.
2. BL validates consent, query size and conversation ownership through the existing services. The reviewed research configuration is stored on the user message. Project sharing remains independent; only the public research request reaches the search adapter, never chat history, code or project context.
3. The existing AI stream emits start promptly, then actual searching/generating stages. The bounded AIR-02 adapter retrieves sources. Provider failure or no usable evidence yields an error without an unsourced research-success response.
4. The existing native runtime fits system instructions, question, history, approved project context, source records and response reserve into one exact ChatML token budget. External text is user-role reference data, never a system instruction. Excerpts stay valid structured records, with source IDs preserved through truncation. Metadata lists only records actually included in the model prompt.
5. Native tokens stream as they are generated. Sources arrive before the first token and carry the BL-created attempt ID. BL checkpoints those sources with partial text and retains them after interruption. Retry clears the old attempt's sources, rejects changed research configuration and ignores stale source events. Completed messages keep the final source list in the existing metadata column.
6. UI renders compact, collapsed Sources only for researched responses. HTTPS links exclude embedded credentials and unsafe schemes; snippets are explicitly identified. Reload restores sources from saved message metadata. No provider/confidence badges, avatars or extra general-chat footer were reintroduced.

Unknown `[S<number>]` references are replaced with `[unverified reference]` in the final response. During streaming they can briefly appear as plain text; they never create a source-list link. Reference validation checks supplied IDs, not semantic support for a claim. Sources may be shortened, and the small local model still requires AIR-06/07 quality evaluation. No browser or simulator execution is inferred from retrieval.

## Verification

- Managed Python `.venv\Scripts\python.exe -m unittest discover -s tests`: **68 passed**.
- Managed Python health/security tests: **5 passed**.
- BL `AiResearchPolicyTest,AiServiceImplTest,ConversationIntegrationTest`: **38 passed**, including ownership, disabled context, retry IDs, persisted interrupted sources and stale-attempt isolation. Integration persistence uses the existing H2 test database, not a new live-development database claim.
- UI `npm run build`: passed TypeScript, Vite production build and bundle budget.
- UI `npm run lint`: passed with 13 existing warnings outside this feature.
- UI `npm run test:chat-browser`: **13 scenario groups passed**. Durable API fixtures cover reviewed-query gating, consent revocation after edits/acceptance, mode-only persistence, safe source links and restored citations. Existing real HTTP chunk fixtures verify incremental paint before completion and stop/reload. The suite also covers project consent, private-context exclusion, failures/retries, history, copying, Markdown, scrolling and desktop/mobile dark/light layouts (320–1440px).
- Python `compileall -q api engine scripts`: passed.
- `git diff --check`: passed in UI, BL and AI; Git reported normal line-ending normalization warnings.

- UI `npm run test:ai-presentation`: passed run states, safe errors, attribution and deduplication assertions.
- `scripts/verify_research_runtime.py`: real checksum-verified GGUF loaded; synthetic evidence packed into **675 prompt tokens** within the configured **4,096-token** context. Produced **12 native chunks**, 44 response characters and the expected `[S1]` reference. Load plus first token: 104.86 seconds; total: 118.84 seconds on this busy development machine. This is a single smoke probe, not a performance benchmark or live retrieval check.

The first accidental test invocation used system Python 3.14 and failed runtime/bootstrap tests. The project supports its managed Python 3.12 environment; rerunning the full suite there passed. Existing test-client deprecation/resource warnings remain. BL also reports the existing default MVC async executor warning; deployment concurrency acceptance remains AIR-07.

## Changed files for this stage

- AI: `api/research.py`, `api/routes.py`, `engine/llm.py`; research contract/adapter tests, new `tests/test_research_streaming.py`, `scripts/verify_research_runtime.py`, README and status documents.
- BL: `AiResearchPolicy.java`, `ConversationStore.java`, `ConversationStream.java`, policy and conversation integration tests.
- UI: `AiChatPanel.tsx`, `ResearchOptions.tsx`, new `ResearchSources.tsx`, `aiPresentation.ts`, `chatWorkspace.css`, chat browser test fixtures and acceptance scenarios.

No new dependencies, conversation system, provider integration, tables or database migrations were introduced. Existing message JSON metadata holds reviewed research settings and citations. No running development application was restarted by this stage.

## Remaining acceptance and next stage

- Configure the existing Brave environment variables using [AIR-02 setup](AIR-02_VERIFICATION.md), then verify real search/fetch and authenticated UI -> BL -> AI streaming, stop/retry, saved citations and refresh against the intended development database. Configuration alone does not prove working connectivity or quota.
- Browser API fixtures and H2 tests establish their individual contracts; they do not close that live end-to-end gate.
- Document ingestion/lexical RAG is AIR-04. Semantic retrieval is AIR-05. Auto research, source-conflict evaluation and prompt-injection quality evaluation are AIR-06. No claim of implemented document RAG or guaranteed factual accuracy is made here.
