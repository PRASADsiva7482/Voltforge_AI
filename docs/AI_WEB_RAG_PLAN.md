# VoltForge: web research and document retrieval

Created: 2026-10-19. Scope: existing UI, BL and AI applications.

## Current status

**AIR-00, AIR-01, AIR-04 and AIR-05 complete. AIR-06 is in progress: bounded research, public caching and targeted evidence/clarification guards are implemented. The original six regressions pass, but broader condition-specific extraction and calculation quality still fail. A same-size general model was rejected after a source-injection failure. AIR-02/AIR-03 live search acceptance needs a configured credential. Next work remains AIR-06 model-quality acceptance.**

The user authorized a staged implementation with completed steps tracked. The machine-readable status record is [ai-web-rag-backlog.json](ai-web-rag-backlog.json). Update its evidence when completing a step; a mocked test alone does not establish live integration.

| ID | Deliverable | Status |
| --- | --- | --- |
| AIR-00 | Trace current model, streaming, context and persistence | Complete: source audit |
| AIR-01 | Trusted retrieval contract, consent and capability reporting | Complete: [verification](AIR-01_VERIFICATION.md) |
| AIR-02 | Web search and bounded public-page retrieval | In progress: [live provider acceptance pending](AIR-02_VERIFICATION.md) |
| AIR-03 | Cited answers streamed through BL to UI and restored from history | In progress: [implementation and verification](AIR-03_VERIFICATION.md); live provider acceptance pending |
| AIR-04 | Authorized document/datasheet ingestion and lexical RAG | Complete: [implementation and live verification](AIR-04_VERIFICATION.md) |
| AIR-05 | Benchmarked semantic/hybrid retrieval and repeatable bootstrap | Complete: [benchmark and verification](AIR-05_VERIFICATION.md) |
| AIR-06 | Bounded research orchestration and engineering evaluations | In progress: [targeted regressions fixed; broader quality gates open](AIR-06_QUALITY_FOLLOWUP.md) |
| AIR-07 | Live acceptance, security, performance and deployment verification | Planned |

## What the initial AIR-00 source audit established

- `config.py`, `model_manifest.json`, `engine/llm.py`: the configured model is a directly loaded local Qwen2.5-Coder-1.5B GGUF. Defaults include a 4,096-token context and 768 response tokens. Model artifact downloads are not web research.
- `api/routes.py` `/chat/stream` invokes native model streaming. It does not search, fetch source pages or retrieve documents before generating.
- `api/schemas.py`: the response has a citations field, but the request has no implemented web/retrieval contract. An empty citations field is not a retrieval feature.
- `engine/llm.py` `_build_messages` and `_fit_messages`: context currently consists of conversation, authorized project information and approved preferences. The token fitter specially handles project data. Retrieved excerpts need an explicit shared token budget, rather than being appended as arbitrary history.
- BL `AiChatRequest.java` reserves `retrievedEvidence`. `AiGatewayPolicy.java` checks nested project/revision consistency, and `AiServiceImpl.java` forwards that field. This does **not** establish source provenance. Python currently ignores it. Before consuming evidence, make it server-produced; never promote browser-supplied evidence to trusted retrieval.
- BL `ConversationStream.java` already allows citations and retrieval metadata to be persisted. Reuse this conversation infrastructure; do not create a second chat system.
- Existing electronic component metadata and validation are not a document index. Approved user preferences are not arbitrary persistent memory or RAG.
- No search-provider configuration was found in AI Settings. No recognized search-provider credential name was observed in this process environment. This is not a full deployment-secret audit; provider readiness must be verified during AIR-02 without exposing values.

This is a source audit, not a new live runtime or database verification. No database changes were needed or executed for planning.

## Architecture and boundaries

```text
UI: message + research mode + selected permitted documents
  -> BL: authentication, ownership, consent, idempotency, conversation persistence
    -> AI: bounded retrieval -> evidence packing -> existing local model stream
      -> BL: persist response + source provenance -> UI: tokens and compact sources
```

Keep the existing local model, authentication, SSE transport and database. BL owns document authorization and authoritative source records. Any AI search index is derived and rebuildable; it cannot become an independent permissions system. General chat needs no project. Project-linked chat may retrieve only sources the server authorizes for that user and scope.

Maintain three independent concepts: conversation history, explicitly approved user preferences, and retrieved reference documents. Do not silently turn conversations, firmware or serial logs into indexed documents.

### Privacy and retrieval modes

- Add Off / Auto / On research modes inside existing Chat options; start with Off by default. Persist the user's explicit preference using existing conventions.
- Auto searches only after opt-in and when current information or external references are needed. On requests research explicitly. If unavailable, report that instead of suggesting an answer was researched.
- Project-context sharing permits using approved project data in the existing model request. It does **not** permit uploading it to a search company.
- An external search query uses only a reviewed public query. Do not forward complete messages automatically when they contain code, logs or private identifiers. Offer a short editable query preview when needed; keep conversation history, project source and memories out of external queries by default.
- Never trust browser user IDs, source ownership, URLs claimed as verified, or supplied retrieval excerpts. BL authenticates; AI accepts internal retrieval scope only over the existing service-authenticated route.

## Implementation steps and acceptance

### AIR-01: trusted contract and capability reporting

Touch BL AI DTO/gateway/conversation services; AI `api/schemas.py`, `config.py`, `api/routes.py`; existing UI AI API/types and Chat options.

Define bounded research mode, public query consent, authorized document references and an attempt/run identifier compatible with existing retry IDs. Define server-produced source records: stable source ID, canonical URL or authorized document ID, title, retrieved time, excerpt, content hash and page/section where available. Distinguish search snippets from fetched page excerpts. Do not mark snippets as full-page verification.

Remove client authority over `retrievedEvidence`. Preserve existing project sharing, ordinary chat and enum preferences. Capability responses distinguish configured, available, temporarily failing and disabled services. Research being unavailable must not disable ordinary local chat.

Acceptance: forged evidence is ignored/rejected; cross-user/project sources fail authorization; general chat works with no project; disabled research causes zero external retrieval calls; disabled project sharing sends no project context; invalid modes/oversized payloads fail clearly.

### AIR-02: web search and safe fetching

Introduce a small search-provider interface in the AI service using existing HTTP infrastructure. Initial proposed adapter: Brave Web Search, configurable and disabled when no key is present. Do not buy a subscription or create an account automatically. A credential cannot be downloaded by bootstrap.

Implement search result normalization, bounded query/result counts, timeouts, rate-limit handling, limited retries and cancellation. Prefer original datasheets, manufacturer documentation and official technical sources for engineering queries. Ranking is a heuristic, not a guarantee of correctness.

Fetch only public HTTPS pages. Block URL credentials, local/private/link-local/metadata addresses and unsupported schemes. Validate every redirect and bind requests to validated DNS destinations to prevent rebinding; disable redirects until that is implemented. Apply response/decompressed-size limits, per-page and total deadlines, content-type checks and bounded extraction. No JavaScript execution. Treat source text as untrusted data, never instructions to run tools or reveal private data.

Acceptance: test provider failures/429/timeouts, no results, malicious redirects, IPv4/IPv6 private addresses, DNS rebinding, huge/compressed bodies and prompt-injection pages. Then perform a real configured search and fetch against public technical references. Record only timings/statuses and public-source evidence, never credentials or private queries.

### AIR-03: sourced streamed answers and clean UI

Use existing SSE and conversation persistence. Send accepted/start promptly; report concise real stages (Searching, Reading, Generating). Stream native generated tokens after evidence is ready. Do not fake token streaming by splitting a finished response.

Extend `_fit_messages` to budget instructions, current question, history, permitted project context, retrieved excerpts and response tokens together. Preserve source IDs through truncation. Disclose when evidence is missing, abbreviated or conflicting. Model answers must not imply browsing if none occurred.

Render compact inline references and a collapsible source list only when sources exist. Validate reference IDs against retrieved records and allow safe links only. ID validation establishes provenance, not that every claim is semantically supported; evaluate claim support separately. Preserve Markdown/code copying and the recently simplified header/composer. Do not restore confidence badges, provider names, assistant avatars or verbose internal reasoning logs.

Persist sources with the response attempt so reload and retry restore the correct references. Stop/disconnect cancels retrieval and generation; partial/failed responses stay honest. Never mix sources from a previous attempt into a new one.

Acceptance: browser-observed incremental tokens through UI -> BL -> AI; stop during search/fetch/generation; no duplicate user messages on retry; failed search never claims research success; citations remain valid after reload; no private query leakage; desktop/mobile clean layout and keyboard operation.

### AIR-04: document and datasheet RAG

First inspect the configured development DB schema and existing migration conventions. Reuse compatible tables. If none exist, create versioned additive migrations for authorized source documents, document versions/chunks and indexing status. Exact DDL follows schema inspection, not assumptions in this plan.

Support explicit authorized uploads or selected public datasheet URLs, bounded text/PDF extraction, checksum-based deduplication, versioning and useful failure states. Preserve page/section provenance. Scanned PDFs without OCR support must be shown as unsupported, not indexed as empty successes.

Start with lexical/exact-part-number retrieval over authorized chunks. RAG means retrieving evidence before generation; a vector database is not required for the first useful release. Use current DB capabilities if suitable. Establish ownership/project scope before retrieval and recheck access before source delivery; never fetch across all users and filter only in the browser.

Acceptance: ingest/read/search a real authorized datasheet, answer with correct page references, isolate two users/projects, handle changed/deleted sources, preserve chat history, recover interrupted indexing. Apply and verify migrations only on the confirmed development database; report unavailable DB access honestly.

### AIR-05: semantic/hybrid retrieval and bootstrap

Benchmark an appropriately sized embedding model locally before selecting and pinning it. Combine exact identifiers/lexical matches with semantic relevance; introduce reranking only if measured quality justifies its latency and memory. Persist embedding model/version identity so an upgrade triggers a safe rebuild instead of incompatible-vector reuse.

Extend existing `bootstrap.py`, `download_model.py`/manifest conventions for optional embedding artifacts with pinned revisions, checksums, licenses, resumable/idempotent setup and clear readiness. Reuse an in-process runtime when practical; no Ollama requirement and no unneeded hosted vector database. Keep the database authoritative and derived indexes recoverable.

Acceptance: clean-machine setup, repeat setup, corrupt artifact, offline restart with existing artifacts, retrieval isolation, rebuild after model change, precision on exact part numbers, recall on paraphrases, bounded RAM/latency. Select the model only after measuring this machine and the engineering evaluation set.

### AIR-06: bounded research and reasoning workflow

Add a small server-controlled search/read/compare/answer/check workflow, not unrestricted autonomous browsing. Limit tool steps, total time, pages, retries, tokens and concurrency. Cache only appropriate public retrieval data with freshness/expiry; never cross-cache private answers between owners.

Show short factual progress summaries. Do not expose hidden chain-of-thought or fabricate thinking activity. Keep code execution, circuit modification and simulator execution outside research tools unless an existing authorized feature explicitly invokes them. Retrieved claims are not verified simulation results.

Evaluate the existing 1.5B model first. RAG cannot by itself make it a larger reasoning model. An optional larger GGUF needs measured RAM/CPU/GPU capacity, license checks, answer-quality improvement and preserved streaming responsiveness before changing defaults.

Acceptance: ambiguous questions ask for missing information; fresh-information questions retrieve with consent; conflicting/insufficient sources produce qualified answers; malicious documents cannot override policies; no endless loops; citations support engineering claims in a reviewed evaluation set.

### AIR-07: release verification

Run existing AI tests, BL tests, UI type/build/lint and focused browser tests. Add relevant tests throughout the preceding steps, not only here. Record the exact commands, results, environment and remaining failures without masking existing failures.

Measure cold/warm readiness, search/fetch latency, first stage event, first generated token, generation throughput, full-response latency, peak memory, cancellation time and concurrent-user behavior. Establish numeric budgets from the actual hardware baseline rather than promise unmeasured performance. Use relevance and citation-support evaluations as well as speed measurements.

Live acceptance requires: authenticated search -> real public evidence -> incremental answer -> persisted citations -> reload; authorized document retrieval; two-user isolation; no-network mode; provider outage; stop/retry; project-sharing off; private query consent; no regression to normal chat/project workflows. Mock tests do not close the live acceptance gate. Keep feature flags available for rollback without deleting stored conversations/documents.

## Initial dependencies and honest limitations

- Search provider credentials and quota must be configured for live search. No credential values should appear in logs, UI or reports.
- Development schema inspection and additive document migration were completed in AIR-04; deployment environments still need the versioned migration before starting the updated backend.
- AIR-05 selected MiniLM Q8 on the existing GGUF runtime and measured a local baseline. Broader production performance targets remain AIR-07.
- Paid/paywalled/blocked websites and scanned documents may remain unavailable. Source summaries must respect access restrictions and avoid reproducing entire copyrighted pages.
- Existing context-window limits require selective retrieval. Internet access is not unlimited knowledge or guaranteed correct engineering advice.

## Primary references used for the design

- Brave official API and authentication: https://api-dashboard.search.brave.com/api-reference/web/search/get and https://api-dashboard.search.brave.com/documentation/guides/authentication
- OWASP SSRF prevention, including redirects and DNS considerations: https://cheatsheetseries.owasp.org/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html
- Sentence Transformers retrieval/reranking architecture: https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html

## Change log

- 2026-10-23: Added the planning-only [platform delivery plan](PLATFORM_DELIVERY_PLAN.md) and [detailed backlog](platform-delivery-backlog.json) for AI delivery/performance, payments and complete mobile/tablet support. These follow canonical RAG acceptance; AIR-06 remains the next implementation item and AIR-02/03/07 retain their existing open gates. This planning addition performs no runtime or database work.

- 2026-10-19: AIR-00 source audit completed; staged plan and status record created. No runtime changes, model downloads, database operations or live search verification performed in this planning stage.
- 2026-10-19: AIR-01 implemented across UI, BL and AI; verification recorded in `AIR-01_VERIFICATION.md`. Research modes are contract-validated and unavailable adapters fail closed. Active mode selection/preference persistence moves to AIR-03 with working execution, avoiding decorative controls. No development DB changes or live provider verification claimed.
- 2026-10-22: AIR-02 code and automated checks implemented; real public Arduino-page fetch passed. Live search-provider verification is pending because retrieval is disabled and no provider credential is configured. Details and activation steps are in `AIR-02_VERIFICATION.md`; AIR-02 is not marked complete.
- 2026-10-22: AIR-03 integrated explicit reviewed queries, token-budgeted evidence, native SSE, attempt-bound source persistence and compact sources in the existing UI. Off/On are implemented; Auto decision-making remains AIR-06. No database migration or configuration activation was needed. Live authenticated provider-to-browser acceptance remains pending, so AIR-03 is not marked complete.
- 2026-10-22: AIR-04 added owner/project-authorized uploads, immutable versions, lexical retrieval and authenticated source excerpts. VF_DOCUMENT_001 was applied and verified against the inspected development database. Document-only chat does not require Brave. PDF parsing runs in a bounded child JVM; slow native prompt processing now emits transport keepalives. See AIR-04_VERIFICATION.md for final acceptance evidence and limits.
- 2026-10-22: AIR-04 final live acceptance passed: 14-page Arduino PDF, 65 native chunks, 3 persisted references, reload and retry replay. Compact evidence budget added after native timing failures. All five current database scripts verified on the shared VPN development DB; no pending migration commit. AIR-04 complete; AIR-05 is next. Earlier public-search and release-load gates remain open.
- 2026-10-22: AIR-05 completed. Benchmarked and pinned MiniLM Q8 with automatic verified setup, no new runtime dependencies, owner-scoped hybrid ranking and model-bound derived vectors. Applied VF_EMBEDDING_001 to the inspected shared development database and verified all six scripts. Fresh environment, offline/corrupt/repeat setup, cache rebuild/deletion, native hybrid conversation persistence and targeted regression tests passed. AIR-06 is next; public-search credentials and wider load/quality gates remain open.
- 2026-10-23: AIR-06 bounded workflow, public-page excerpt cache and Windows retry scheduling repair implemented. Native 1.5B evaluation passed 5/6 pattern checks but only 4/6 reviewed rubrics; dimensional confusion and conflict qualification remain open. AIR-06 stays in progress. Auto stays unavailable; external queries still require explicit review and consent. No database changes. See AIR-06_VERIFICATION.md.
