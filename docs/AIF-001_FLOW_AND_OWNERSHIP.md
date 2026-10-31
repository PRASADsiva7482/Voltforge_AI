# AIF-001 — current AI flow and ownership

Reviewed 2026-10-26. Completed as a source/contract mapping task under the user's
explicit authorization to start platform development with a local/document-only
baseline. This is not full RAG or production acceptance. Live search, tunnel-free
identity restoration and final real-account integration remain release gates.

## Source entry points

| Boundary | Inspected source and responsibility |
| --- | --- |
| UI presentation/current request | [AiChatPanel.tsx](../../Voltforge_UI/src/features/ai/AiChatPanel.tsx): `handleSendStream`, `buildPayload`, `restoreMessages`, `cancelStream` |
| Browser transport/auth | [services.ts](../../Voltforge_UI/src/api/services.ts): `aiApi.chatStream`, token refresh, authenticated POST, abort signal |
| History projection | [useConversationHistory.ts](../../Voltforge_UI/src/features/ai/useConversationHistory.ts): selection, paginated reads, reconciliation, generation counters |
| History API | [conversationApi.ts](../../Voltforge_UI/src/features/ai/conversationApi.ts): CRUD/stop, saved-message types and merge by ID/sequence |
| BL identity/API | [ConversationController.java](../../Voltforge_BL/src/main/java/in/voltforge/api/ai/conversation/ConversationController.java): JWT subject, persisted conversation endpoints, nonbuffered SSE headers |
| Durable turn/context | [ConversationStore.java](../../Voltforge_BL/src/main/java/in/voltforge/api/ai/conversation/ConversationStore.java): ownership, idempotency, consent, history, attempts, checkpoints and stop |
| Durable stream lifecycle | [ConversationStream.java](../../Voltforge_BL/src/main/java/in/voltforge/api/ai/conversation/ConversationStream.java): accepted/saved events, partial checkpoints, errors and cancellation cleanup |
| Document authorization/retrieval | [DocumentService.java](../../Voltforge_BL/src/main/java/in/voltforge/api/ai/document/DocumentService.java): selected owner/project versions, bounded lexical/hybrid candidates and private excerpts |
| Gateway | [AiServiceImpl.java](../../Voltforge_BL/src/main/java/in/voltforge/api/ai/service/impl/AiServiceImpl.java): request validation/body, WebClient stream, telemetry, typed failures and lease release |
| BL admission | [AiRequestAdmission.java](../../Voltforge_BL/src/main/java/in/voltforge/api/ai/gateway/AiRequestAdmission.java): process-local per-user/project active-stream counts |
| AI request/SSE | [routes.py](../api/routes.py), [schemas.py](../api/schemas.py): service-authenticated document/attempt contract, sources/deltas/completion, keepalives |
| Evidence/prompt | [research_workflow.py](../engine/research_workflow.py), [llm.py](../engine/llm.py): evidence boundaries, token budget, capability clarification and native generation |
| Native ownership | [runtime.py](../engine/runtime.py): one native lock, bounded admission slots, bounded token queue, cancellation and cleanup |

## Trace A: ordinary persisted chat

1. `handleSendStream` checks authentication/history readiness, holds its local send
   lock, creates a stable UUID `clientRequestId`, and creates a conversation if needed.
   Ordinary workspace chat has no invented project association or context payload.
2. `aiApi.chatStream` refreshes the existing Keycloak token and POSTs to
   `/ai/conversations/{id}/messages/stream`, carrying the request and abort signal.
3. Controller derives the subject from the JWT. `ConversationStore.begin` checks
   ownership, request shape, research policy and prior operation identity. The user
   message and pending assistant message commit before the stream's `accepted` event.
4. Existing completed history is bounded to 20 messages/60,000 characters, with
   individual excerpts bounded to 12,000 characters. Stored preferences are supplied
   through the memory service; browser-supplied private memory/evidence is cleared.
5. BL gateway admission acquires a user/project lease and sends the bounded private
   request. AI runtime serializes native work and enforces its own queue/deadline.
6. AI emits genuine answer deltas. BL checkpoints partial content and maps terminal
   outcomes into durable message status. `complete` is an AI output event; `saved`
   is the backend's authoritative persisted-message event.
7. UI renders only after `accepted`, reconciles `saved` by server message identity,
   and refreshes history summaries. Reload reads the backend; localStorage stores
   only a selected conversation ID, not authoritative transcript content.

## Trace B: consented project plus selected private document

1. A conversation is associated with the selected project. In the standalone AI
   workspace, project context is loaded by that association; unrelated live canvas
   state is not substituted. Sharing-off omits project data. A document may be
   selected while public research remains `off`.
2. BL checks project access and the supplied revision when sharing is enabled.
   Project-scoped documents require that matching project context; document access
   is checked on the server independently of UI visibility.
3. `DocumentService.retrieve` chooses current authorized document versions, bounds
   hybrid candidates to 2560, and returns at most five reference excerpts with
   immutable document/version/page identifiers and hashes.
4. BL creates the attempt ID and sends authorized excerpts through its private
   transport. AI validates that each excerpt belongs to the selected documents and
   that an attempt ID exists. It fits evidence to the actual prompt budget and
   performs capability/calculation/reference checks within the existing runtime.
5. AI emits attempt-bound `sources` and answer deltas. BL rejects stale source
   metadata, persists citations with the assistant message, and emits `saved`.
   The UI restores citations from durable metadata on reload; source reads go back
   through the authorized document endpoint.

## Authoritative state and projections

| State | Owner | Permitted projection/lifetime |
| --- | --- | --- |
| Identity and project/document authorization | BL validated identity and access services | UI auth state controls interaction; it grants no server permission |
| Conversation, ordering, messages, terminal status | BL repositories through `ConversationStore` | UI history cache and rendered messages are disposable projections |
| Logical send identity | UI creates UUID; BL enforces its meaning per conversation | Retry reuses the original `clientRequestId`; distinct content cannot reuse it |
| Attempt identity and pending/terminal transition | BL assistant record and checkpoint comparison | AI transports the supplied ID; a stale checkpoint cannot replace a new attempt |
| Active UI request | Panel request generation/ref and AbortController | Invalidated on switch/close; never a substitute for durable status |
| Draft, consent controls, overlay/scroll state | UI | Temporary interaction state; research selection must agree on retry |
| Saved project state/version | Existing project service/editor stores | Explicitly shared, revision-checked request snapshot |
| Retrieved references | BL authorized versioned chunks; AI packed-source projection | Attempt-bound citation metadata retained with the saved answer |
| Persistent preferences | Existing memory service | Distinct from conversation history and prompt-local context |
| BL active-stream lease | `AiRequestAdmission` | Per-process user/project limits, released on terminal/error/cancel |
| Native model/queue/worker | `LocalRuntime` | One active native owner; cancellation retains lock until iterator closes |

The panel and history hook both hold message projections today. They must not grow
into separate durable stores. AIF-003 will make their relationship explicit behind
one request/controller interface while retaining server authority.

## Failure, stop and retry behavior

- Pre-acceptance failure preserves the draft. A reused ID with changed content is
  rejected; replay of an already completed send returns stored user/assistant records.
- Explicit retry is allowed only for the latest response, retaining the logical
  send identity and using a new attempt. Changed research selections require a new
  send or review of the original selection/consent.
- UI Stop aborts fetch and posts `/stop`. The store marks pending records interrupted.
  Upstream Reactor cancellation and AI's cancellation signal stop generation at a
  native token boundary. The stop endpoint alone is not proof of immediate compute
  termination; this remains an AIF-006 lifecycle concern.
- BL checkpoints compare attempt/status and preserve partial content on failure.
  On reload/focus, the UI requests current saved state rather than resending blindly.
- BL stream uses a 120-second inactivity timeout and a three-minute total bound.
  Native generation uses a 120-second deadline in the selected local profile.
  Heartbeats are transport liveness, not generated words; AIF-006 must reconcile
  proxy, gateway, retrieval, queue and inference deadlines as a single budget.

## Smallest justified changes, assigned to existing backlog IDs

| Item | Concrete finding and next change | Boundary to preserve |
| --- | --- | --- |
| AIF-002 | Panel parses individual lines and ignores JSON errors; multiline SSE, EOF, event bounds and terminal rules need explicit fixtures. Extract and validate the transport contract first. | Existing flat/canonical envelopes and BL accepted/saved messages |
| AIF-003 | The same large handler owns consent, fetching, event parsing, message patches and retry state. Move request lifecycle behind one controller after its contract is covered. | History cache remains a server projection; no new global chat store |
| AIF-004/005 | `begin` is transactional and invokes `documents.retrieve`; hybrid ranking may call the private encoder while that transaction is active. Measure and redesign reservation/context assembly before shortening the transaction. | Ownership, snapshot consistency, idempotency and atomic user/assistant persistence |
| AIF-006 | Stop persistence and compute cancellation have distinct mechanisms; BL per-user/project limits do not equal the AI process-wide limit. | Honest stopped/partial status and bounded resources on failure |
| AIF-007/PERF-001 | Existing telemetry lacks a unified measured queue/retrieval/prefill/render view across these boundaries. Establish correlations and comparable workloads before optimizing. | No transcript, token or private evidence in ordinary telemetry |
| PERF-002 | Rendering work is coupled to each delta in the panel. Measure batching and historical Markdown costs after correctness fixtures. | Final flush, copy/selection, scroll anchors and exact answer content |

No additional network service, model, message broker, vector store or database is
needed for this first extraction. Findings are implementation targets, not claims
that every possible failure has been reproduced.

## Deployment and verification

DEC-01 is resolved: this PC, one active native request, bounded overload rejection.
The qualified isolated profile has zero waiting requests, four threads, 4096 context
tokens and 768 output tokens. Normal configuration still permits two waiters unless
explicitly changed. See [local qualification](AIR-06_07_LOCAL_CLOSEOUT.md) for hardware,
model hash, latency/memory data and limits. Six samples are not a production SLA.

Current-turn development verification:

- UI `npm run test:chat-browser`: **14 groups passed** using actual Chromium and
  controlled auth/API responses, including real local HTTP chunk delivery. Covers
  general chat, saved/reload, stop/retry, project sharing and document selection with
  web off. This is a development browser session, not a real-issuer live RAG test.
- BL targeted Maven suites: **49 tests passed**, zero failures/errors/skips. Covers
  research policy, gateway streaming, conversations, document ownership and hybrid
  retrieval. Uses H2/provider/JWT fixtures, not the shared development database.
- The preceding AI increment passed 161 discovery and five health/security tests,
  V2/V3/V5/V6/V7 rubric review, and the native HTTP capacity profile. Those results
  were inspected and reused, not rerun or relabelled as one integrated session.

Commands from the corresponding repository roots:

```powershell
rtk proxy npm.cmd run test:chat-browser
rtk proxy .\mvnw.cmd "-Dtest=AiResearchPolicyTest,AiServiceImplTest,ConversationIntegrationTest,DocumentIntegrationTest,DocumentHybridIntegrationTest" test
```

AIF-001 mapping acceptance is complete within the authorized development scope.
Next is **AIF-002**. Full real-account cross-app persistence, live public search,
tunnel-free SSO and production activation remain tracked release gates; the user
authorized starting development without claiming those gates complete.
