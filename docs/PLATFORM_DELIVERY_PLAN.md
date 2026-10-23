# VoltForge platform delivery plan

Planning baseline: **2026-10-23**. Scope: `Voltforge_UI`, `Voltforge_BL`, `Voltforge_AI`, and their deployment/authentication integration.

**This is a planning deliverable. No application, database, payment, or deployment changes are authorized by this document alone.** The current request is to prepare the backlog before implementation. Existing authorization for development work does not make a planned task completed.

The executable work list is [platform-delivery-backlog.json](platform-delivery-backlog.json). It contains dependencies, responsibilities, implementation scope, acceptance criteria, verification, and evidence fields for every item. This document provides architecture, product decisions, cross-cutting contracts, and release gates. Update both when an agreed scope changes; record progress/evidence in the JSON.

## 1. Delivery order and current status

| Stage | Scope | Entry condition | Exit condition | Current state |
| --- | --- | --- | --- | --- |
| 0 | Finish existing web/RAG work | Continue canonical AIR-06 | AIR-06 quality, AIR-02/03 live web acceptance, AIR-07 release acceptance supported by evidence | Open prerequisite |
| 1 | AI delivery layers and application performance | PLAT-000 RAG gate accepted | AIF-010 and PERF-004 accepted | Planned |
| 2 | Payments and server-side entitlements | Stage 1 accepted; merchant/product decisions recorded | PAY-011 test-mode acceptance; PAY-012 operating readiness | Planned |
| 3 | Full mobile/tablet experience | Stage 2 implementation ready | MOB-009 route/device acceptance | Planned |
| 4 | Integrated release | All three workstreams accepted | REL-001 evidence and REL-002 authorized release | Planned |

Payment work must be responsive from its first implementation; Stage 3 is the complete application pass, including the engineering workspace. These stages are dependency gates, not a promise of calendar dates. Small implementation items may be split after inspection without changing the acceptance requirements or marking a parent complete prematurely.

**Next implementation work remains AIR-06, not a new platform item.** Follow [the existing RAG plan](AI_WEB_RAG_PLAN.md) and [its authoritative status file](ai-web-rag-backlog.json). This plan does not duplicate AIR tasks or reset their evidence.

### RAG evidence that must be closed first

- AIR-00/01/04/05 are recorded complete in the existing backlog. Document ingestion, conversation persistence and hybrid retrieval already exist; do not build them again.
- AIR-06 remains in progress. The follow-up passes the six original regressions using three native responses and three server guidance responses. The expanded seven-case production-limit transfer evaluation passes only 3/7 pattern checks. Condition/revision selection, missing-input handling and calculations require broader quality work. See [AIR-06 quality follow-up](AIR-06_QUALITY_FOLLOWUP.md).
- Evaluate an appropriately licensed larger local model on actual deployment hardware, fresh held-out engineering cases, malicious sources, firmware tasks, memory and latency. Do not equate more parameters, RAG, or a passing keyword check with trustworthy engineering reasoning.
- AIR-02/03 still require configured search-provider access and a live authenticated search → evidence → incremental answer → persisted citation → reopen test. A disabled provider or mocked response cannot close this gate.
- AIR-07 remains planned: isolation, consent, failure/retry/stop, persistence, deployment and measured performance acceptance. Retain unsupported Auto research as unavailable unless the canonical backlog explicitly implements and verifies it.

If credentials or hardware prevent acceptance, record the exact external dependency and continue only work independent of that gate. Do not rename an incomplete RAG feature to complete to begin platform implementation. Planning is independent and is being completed now.

## 2. Inspected architecture and constraints

| Area | Current implementation to reuse | Planning implication |
| --- | --- | --- |
| UI | React/TypeScript/Vite, React Router, TanStack Query, Zustand, Keycloak; lazy routes; Graphite + Signal Orange | Improve existing boundaries and state ownership; preserve theme and working routes |
| AI chat | `AiChatPanel.tsx`, `useConversationHistory.ts`, `conversationApi.ts`, document/research/memory dialogs | Extract responsibilities incrementally; no second chat store or duplicate conversation service |
| BL | Java/Spring Boot, JPA/MySQL, authenticated conversation/document services | BL remains the authority for identity, project access, persistence and future entitlements |
| AI runtime | FastAPI, existing native GGUF inference/embedding runtime, bounded research workflow and SSE | Keep one provider/runtime integration; no Ollama or mandatory new inference service |
| Admission/metrics | `AiRequestAdmission`, `AiGatewayPolicy`, `AiTelemetry` | Extend existing controls; determine deployment scope before distributed infrastructure |
| Engineering UI | Konva, Monaco, AVR/solver workers and existing project/simulation stores | Rendering optimizations must preserve circuit truth and execution semantics |
| Responsive shell | `AppLayout.tsx`, mobile topbar/bottom navigation, chat visual-viewport handling | Consolidate existing behavior; avoid adding a competing mobile shell |
| Billing | No billing/payment/entitlement implementation found in inspected application source | New bounded BL domain is expected; inspect the actual schema before proposing tables |

Source entry points:

- [UI routes](../../Voltforge_UI/src/App.tsx), [shell](../../Voltforge_UI/src/components/layout/AppLayout.tsx), [chat panel](../../Voltforge_UI/src/features/ai/AiChatPanel.tsx), [history](../../Voltforge_UI/src/features/ai/useConversationHistory.ts), [chat styling](../../Voltforge_UI/src/features/ai/chatWorkspace.css).
- [BL AI modules](../../Voltforge_BL/src/main/java/in/voltforge/api/ai), [existing user entity](../../Voltforge_BL/src/main/java/in/voltforge/api/user/entity/User.java), [schema convention](../../Voltforge_BL/docs/schema).
- [AI routes](../api/routes.py), [runtime](../engine/runtime.py), [inference](../engine/llm.py), [research workflow](../engine/research_workflow.py).
- [Existing simulation optimization follow-ups](../../Voltforge_UI/docs/optimization-followup-backlog.json). Reuse OPT-FB evidence and remaining tasks; do not duplicate or close them through this plan.

This is a source/status audit. No fresh database inspection, runtime acceptance, payment-provider account check, model benchmark, or browser performance measurement was executed for this planning deliverable. Existing test reports are historical evidence, not new test results.

## 3. Workstream 1: reliable AI delivery and performance

### 3.1 Logical layers inside the existing applications

| Boundary | Owns | Must not own |
| --- | --- | --- |
| UI presentation | Messages, composer, source disclosure, responsive panels, accessible actions | Authorization, payment truth, database writes, model internals |
| UI conversation controller and transport | Active conversation/attempt state, SSE parsing, cancellation, server reconciliation, cache invalidation | Independent permanent history or unrestricted automatic retries |
| BL access/policy boundary | Authenticated user, ownership, consent, request limits, capability and future entitlement checks | Trusting a client user ID, paid flag, board capability or project permission |
| BL conversation application service | Idempotent sends, attempts, message status/order, short persistence transactions, context assembly | Holding a database transaction for the entire model generation |
| BL gateway/admission boundary | Bounded admission, deadlines, upstream calls, transport mapping, operational telemetry | A second model implementation or bypass of owner checks |
| AI orchestration boundary | Validated evidence, retrieval budgets, prompt budgeting, model/runtime adapter, output events | Access to arbitrary user projects, financial accounts or unrestricted tools |
| Existing persistence/runtime adapters | Conversation/document repositories, local inference, embeddings, consented web access | Making UI state authoritative |

These are responsibility boundaries, not seven new network services. Use existing services/hooks/modules when their responsibility already matches. Add a small abstraction only where tests or measured coupling justify it. Do not introduce Redis, Kafka, a vector database, a second API gateway, or multi-agent orchestration without a measured need and a separate decision record.

### 3.2 End-to-end message contract

1. UI identifies the conversation and creates a stable operation/idempotency key for the send; disable duplicate submission for that operation. Project association is optional: ordinary chat has no fabricated project reference.
2. BL derives identity from authentication; validates conversation/project access, consent, input size and capability. Admission/entitlement checks happen before expensive work. A rejected request returns a typed actionable error.
3. Existing persistence saves the user message and response attempt atomically where required. Retries link to the original message rather than inserting another user message.
4. BL assembles only authorized, permitted context. Saved project data and user-approved live editor/output data have explicit provenance. Private memory and document scopes are checked independently of UI visibility.
5. BL invokes the existing AI service with bounded deadlines and permitted context. AI emits actual status, answer deltas and source information. Keepalives are transport events, not generated words or evidence of progress.
6. UI parses event boundaries across arbitrary network chunks, ignores stale attempt events, and batches paints while retaining every delta. Apply the final buffered delta before terminal state. Never use fake typing to simulate streaming.
7. BL persists the completed answer, or an honest partial/failed/stopped state with attempt-bound sources. UI reconciles server IDs/status. An answer is not labelled saved before acknowledgement.
8. On connection loss or reload, fetch the authoritative conversation/attempt state. Do not automatically resend a possibly successful operation. Offer a safe explicit retry when the recorded state permits it.

Preserve existing field/event names where possible. AIF-002 must first document the actual wire protocol, then add backward-compatible changes with contract fixtures. Do not silently change persisted message meanings or split application state into conflicting stores.

### 3.3 Reliability and privacy requirements

- Bound queue capacity, waiting time, retrieval time, generation time, response size and per-user/project concurrency. Model load and readiness are explicit; an unavailable model returns a useful failure.
- Reconcile UI, BL, proxy and AI deadlines so an outer timeout does not prematurely kill valid inner work. Measure queue, retrieval, prefill and generation separately. Cancellation must release the corresponding leases and native work when supported; distinguish stopping display from stopping computation.
- Process-local admission is appropriate only for a documented single-instance capacity. Before adding replicas, decide whether shared admission is needed and test aggregate limits. Avoid claiming global limits from a per-process map.
- Enforce circuit/project sharing on the server; turning it off excludes private project content from all relevant requests. Preserve separate explicit consent for public web queries that may reveal private details.
- Keep document retrieval owner/project scoped. Cache keys and invalidation must include actual authorization/version boundaries. Permission revocation, document deletion and account switching must invalidate access even on a cache hit.
- No transcript, code, JWT, API key, financial data or hidden reasoning in ordinary telemetry. Use opaque operation correlations with bounded retention/access; avoid user/project IDs as metric labels. Show concise factual progress only when useful, not source/confidence/internal metadata on every message.
- Persistent memory, conversation history and temporary context remain distinct. Changes to their policies are not implied by this optimization plan.

### 3.4 Performance work and proposed budgets

PERF-001 measures before optimization. Use named hardware, model/artifact hashes, corpus size, route, viewport, network profile, concurrency and cold/warm conditions. Report p50/p95, errors, sample count, peak memory and full commands. Do not compare a warm empty chat against a cold retrieval-heavy project request.

| Measure | Initial target or rule | Evidence required |
| --- | --- | --- |
| Local send/stop interaction feedback | Proposed p95 ≤100 ms under the agreed client profile | Browser trace; excludes network/model completion |
| Rendering after a delta reaches the browser | Proposed p95 ≤100 ms in foreground; preserve final flush | Timestamped transport/render measurement with long Markdown/code |
| Page usability | Target LCP ≤2.5 s, INP ≤200 ms, CLS ≤0.1 at p75 where field data exists | Separate lab and field reports; these do not measure inference speed ([Web Vitals](https://web.dev/articles/vitals)) |
| Bundle budgets | Preserve current `test:bundle-budget`; any change needs a measured justification | Build output and route/network trace |
| Native readiness, TTFT, tokens/s and total response | Set numeric budgets after AIR-07/PERF-001 hardware baselines | Queue/retrieval/prefill/generation timings; no invented universal TTFT target |
| Concurrent users | Define supported concurrency and rejection behavior in DEC-01 | Stepped load and soak tests through real BL/AI, bounded memory and no starvation |
| Circuit responsiveness | Retain existing fidelity/solver correctness; budget rendering independently | Relevant OPT-FB browser/worker evidence, not only unit tests |

Optimization order: reduce unnecessary work → bound long-lived data → isolate expensive rendering → improve transport/backend bottlenecks → consider infrastructure changes only if evidence requires them.

- Keep lazy routes; measure whether Monaco/canvas/model assets load on unrelated routes. Prefetch only likely routes and respect constrained connections.
- Use precise Zustand selectors and stable query keys; scope private queries by authenticated identity and project. Cancel/invalidate safely on logout and project switch.
- Batch streaming updates; avoid reparsing all historic Markdown on every token. Consider history windowing only after measuring, preserving selection, copy, code blocks and scroll anchors.
- Bound serial logs and telemetry buffers using existing semantics; do not discard user-visible data silently. Avoid refetch loops and N+1 retrieval queries.
- Reuse existing simulation scheduling/worker/rendering backlog. Never improve FPS by presenting stale solver values as current or silently lowering engineering fidelity.
- Clean UI remains a requirement: one useful message copy action, no duplicate avatars/status rows, compact context controls, and one clear history toggle. Performance work must not restore the clutter previously removed.

## 4. Workstream 2: payments, entitlements and billing UI

### 4.1 Decisions before implementation

Do not choose a provider from the developer's timezone or assume a merchant country. PAY-001 records merchant legal entity/country, customer countries, currency, required payment methods, provider availability, settlement/refund operations, and business approval. Compare eligible providers using their official current documentation and account availability. **No vendor is selected by this plan.**

Recommended initial shape: one provider, hosted checkout/customer management where suitable, individual account billing, simple server-defined plans. Recurring subscriptions versus one-time purchases is a required product decision. Team billing, prepaid credits, promotions and usage overages are optional scope decisions, not silently included features. Default to no automatic overage charges until an explicit policy exists.

Hosted checkout can reduce the application's payment-form responsibility, but does not remove merchant obligations. For an example of the pattern, see [Stripe Checkout](https://docs.stripe.com/payments/checkout); this reference is not a provider selection or a claim that the merchant is eligible.

### 4.2 Data and authority

Inspect the development database and migration history before implementation. Proposed logical records below must be adapted to existing conventions and provider semantics; these are not migration scripts or final table names.

| Logical record | Minimum responsibility and integrity |
| --- | --- |
| Billing customer | Existing internal user FK; provider/customer/merchant/environment references; unique mappings |
| Product/price version | Server-owned feature mapping, immutable historical price/currency/interval references; no client price authority |
| Checkout operation | Owner, idempotency key, provider session/order, status and expiry; unique operation/provider references |
| Subscription or purchase | Provider-authoritative lifecycle, paid period boundaries, cancellation intent; distinguish pending payment from active entitlement |
| Invoice/payment/refund/dispute | Minimal reconciliation references, exact currency/amounts/status; preserve audit history without card data |
| Webhook inbox | Unique provider + merchant/environment + event ID; verified receipt, processing/retry state, payload minimization/retention |
| Entitlement projection | User/feature limits and effective interval derived from verified events/policy; version for cache invalidation |
| Usage ledger/reservation | Stable operation and attempt links, measured units, reserved/finalized/released status, unique settlement |
| Billing audit | Actor/action/time/reference and change reason with least-privilege access; redact payment secrets and personal data |

Use exact monetary representation according to the provider/currency exponent, normally integer minor units; do not assume every currency has two decimals or use binary floating point for financial arithmetic. Keep application roles separate from paid plans. UI visibility is not an entitlement check.

Reuse the existing MySQL/JPA deployment approach. Add versioned non-destructive migrations only when needed; inspect environment, preserve relevant records/schema, apply in development and verify constraints/read/write/idempotent reapplication according to existing conventions. Do not claim MySQL DDL can always be rolled back as a transaction. Plan forward-compatible recovery and deployment order explicitly.

### 4.3 Checkout and webhook lifecycle

1. Authenticated UI requests checkout using a server-known price selection and operation key. BL checks owner, current subscription, price eligibility and allowed return destinations.
2. BL creates/reuses the provider operation idempotently. Handle an ambiguous timeout by reconciling the same operation, not creating another charge.
3. User completes provider-hosted flow. Return page shows pending verification until BL confirms authoritative payment state. Query parameters and success redirects never grant access.
4. The selected provider's webhook signature is verified against the original request bytes and its documented replay protection. Validate merchant/environment and event shape. The endpoint has a narrowly scoped provider-authentication policy; do not disable application security broadly.
5. Durably record a verified event before acknowledging it. Process idempotently with bounded retries using the existing scheduling approach or a small DB-backed worker if needed. Duplicate or out-of-order events must not grant twice, revoke incorrectly or lose changes. Fetch current provider state where ordering cannot be inferred safely.
6. Update the billing record, audit and entitlement projection consistently. Invalidate affected entitlement caches. Schedule reconciliation to recover missed events or processing failures.
7. UI refetches authoritative billing status; unavailable/pending/failed states remain actionable. Existing projects and saved conversations are preserved through billing changes.

The webhook design must handle retries, duplicate deliveries and event ordering. [Stripe's webhook guidance](https://docs.stripe.com/webhooks) documents these concerns for that provider; revalidate the selected provider's exact requirements in PAY-001/PAY-004.

### 4.4 Subscription and usage policy

Document lifecycle transitions before exposing actions: initial purchase, renewal, trial if offered, past due, grace interval, recovery, cancel at period end, immediate cancellation if permitted, plan change/proration, refund, dispute and account deletion. Define the effective entitlement for each transition. Do not infer policy from a provider's payment status alone.

The AI layer exposes a neutral usage-measurement seam first. Billing subsequently enforces quotas server-side before expensive work. Reserve capacity atomically, settle once, and release unused reservations on cancellation/failure/expiry. A retry of the same billed operation must not double-charge. Explicitly decide treatment of partial responses, provider failures, cancelled runs, retrieval-only work and repeated attempts. Only use actual supported runtime measurements; approximate UI token counts are not billable facts.

Start with included quotas/limits if approved; metered charges or credit wallets require their own approved tariff, rounding, period reset, dispute and refund rules. Existing app data remains readable/exportable according to an approved downgrade policy; never delete projects because payment failed.

### 4.5 UI and API scope

Use existing Settings navigation with a Billing destination, selected after route audit. Provide current plan, renewal/cancellation date, confirmed usage, plan comparison, checkout, payment-pending recovery, invoices/receipts, payment-method management, cancel/change plan and support links as supported by the chosen product/provider. Keep mobile layouts and both themes from the first increment.

Logical BL operations: list available plans, fetch own billing/usage, create checkout, open a bounded provider customer session, request permitted subscription changes, receive verified events, reconcile provider state. Define exact paths under existing API conventions in PAY-002. Read/write operations enforce owner and least-privilege admin access. Never put provider secrets in UI or AI requests.

Tax handling, invoice requirements, refunds, cancellation disclosure, customer privacy/retention and merchant eligibility require a documented business/compliance review for the actual markets. No prices, tax rates or legal compliance claims are invented here.

### 4.6 Rollout and recovery

Test mode first with isolated credentials and records. Live money movement requires a separate release decision after reconciliation, webhook reachability, alerts, refunds and support operations pass. Rolling back new checkout must not stop webhook processing or remove access already purchased. Feature flags control new sales independently from existing subscription servicing. Never test real charges or refunds as an incidental build check.

## 5. Workstream 3: complete mobile and tablet web experience

### 5.1 Layout contract

Use existing responsive tokens and component/container width. Audit breakpoints before adding another breakpoint system. Verification widths: 320, 360/390, 768, 820, 1024 and 1280 CSS px, plus a common 1366×768 laptop. Check portrait/landscape, 200% zoom, enlarged text, both themes and long localized labels. These are test sizes, not hardcoded device assumptions.

- Desktop: recognizable navigation, collapsible conversation/library panels and readable content widths.
- Tablet: collapsible rail/drawers; two useful engineering panes where width permits; no permanently squeezed three/four-pane workspace.
- Phone: compact primary navigation with an accessible More/drawer pattern as needed; one active engineering pane, persistent project/run/save context, contextual inspector and history drawers.
- Reuse safe-area and visual-viewport handling for the composer and fixed navigation. Scope overflow to content regions. Use `min-width: 0`/`min-height: 0` correctly rather than arbitrary offsets.
- One overlay focus owner at a time; Escape, backdrop, focus restore, scroll locking and screen-reader labels. Browser back should close navigational drawers predictably without discarding edits.
- Prefer approximately 44×44 CSS px touch hit areas for important controls, even when icons are small. Evaluate WCAG 2.2 AA requirements and exceptions accurately; its target-size minimum is 24×24 CSS px or qualifying spacing/other exceptions, not a universal 44 px rule ([W3C explanation](https://www.w3.org/WAI/WCAG22/Understanding/target-size-minimum.html)).

### 5.2 Route coverage

Every listed route receives empty/loading/error/populated checks, keyboard/touch coverage, theme checks and state restoration where relevant. Route strings below reflect the inspected router; proposed Billing lives under the existing settings pattern after design approval.

| Existing route(s) | Required mobile/tablet behavior |
| --- | --- |
| `/` | Navigation, readable sections, calls to action and auth handoff |
| `/dashboard` | Real metrics/recent projects, clean hierarchy, loading and empty states |
| `/projects`, `/projects/new` | Search/filter/sort, safe actions, mandatory trimmed project name, board/privacy validation, keyboard-visible forms |
| `/ai-chat` | History drawer, compact header, independent message scroll, safe-area composer, streaming/copy/sources/memory/document controls |
| `/explore`, `/public-project/:projectId` | Browsing, authorized public preview, unavailable/private error, open/fork behavior only where implemented |
| `/labs`, `/labs/:labId` | Catalog, challenge instructions/results, workspace controls and honest capability limits |
| `/profile`, `/profile/:userId`, `/settings` | Account/preferences/forms, validation, long text, theme and privacy controls; future Billing included |
| `/admin` when enabled | Authorized responsive tables/filters/actions; no invented new admin rights |
| `/editor/:projectId` | Project loading, code/board/canvas restoration, saving/conflicts, run/stop/serial, AI, panels and touch editing |
| `/editor/share`, `/editor/sandbox`, `/sandbox` alias | Existing share/guest boundaries, correct fallback and navigation; no false persistence claim |
| Auth/redirect/error flows | Existing Keycloak theme where available, login/logout/expiry, deep links, unavailable route and checkout return |

Component inventory, inspectors, engineering lab subviews, firmware/board configuration, project sharing/settings, notifications, menus and dialogs are included through their owning routes. Do not create duplicate routes just because these are named product features.

### 5.3 AI and engineering workspace behavior

AI full-page mode uses the page, not a popup inside a page. Playground AI is a contextual panel/drawer with the same conversation controller. General chat has no project context unless intentionally associated. Panel collapse, orientation and route transitions must not create new conversations, duplicate streams or lose drafts. Reveal source details on demand; preserve clean messages without confidence/role/internal metadata clutter.

On phones, make Canvas, Code, Serial and AI reachable as workspace views while retaining shared state. Decide whether inactive Monaco/canvas views remain mounted or serialize view state based on memory measurements; changing layout must not recreate the execution engine, lose undo history or reset the active run. Tablet can expose two panes with measured minimum widths. Inspectors and the component library become drawers/sheets with clear return focus.

Canvas touch acceptance includes select, insert, drag, wire endpoints, pan, pinch zoom, snap, delete, undo/redo and contextual actions. Use scoped pointer capture/gesture discrimination so scrolling does not accidentally draw wires or move components. Provide alternatives for actions currently dependent on hover, right click or precise dragging. Preserve browser zoom outside the canvas and offer keyboard/button alternatives.

Serial keeps real run-bound logs, bounded local scrolling, expand/collapse and supported input; it must not cover Run/Stop/Save or invent output. Code editor must remain usable with touch selection, virtual keyboard, horizontal code scroll and current diagnostics. Firmware/toolchain limitations stay visible. Unsupported mobile execution must be explicitly reported if hardware testing establishes a limitation, not silently replaced by fake results.

### 5.4 Performance, accessibility and offline boundaries

Use representative Android and iOS hardware in addition to Playwright Chromium/Firefox/WebKit. Browser engine emulation is not proof of physical keyboard, Safari safe-area, GPU, heat or battery behavior. Test reconnect and interrupted saves using existing conflict/revision semantics. Do not add offline caching of private transcripts/code, a service worker, native apps or push notifications without a separate decision.

Measure long chats, large component libraries, realistic circuits, repeated orientation changes, background/foreground and 20-minute use on representative hardware. Use render budgets without altering physical solver truth. Define what backgrounding does to a live run; never show stale results as currently running after a suspended runtime.

Accessibility acceptance includes focus visibility/order, screen-reader names/status announcements, reduced motion, contrast in both themes, reflow, touch targets, non-drag alternatives and IME-safe Enter/Shift+Enter handling. Avoid announcing every streamed token. Long code/tables may scroll locally; the entire page must not scroll horizontally because of them.

## 6. Decision register

| ID | Decision | Required before | Default planning position |
| --- | --- | --- | --- |
| DEC-01 | Deployment hardware, single/multiple instances, expected concurrency and workload | AIF-001 | Measure existing topology; no distributed infrastructure assumed |
| DEC-02 | Licensed local model, quality rubric, hardware budget and release thresholds | AIR-06 / PLAT-000 | Current default unchanged until evidence supports promotion |
| DEC-03 | Merchant country/entity, provider eligibility, markets, currencies/payment methods | PAY-001 | Provider unselected; hosted checkout preferred where suitable |
| DEC-04 | Purchases/subscriptions, plans/prices, quotas, trials, overages, grace/refund/tax policy | PAY-001 | No price invented; no automatic overage billing |
| DEC-05 | Individual versus organization billing and account lifecycle | PAY-001 | Individual account scope proposed; teams need separate design |
| DEC-06 | Minimum supported browsers/devices and mobile execution expectations | MOB-001 | Responsive web first; actual device acceptance required |
| DEC-07 | Retention/access for billing audit, webhook payloads, usage and operational traces | PAY-002 / AIF-007 | Minimize data; retain only documented operational/legal need |
| DEC-08 | Production activation, domains/webhooks, merchant operations and support owner | PAY-012 / REL-002 | Test mode and feature flags until approved activation |

Unknown decisions are not blockers to writing this plan. They are explicit entry criteria for their implementation items; record answers once in a decision record and link it from the backlog. Do not repeatedly ask permission for already authorized routine development work.

## 7. Verification, release and backlog rules

Statuses: `planned` = not implemented; `in_progress` = work underway with remaining acceptance; `blocked` = named unmet external prerequisite; `complete` = implementation and item-specific evidence accepted. A dependent item cannot start merely because code for its prerequisite exists. Evidence must identify environment, commit/working-tree snapshot, commands, dates, results and remaining limitations. Keep sensitive payloads out of evidence.

Use existing test frameworks and scripts. Add focused tests with implementation, not as a final afterthought:

- AI: current unittest suites and native quality/retrieval evaluations; include real artifacts and resource/cancel/stream tests for release acceptance.
- BL: existing Maven tests for conversation, policy, gateway, persistence and document isolation; extend with billing domain/reconciliation/security tests and selected-provider sandbox contract tests.
- UI: existing type/build/lint, bundle budget, AI presentation/chat browser, project flow, execution and relevant canvas/worker suites; extend route/device and billing browser coverage.
- End to end: authenticated UI → BL → AI token stream → persistent reopen; two-user isolation, sharing off/on, invalid/missing project, stop/retry/timeouts, model/search unavailable, malformed/stale events, logout/account switch and pending-request reload.
- Payments: duplicate/out-of-order webhook, invalid signature, environment mismatch, missed-event reconciliation, provider timeout, async payment/3DS where supported, failed renewal/recovery, cancel/proration/refund/dispute policies and exactly-once usage settlement. No real charges in automated development tests.
- Responsive: route matrix, both themes, keyboard/touch/IME/zoom, real mobile keyboard and safe areas, long histories/code/logs, editor state across layout changes, save/reopen and checkout return.

Release separately reports automated, live integration, load, physical-device and manual accessibility evidence. Passing lint/build is not browser, model quality, database or payment acceptance. Do not mask existing warnings/failures; distinguish baseline failures from regressions.

Deployment sequence is additive schema → backward-compatible BL/AI → UI → controlled flags, refined by the actual change. Test old/new version compatibility, model setup/checksums/offline restart, readiness, backup/recovery and the ability to disable new features without deleting stored data. REL-002 records a limited rollout, monitored thresholds and rollback triggers before expansion.

Relative sizing in the backlog: **S** one bounded module/contract; **M** several related modules; **L** cross-app/domain work; **XL** integration/hardware/provider acceptance. These are complexity indicators, not time estimates. Re-estimate after decisions and baselines; never trade security or verification for a fixed date.

Newly discovered work outside a task becomes a linked follow-up with owner/dependency/acceptance. Do not quietly expand the current scope, duplicate OPT-FB/AIR IDs, or count this planning document as implementation completion.

## 8. Planning deliverables and next action

- This document defines the three workstreams and their integration contracts.
- [The detailed backlog](platform-delivery-backlog.json) defines independently reviewable tasks and completion evidence.
- The existing RAG plan/backlog remain authoritative for prerequisite implementation.
- When implementation is requested, continue **AIR-06** first, close the remaining canonical RAG gates, then execute **PLAT-000 → AIF/PERF → PAY → MOB → REL** in dependency order.

Planning changes only: no schema execution, model change, provider activation, payment operation or application deployment occurred as part of preparing this plan.

Planning validation (2026-10-23): parsed both JSON backlogs; verified 38 unique platform items, required task fields, acyclic/resolvable dependencies, canonical AIR status snapshots, RAG gating for every implementation item, complete dependency coverage for the final release, decision references, and all 18 local Markdown links. Scoped `git diff --check` passed; Git emitted its existing LF-to-CRLF normalization notice. Application builds, runtime tests, database checks and provider/device acceptance were not rerun for this documentation-only change. All 38 implementation items remain planned with empty completion evidence.
