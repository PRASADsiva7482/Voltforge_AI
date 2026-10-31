# AIF-002 — persisted chat stream contract v1

Implemented 2026-10-26 under the approved local/document-only development scope.
Full RAG/live-identity/production release gates remain unchanged.

## Request and identity

UI POSTs to `/ai/conversations/{conversationId}/messages/stream` through BL with
the existing authenticated token, `message`, UUID `clientRequestId`, `retry`,
`includeProjectContext`, research selection/consent and optional shared context.
BL derives the user from JWT, verifies ownership/revision and creates the response
attempt. The browser cannot supply authoritative history, user identity, private
document excerpts or payment state. No request field, database migration or existing
saved-message meaning changed in this increment.

`clientRequestId` identifies the logical user send; retry retains it and creates a
new backend attempt. `conversationId` scopes the send. BL message IDs and sequence
numbers identify durable messages. `attemptId` scopes transient generation/source
events. A new optional top-level `attemptId` on BL `accepted` binds the UI to the
current attempt without changing the nested saved-message shape. Replayed stored
answers may omit it; legacy streams remain readable.

## SSE framing and limits

The UI now uses `chatStream.ts` instead of parsing individual lines inside the
panel. Frames dispatch at blank lines; repeated `data:` lines join with newlines.
UTF-8 decoding spans arbitrary byte boundaries. Comments are ignored; a missing
`event:` field falls back to the JSON `type`. Empty data fields do not inherit a
previous event name. See the [HTML SSE parsing rules](https://html.spec.whatwg.org/multipage/server-sent-events.html#parsing-an-event-stream).

- Actual AI producers emit LF. The Spring gateway accepts LF and CRLF. A shared
  fixture exposed that its current decoder rejects bare CR inside event headers;
  it returns `AI_GATEWAY_STREAM_FAILED`. This limitation is tested and retained
  explicitly in the fixture, not represented as full newline-format support.
- The new browser decoder additionally handles bare CR and split CRLF.
- UI bounds: 1,000,000 decoded characters per frame, 8,000,000 bytes per response,
  120,000 answer characters, five source records per `sources` event. Bounds are
  independent of TCP chunk sizes. BL already bounds saved answers to 120,000.
- Invalid UTF-8, malformed JSON, bad control fields, incompatible declared versions,
  oversized data or EOF before authoritative `saved` produces a fixed safe error.
  Partial displayed content remains available. No malformed JSON or transcript is
  embedded in that error.
- Reader cancellation/release runs on early exit and failure. This decoder does
  not reconnect, replay requests or alter the existing AbortController behavior.
- HTTP success must also have `text/event-stream` content type. Unexpected HTML/JSON
  success bodies are cancelled and treated as unconfirmed responses.

## Envelope compatibility

Both existing flat JSON and canonical `{type, schemaVersion, contractVersion,
payload: {...}}` envelopes are supported. The payload is flattened for presentation
after control-field validation. Optional declared schema must be `1`; declared
contract must have major version `1` and numeric minor/patch. Legacy events with
neither field are still accepted. Event header/type disagreement is rejected.

Unknown event names are ignored and cannot complete a response, even if they carry
a `reply`. Known events with malformed required control fields fail safely.
Optional engineering metadata continues through the existing presentation/proposal
validators; this is not a replacement schema for every engineering action.

| Event | Control and terminal meaning |
| --- | --- |
| `accepted` | BL committed user message. Validate conversation/request IDs, user role, server ID, content, status and sequence. A duplicate same-ID acknowledgement does not insert a second message. |
| `start`, `metadata`, `status`, `thought`, `tool` | Existing metadata/progress presentation; not a save acknowledgement. Transport comments are never progress. |
| `sources` | Bounded citation list. When the accepted attempt is known, missing or mismatched attempt IDs are ignored. BL continues its own source-attempt check. |
| `token`, `delta` | String answer delta, bounded cumulatively. Preserve exact content, including Unicode. Deltas after output terminal are rejected. |
| `citation`, `uncertainty`, `proposal` | Existing bounded presentation/action review. No automatic project mutation introduced. |
| `complete`, `done` | AI output ended; optional reply replaces the displayed partial answer. Does not establish durable success. Duplicate output terminals are ignored. |
| `error` | Existing typed failure presentation, preserving partial text; still await backend persistence. Later AI completion cannot override this output terminal. |
| `saved` | Authoritative BL assistant message for the same conversation/logical send. Apply durable status/content, reconcile exactly once, stop consuming the stream. |
| `[DONE]` data sentinel | Legacy transport marker only. Never substitutes for `saved` on the persisted conversation path. |

`saved` statuses remain `pending`, `completed`, `failed`, `interrupted`. A replayed
pending record is not relabelled completed. The contract requires `accepted` before
answer events or `saved`. A source/event with an explicit stale attempt is discarded;
legacy events without IDs retain existing server enforcement. It does not infer an
attempt from an untrusted source event. UI request-generation guards remain active
on conversation switches and cancellation.

The final buffered frame is applied before reader cleanup. Once a valid `saved`
message arrives, further duplicate or late events do not overwrite durable state.
An AI completion followed by EOF without `saved` results in an unconfirmed/offline
UI state with retained text and an explicit refresh-before-retry message.

## Verification and reusable fixture

Single source fixture: [chat-stream-v1.json](contracts/chat-stream-v1.json), read by
the UI, Java relay test and Python producer test. Test workspaces must contain the
existing sibling UI/BL/AI repositories; there are no copied drifting fixtures.

- UI contract runner: **191 split-boundary cases**, single-byte UTF-8 input,
  multiline CRLF/bare-CR/default events, malformed/truncated inputs, frame/response
  bounds, legacy/canonical envelopes, version mismatch, logical identity mismatch,
  stale/missing source attempt, duplicate terminals, error/partial flow and reader
  cancellation/lock release passed.
- UI browser suite: **16 groups passed**, including real local HTTP incremental
  transport, new missing-save/malformed/unknown-event cases, multiline Unicode and
  stale sources. Auth/API/persistence are controlled fixtures, not real issuer/MySQL.
- Java `AiServiceImplTest` and `ConversationIntegrationTest`: **37 tests passed**.
  Covers the shared bytewise HTTP fixture, explicit bare-CR rejection, accepted
  attempt identity, stored-answer replay and existing persistence/retry/isolation.
  Database/JWT/provider are test fixtures.
- Python `tests.test_research_streaming`: **24 tests passed**. The real SSE producer
  emits the shared Unicode answer and terminal attempt/reply through a controlled
  model; existing document/consent/cancellation checks pass.
- UI TypeScript/production build/bundle budget, presentation contracts and lint pass.
  Lint retains 13 pre-existing warnings. Initial static JS remains 495.5 kB within
  the existing 550 kB budget. Whitespace checks pass across changed repositories.

The initial broad relay fixture failed on bare CR. Inspection identified Spring's
line-separator rejection; production LF/CRLF was not changed. The fixture now
explicitly asserts that rejection in Java and successful parsing in the browser.
No passing score implies that unsupported upstream bare-CR streams became supported.

Reproduce from the respective repository roots:

```powershell
rtk proxy npm.cmd run test:chat-stream
rtk proxy npm.cmd run test:chat-browser
rtk proxy npm.cmd run test:ai-presentation
rtk proxy npm.cmd run build
rtk proxy npm.cmd run lint
rtk proxy .\mvnw.cmd "-Dtest=AiServiceImplTest,ConversationIntegrationTest" test
rtk proxy .venv\Scripts\python.exe -m unittest tests.test_research_streaming
```

No model, retrieval policy, schema or deployment changed. Authenticated live-provider,
real-account persistence and physical-device acceptance are still excluded. Next in
the existing sequential backlog is **AIF-007** (telemetry), followed by PERF-001;
the broader AIF-003 controller extraction follows those baselines.
