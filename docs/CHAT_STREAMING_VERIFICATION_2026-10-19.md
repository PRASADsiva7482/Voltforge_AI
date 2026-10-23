# Chat streaming verification — 2026-10-19

The current native model, AI SSE endpoint, backend gateway, persisted conversation
endpoint and browser already support progressive responses. No full-response
buffering was reproduced on the configured development path. A short response
arrived as 11 token events over 590 ms, which can look instantaneous. Model prompt
processing still happens before the first generated token.

## Corrections

- The UI's waiting indicator previously depended on the absence of internal
  `thought` metadata. The `start` event populated that metadata before generation,
  hiding the indicator. It now stays inside the one assistant bubble until actual
  text arrives. A cursor accompanies arriving text; normal responses retain the
  clean message-only presentation.
- Both backend streaming endpoints now return `Cache-Control: no-store,
  no-transform` and `X-Accel-Buffering: no`. The AI endpoint uses the same headers.
  These discourage caching/transformation and buffering by compatible proxies;
  they cannot override every external proxy's configuration.
- The real streaming paths remain incremental. No artificial typing delay, token
  splitting, replacement model, second chat service or database migration was added.

## Verification performed

- AI: 41 tests passed. A gated native generator proves its first token reaches the
  SSE iterator before generation is allowed to finish.
- Backend: 38 tests passed across conversation integration, AI controller/gateway,
  admission and telemetry. A real HTTP test receives two successive token events
  while completion is withheld, then verifies the final persisted message.
- Development database/model: two opt-in tests passed, using synthetic accounts
  removed afterward. A longer real native response delivered **214 token events**:
  first token at **1,188 ms**, token delivery spread over **13,046 ms**. The completed
  answer reopened from the database, and replay did not duplicate either message.
  This test uses the real configured AI and database with a test-only JWT decoder
  in a separate backend test server; production authentication is unchanged.
- Browser: ten chat verification groups passed. The new fixture uses actual HTTP
  writes and holds completion until Playwright observes partial text. It verifies
  progressive rendering, waiting indicator, duplicate-submit prevention, final
  reconciliation, reload, and Stop preserving partial text. Other existing groups
  cover history, retries, consent, Markdown and both themes/responsive layouts.
  Browser transport tests use a controlled provider fixture, not the native model.
- UI TypeScript/production build/bundle budgets passed. Lint passed with 13 existing
  warnings outside this change. Diff whitespace checks passed in all three repos.
- Configured development AI and backend services restarted with the changes;
  authenticated AI health confirmed the model ready.

Existing conversations were preserved. These timing measurements are observations
from this machine, not performance guarantees. Cold startup, long context, CPU load
and model generation speed can change time to the first token and response duration.
