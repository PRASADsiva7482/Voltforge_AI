# Repeated incomplete responses after restarting — 19 October 2026

## Confirmed cause

The development AI listener was running under global Python **3.14**, launched with `main.py`. The native model dependency is installed and tested in the project's Python **3.12** environment. Authenticated live health returned `ready: false` and `MODEL_NOT_LOADED`. Read-only inspection of the configured development database showed the two most recent assistant failures recorded as `AI_MODEL_UNAVAILABLE`, with empty response content.

The prior verification had tested the managed runtime, but did not cover restarting `main.py` with the global Python interpreter. That launch gap allowed the same failure to recur after every restart. No model swap or new provider was needed.

A second defect obscured the diagnosis: both the UI's saved-event handler and history restoration replaced every failed response with the generic `Response incomplete` presentation. The saved error code was already present in the backend, but was discarded by presentation code.

## Changes

- **AI — `bootstrap.py`, `main.py`:** Executing `python main.py` now validates the interpreter and pinned native dependencies before importing the API. An incompatible global interpreter delegates to the existing bootstrap, which starts the managed Python 3.12 runtime. A guard prevents repeated setup/restart loops if native initialization remains broken. Imports remain side-effect free. A missing model now fails service startup explicitly instead of accepting guaranteed-to-fail chat requests.
- **UI — `aiPresentation.ts`, `AiChatPanel.tsx`:** One shared function presents persisted response state during streaming and after reopening. The actual safe error reason survives the saved event and refresh. Empty failed replies are not labeled partial output. Successful retries clear failure presentation and preserve message IDs. Errors remain operational messages rather than normal assistant text or extra model metadata.
- **BL — `ConversationStream.java`, `AiServiceImpl.java`:** Interrupted/error terminations always retain a safe error code. Gateway failures distinguish authentication, unavailable service, busy service, and rejected request size without exposing upstream bodies or credentials. Existing ownership, persistence, and retry protections are retained.
- **Tests:** Added executable-launch/startup guards, saved-error rendering/reload/retry browser checks, gateway-error mapping, persisted error/recovery assertions, and a real HTTP retry against the existing native model and development database.

## Verification executed

| Check | Result |
| --- | --- |
| Start through the same global `python main.py` path | Passed. The resulting listener uses managed Python 3.12 and authenticated health reports model ready. |
| Python regression suite | 40 passed. |
| Python health/security tests | 5 passed. |
| Backend AI/conversation regression suites | 37 passed. |
| Opt-in development DB/native-model smoke tests | 2 passed. One verifies the existing gateway flow. The added test uses authenticated HTTP, retries a synthetic persisted `AI_MODEL_UNAVAILABLE` response, receives real generated output, reopens it over HTTP, and verifies idempotent replay with exactly two messages. Only the test server's JWT decoder is mocked; the database, HTTP handlers, persistence, AI gateway, and local model are real. |
| Chat browser suite | Passed, including model-unavailable error → saved failure → page reload → same-ID retry → completed response → reload without failure UI. Browser tests use durable API fixtures; real HTTP/database/model behavior is tested separately above. |
| AI presentation contract tests | Passed. |
| UI TypeScript/production build/bundle budgets | Passed. |
| UI lint | Passed with 13 existing unrelated warnings. |

The AI and backend development services were restarted with the changes. No database schema changes were needed. Existing user conversations were not altered; the live tests cleaned up only their own synthetic records. Refresh the UI to load the updated presentation, then retry the latest failed response in an affected conversation. Historical failures are not silently converted into successful replies.
