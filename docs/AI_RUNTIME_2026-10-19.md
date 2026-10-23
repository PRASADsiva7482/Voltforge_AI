# AI chat and direct local runtime — 19 October 2026

## Implemented

Normal chat messages no longer render source/model/confidence labels, grounding/citation panels, internal activity steps, or repeated engineering disclaimers above and below messages. The UI retains readable Markdown, code copying, project-context consent, memory controls, real errors/retry, Stop while generating, and review of actionable circuit proposals. Transport metadata remains compatible with the existing backend; hiding presentation metadata does not remove saved conversations or authorization.

`python bootstrap.py --verify --run` prepares the environment, downloads and verifies the model, checks real inference, and starts the existing service. It uses the model directly through native `llama-cpp-python`; no separate model-hosting app is required. Windows launchers call the same bootstrap. Model download remains automatic on service startup if the configured pinned artifact is missing.

The selected default remains [official Qwen2.5-Coder-1.5B-Instruct Q4_K_M](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/blob/main/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf). The inspected machine has an i5-8265U with four physical cores and integrated graphics, so a compact CPU model is a practical default. This is not a comparative model-quality benchmark or a claim of ChatGPT-equivalent reasoning.

## Verified defects and fixes

| Defect | Change |
| --- | --- |
| Synchronous native token iteration blocked the async web path | Inference now runs in a worker with bounded token delivery; health requests remain responsive. |
| Concurrent requests could enter one native model instance | One lock covers synchronous and streaming inference, with bounded admission, queue timeout, and busy feedback. |
| Disconnect/cancellation did not reliably release generation resources | Cancellation reaches native stopping criteria; generators close explicitly and retain the model lock until evaluation ends. |
| Histories and duplicated project payloads could exceed the model context | Real tokenizer budgeting reserves response space, prioritizes the current question and relevant excerpts, and removes duplicated circuit/code data. |
| A proposed cancellation argument was incompatible with the installed library | Real inference testing caught it. The implementation uses the supported completion API with explicitly tokenized Qwen ChatML. User text is encoded without special role tokens. |
| Downloads relied on size checks and could replace working weights before validation | Exact revision, byte size and SHA-256 are pinned; downloads resume/retry in a temporary file and promote atomically after validation. |
| Missing model configuration could silently select another local GGUF | Only the configured artifact is loaded; custom models require an explicit digest and compatible ChatML. |
| Initialization/bootstrap was fragile across Python versions | A single script selects/installs Python 3.12, creates a venv, installs pinned native dependencies, and verifies the model. NumPy's actual Python requirement was checked. |
| Importing the service could re-execute another Python interpreter | Removed import-time process replacement; bootstrap owns environment selection. |
| Missing service credentials permitted access | Missing credentials fail closed. Existing gateway credentials were moved from tracked JSON into an ignored private token file, preserving the configured value. Validation errors hide input values. |
| The body-size setting was not enforced | ASGI middleware bounds bodies before JSON parsing, including chunked requests. |
| Empty output could look like successful completion | Empty generation, interruptions, timeouts, unavailable models and busy conditions produce explicit errors. |
| Wiring helpers guessed board pins, 5V supplies, motor wiring and a resistor named `r1` | Automatic suggestions now require actual board/component IDs and explicit I2C pin definitions. Existing endpoint connections are not duplicated. Unsupported routing is unavailable. |
| Coverage marked all catalog boards verified without records; response shape lacked UI summary fields | Coverage uses the existing catalog, reports uncurated status honestly, and supplies the summary/entries shape consumed by the UI. |
| OLED was matched as a discrete LED by a substring heuristic | Discrete LED checks now use the appropriate type prefix. Heuristics remain explicitly limited; missing wire IDs cannot become fabricated removal proposals. |
| Backend debug codecs could log private chat text | Restricted codec/web logging and verified the relevant loggers are not DEBUG-enabled during the real gateway/database smoke test. |
| Dockerfile copied removed legacy modules | Replaced those paths with the active service layout and non-root execution. Container execution remains unverified. |

## Changed files

In **Voltforge_AI**:

- `engine/llm.py`, new `engine/runtime.py`: token budgeting, ChatML formatting, context assembly, serialized native execution, cancellation and failure behavior.
- `engine/electronics.py`: evidence-limited routing, truthful coverage, corrected LED heuristic.
- `api/routes.py`, `api/schemas.py`, `api/security.py`, new `api/limits.py`, `main.py`: validation, request bounds, authenticated lifecycle, explicit stream cleanup/errors and truthful readiness.
- `bootstrap.py`, `download_model.py`, `model_manifest.json`, `requirements.txt`, `config.py`, `config.json`, `run.bat`, `download_model.bat`: repeatable direct setup, pinned dependencies/artifact, private configuration and tested runtime defaults.
- `Dockerfile`, `.dockerignore`, `.gitignore`: active module packaging and exclusion of credentials/artifacts/environments.
- `tests/test_chat_integrity.py`, `tests/test_runtime.py`, `tests/test_http_limits.py`, `tests/test_bootstrap.py`, `test_health_status.py`, `test_security_va.py`: regression coverage and isolation of test configuration changes.
- `scripts/verify_runtime.py`, `scripts/verify_service.py`, `scripts/verify_download.py`, `README.md`, this report: reproducible verification and operating instructions.

In **Voltforge_UI**, this pass changed `src/features/ai/AiChatPanel.tsx`, `src/features/ai/chatWorkspace.css`, and `scripts/run-chat-browser-tests.mjs` for the uncluttered transcript and its browser assertions. Other memory/context changes already present in the working tree were preserved.

In **Voltforge_BL**, this pass changed `src/main/resources/application.yml` and `src/test/java/in/voltforge/api/DevelopmentConversationSmokeTest.java` for privacy-safe logging and its runtime assertions. Existing conversation and memory services were reused.

## Database and running services

The configured **dev/development MySQL** environment was checked before service restarts. No pending persisted responses were present. No schema changes or migrations were required for this pass, and no existing conversations were changed or deleted.

The opt-in `DevelopmentConversationSmokeTest` used a randomly identified synthetic account, persisted an approved preference, created a conversation, streamed a real model response through the existing Java gateway, reopened its two saved messages, searched the title, and verified idempotent replay. Its `finally` block removed only its synthetic conversation, preference and account. The test passed, including the logging privacy assertions.

The configured AI service on port 2002 and application backend on port 2001 were restarted with the updated runtime/configuration. Authenticated AI health confirms the model is loaded. Existing gateway credentials still match.

## Checks actually executed

| Check | Result |
| --- | --- |
| Python `unittest discover -s tests` | 36 passed. Covers context/role isolation, missing/empty/interrupted replies, wiring safety, body/auth limits, concurrent inference, cancellation, timeouts, context limits, download corruption/resume, bootstrap/token migration and privacy. |
| Python `test_health_status` + `test_security_va` | 5 passed. |
| Clean environment bootstrap `.venv-migration --verify` | Passed; installed Python 3.12/native CPU wheel/dependencies and generated an actual response. |
| Repeated bootstrap in the configured environment | Passed; verified artifact reused; no model redownload required. |
| Full direct download of separate temporary artifact | 1,117,320,768 bytes downloaded and SHA-256 verified in 359.2 seconds; test copy removed without replacing the active model. |
| Real native inference smoke checks | Passed in both configured and clean environments. Model load 3.50–4.58 seconds; short cold response 13.19–13.66 seconds in these runs. |
| Isolated real HTTP/model test | Passed auth, 413 body limit, concurrent busy response, client disconnect cancellation, and a subsequent completed response. Maximum measured health latency during inference: 0.016 seconds; subsequent warm response: 1.55 seconds. |
| Java `AiControllerTest`, `AiServiceImplTest`, `AiRequestAdmissionTest`, `AiTelemetryTest`, `ConversationIntegrationTest` | 35 passed, zero failures/errors. |
| Java `DevelopmentConversationSmokeTest` | Passed against the configured dev database and actual local model; persisted assistant status `completed`. |
| UI `npm run test:chat-browser` | Passed conversation persistence/restoration fixtures, memory operations, consent/project association, safe Markdown/copy, failure/retry, history operations, scroll preservation, desktop/mobile and dark/light layouts; transcript metadata absence asserted. Browser tests use API fixtures; the real backend/database/model path was verified separately. |
| UI `npm run build` | Passed TypeScript/Vite build and bundle budgets. |
| UI `npm run lint` | Passed with 13 pre-existing warnings elsewhere in the UI; no lint errors. |
| Python compileall, pip dependency check, Git whitespace checks | Passed. |

## Remaining limits

- Docker is not installed here, so the container build and Linux/GPU execution have not been verified. The tested deployment is Windows x64, Python 3.12, CPU.
- The model is a compact 1.5B model. Engineering answers and generated code remain suggestions requiring appropriate compiler/simulator checks. No electrical verification corpus, full netlist solver, or universal automatic router was introduced.
- Native cancellation and deadlines are cooperative at evaluation/token boundaries. The web event loop stays responsive, but a stalled native call cannot be forcibly terminated inside the process safely.
- Context is bounded to 4,096 tokens by default; larger project/history inputs are excerpted. Arbitrary user facts are not persistent memory; existing allowlisted preferences remain in the backend.
- The Python HTTP test stack reports a Starlette/httpx deprecation notice. Tests pass; migrating that test-client dependency is a future compatibility task. Java test tooling also emits existing JVM/Mockito warnings.
- First installation requires a Python launcher and network access. Deployment authentication and the backend/database remain environment configuration; bootstrap cannot invent credentials or migrate an unrelated machine's database.

No claim is made that every possible AI quality, deployment, or hardware issue has been eliminated. The verified defects above are repaired and covered by the recorded checks.
