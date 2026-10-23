# VoltForge AI

FastAPI service running a GGUF model directly through `llama-cpp-python`. There is no Ollama, LM Studio, external inference server, or cloud-provider dependency. VoltForge_BL owns authentication, project permissions, durable conversations, and approved user preferences. This service performs inference and limited circuit checks; it does not maintain a second chat database.

## Start or move to another machine

From this directory, with Python available on PATH:

```powershell
python bootstrap.py --verify --run
```

Windows shortcuts: `run.bat` installs missing prerequisites and starts the service; `download_model.bat` performs setup and an inference check.

Launching `python main.py` from an IDE or terminal is also supported. The executable entry point checks Python and native dependencies before loading the API, then delegates to bootstrap when the global interpreter is incompatible. Importing `main:app` never starts another process. Direct ASGI launches must use the managed environment. Startup now fails explicitly if the model cannot load instead of accepting chat requests that can only fail.

Bootstrap creates `.venv`, installs the pinned dependencies and native CPU wheel, downloads the model directly from its pinned Hugging Face revision, verifies SHA-256, and prepares service authentication. If the launching Python is not 3.12, it finds or installs managed Python 3.12 using uv. The pinned NumPy dependency requires Python 3.12 or later; native wheel compatibility is verified specifically on 3.12. uv is only a Python installer, not an AI application. Initial setup needs internet access and roughly 3 GB of free space. Windows x64/Python 3.12 has been tested on this workstation.

On migration, copy the application source, not `.venv`. The script recreates the environment and downloads missing weights. Verified weights are reused offline on subsequent runs. Downloads resume through a `.part` file, retry transient failures, and replace the active artifact only after checksum verification. A failed installation exits nonzero. The script never drops databases or deletes chats.

The existing configured gateway token is migrated to the ignored `.service-token` file without changing its value. Fresh installations generate a private token. Set the matching token in **both** this service (`VOLTFORGE_AI_API_TOKEN`) and the existing VoltForge_BL configuration. Secrets must come from your deployment configuration or secret storage; they cannot be discovered automatically on a new machine. Do not commit `.service-token`, include it in images, or expose the inference service directly to browsers. Missing authentication fails closed.

Stop the service before upgrading dependencies or forcing a replacement download:

```powershell
python bootstrap.py --verify
python bootstrap.py --force --verify
```

`--force` is optional; ordinary setup already downloads missing/corrupt weights. For platforms without a matching native wheel, the script stops rather than silently attempting a long C++ build. `--source-build` explicitly opts into a native build and requires an installed C++ toolchain. GPU builds are not installed automatically.

## Model and resource defaults

The default is [Qwen2.5-Coder-1.5B-Instruct Q4_K_M](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/blob/main/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf), Apache-2.0, 1,117,320,768 bytes. `model_manifest.json` records its exact repository revision and SHA-256. This compact model is retained for the inspected four-core i5-8265U laptop with integrated graphics. It is not claimed to match larger hosted models in reasoning quality.

Defaults: CPU, four threads, 4,096-token context, up to 768 response tokens, one active inference plus two waiting requests, 15-second queue wait, 120-second generation deadline. Long questions may be rejected; history and project context are explicitly excerpted to reserve response space. Firmware, properties, wiring, and diagnostic excerpts come only from supplied data. Repeated context is removed before token budgeting.

Native inference runs outside the web event loop. A single model lock covers both streaming and ordinary generation. Disconnect/Stop propagates cancellation at native token boundaries, while the lock remains held until inference actually finishes. Native evaluation cannot be force-killed safely inside the process; a stalled native call can exceed the cooperative deadline. This is a bounded single-model service, not a multi-user GPU inference cluster.

`config.json` contains non-secret settings. Environment variables prefixed with `VOLTFORGE_AI_` override them, for example `VOLTFORGE_AI_N_THREADS`, `VOLTFORGE_AI_N_CTX`, `VOLTFORGE_AI_MAX_TOKENS`, `VOLTFORGE_AI_GENERATION_TIMEOUT_SECONDS`, `VOLTFORGE_AI_MAX_PENDING_REQUESTS`, and `VOLTFORGE_AI_AUTO_DOWNLOAD_MODEL`. Invalid settings fail explicitly. `VOLTFORGE_AI_ENV=production` disables API documentation. Default address: `127.0.0.1:2002`.

The prompt formatter is ChatML for the pinned Qwen model. A manually supplied alternative must use compatible ChatML, have a configured `model_sha256`, and pass inference checks. Arbitrary GGUF files are never selected as a fallback. For a different model family, implement and test its formatter rather than changing only the filename.

## API and integration

All `/voltForge-ai/api/v1/model/*` routes require the configured service token via `X-Voltforge-AI-Token` or Bearer authorization:

- `GET /health`: distinguishes service availability from an actually loaded model.
- `POST /chat`, `POST /chat/stream`: bounded inference; streaming retains start/token/proposal/complete/error compatibility with VoltForge_BL. Busy, timeout, interruption, oversized context, and empty replies are explicit failures.
- `POST /generate-code`: an unverified firmware suggestion, never a fabricated compilation result.
- `POST /suggest-wiring`: only explicit I2C signal endpoints from the supplied board/component pins. No guessed power rails, automatic motor wiring, fallback Uno, or nonexistent resistor. Unknown/ambiguous mappings return unavailable.
- `POST /validate-circuit`: limited pin-name/component heuristics, not a circuit solver or safety certification.
- `GET /hardware-coverage`, `GET /component-coverage`: inventory entries without unsupported verification claims.
- Legacy project-memory routes report unsupported functionality. Persistent allowlisted user preferences are managed by VoltForge_BL.

The backend validates conversation ownership and project-context consent before invoking this private service. Ordinary messages and history do not require a project. Do not bypass the backend with direct public access to this gateway. Request bodies are limited to 2 MB before JSON parsing. Private messages and credentials are not included in routine logs.

Chat UI changes are in the sibling Voltforge_UI project: source/model/confidence/grounding metadata is hidden from the normal transcript. Markdown, code copy, actionable proposal review, context controls, and actual failed/interrupted response recovery remain available.

## Verification

### Web retrieval (AIR-02)

The internal `POST /voltForge-ai/api/v1/model/research/search` route supports bounded Brave web search and public HTTPS text-page retrieval. It requires the existing private service token, an explicitly reviewed `publicQuery`, and boolean consent. It does not accept chat history, project context or arbitrary URL commands. `/chat/stream` now integrates retrieval with native model streaming and persisted source references through BL. **Research remains disabled in the configured development environment; live provider acceptance is pending.**

Supply `VOLTFORGE_AI_BRAVE_API_KEY` through your local/deployment environment (alias: `BRAVE_API_KEY`), and set `VOLTFORGE_AI_WEB_SEARCH_ENABLED=true` to permit retrieval. Keep credentials out of committed `config.json`; `.env` is not automatically loaded by this application's settings loader. Restart the AI process after changing its environment. The key/subscription must be provisioned separately; model/bootstrap migration cannot create it.

`GET /research/capabilities` (under the same model prefix) distinguishes adapter configuration from chat availability. It does not probe the provider or claim a working subscription based merely on a nonempty key.

In Chat options > Web and documents, On becomes available when the AI process has both the enable flag and provider credential. Each turn needs an explicitly approved public query; editing it revokes approval. Only the mode is remembered in the browser. Project-sharing consent is independent. Auto research remains unavailable pending later stages; synchronous `/chat` does not support research. Source references identify supplied evidence, not verified engineering claims. See [AIR-03 implementation and verification](docs/AIR-03_VERIFICATION.md).

Chat options > **Reference documents** supports PDF/TXT/Markdown uploads, immutable versions, explicit document selection and lexical retrieval through the existing backend database. Document-only chat works with web research Off; private files and selected IDs are not search-provider input. Project documents require the matching owned project and context sharing. The backend must apply `docs/schema/VF_DOCUMENT_001.sql` before starting the updated application (development helper: `scripts/document_schema.py --apply`). Follow [AIR-04 verification and limits](docs/AIR-04_VERIFICATION.md); no vector database or embedding model is required for this stage.

Retrieval adds at most 384 prompt tokens and three references by default, keeping CPU prefill bounded. `VOLTFORGE_AI_RESEARCH_SOURCE_TOKENS` (128–1200) and `VOLTFORGE_AI_RESEARCH_SOURCE_COUNT` (1–5) can adjust those limits after benchmarking. Increasing them may increase first-token latency. Transport keepalives maintain the stream while the model prepares; they are not generated answer tokens.

### Hybrid document retrieval (AIR-05)

The default backend document search now combines exact part-number matching, keyword scores and local MiniLM semantic embeddings. The 25 MB Q8 GGUF uses the existing `llama-cpp-python` runtime; no Ollama, hosted embedding API, vector database, ONNX or PyTorch dependency is needed. The pinned conversion and upstream Apache-2.0 model are recorded in `embedding_manifest.json`; `bootstrap.py` downloads/checks both chat and embedding artifacts automatically. `embedding_model.py --verify-only` verifies the embedding cache without downloading.

Apply BL `docs/schema/VF_EMBEDDING_001.sql` before starting the updated backend (`python scripts/embedding_schema.py --apply`). Vectors are derived from authorized, explicitly selected current document versions and stored in MySQL with their model identity. Missing vectors build in batches and survive retries; changed model identities rebuild without mixing incompatible vectors. A large cold selection can require another attempt after the bounded indexing window. Deleted source chunks cascade to their vectors; old document versions remain readable until deletion.

For an explicit lexical-only deployment, set `VOLTFORGE_AI_HYBRID_ENABLED=false` in BL and `VOLTFORGE_AI_EMBEDDINGS_ENABLED=false` in AI. Encoder failure is otherwise reported clearly, not silently disguised as semantic retrieval. Ordinary chat still works if the embedding model is unavailable. The private `GET/POST /voltForge-ai/api/v1/model/embeddings` endpoints expose readiness/encoding only to the authenticated backend. Do not expose them directly to browsers.

See [AIR-05 verification, benchmark and limits](docs/AIR-05_VERIFICATION.md). The encoder reads at most 256 word pieces per chunk and the first 2,000 query characters. Semantic similarity and citations do not verify electrical correctness. This initial English-focused benchmark is small; larger quality and load evaluation remains AIR-06/07.

Run `.venv\Scripts\python.exe scripts/verify_web_research.py --live-search` for a fixed public Arduino-page and provider-query check. Without a configured provider it reports the missing verification; it never prints credentials or page text. See [AIR-02 configuration, limits and evidence](docs/AIR-02_VERIFICATION.md).

### Existing runtime checks

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m unittest test_health_status test_security_va -v
.venv\Scripts\python.exe scripts/verify_runtime.py
.venv\Scripts\python.exe scripts/verify_service.py
.venv\Scripts\python.exe scripts/verify_download.py
```

`verify_runtime.py` loads the real model and generates a synthetic response. `verify_service.py` launches an isolated temporary service and checks authenticated HTTP, body limits, responsiveness during native generation, busy handling, disconnect cancellation, and subsequent generation. `verify_download.py` downloads a separate temporary copy of the complete model, checks its digest, and cleans up only that copy; it costs about 1.12 GB of network transfer. Tests do not print private chat contents or modify existing conversations.

See [the implementation and verification report](docs/AI_RUNTIME_2026-10-19.md) for executed results and limits. CPU timings are measurements from this laptop, not performance guarantees.

## Container deployment

The Dockerfile copies the active service modules and runs as a non-root user. Supply `VOLTFORGE_AI_API_TOKEN` through your deployment secret mechanism and mount a writable volume at `/app/model` for both chat and embedding artifacts. First startup downloads and verifies the pinned artifacts; persist the volume for offline restarts. Run one Uvicorn process per model allocation. The container build has not been executed here because Docker is not installed; GPU/container behavior is not verified.
