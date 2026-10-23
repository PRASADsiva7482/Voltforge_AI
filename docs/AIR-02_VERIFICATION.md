# AIR-02: web search and public-page retrieval

Date: 2026-10-22.

**Implementation and automated verification finished. Live provider acceptance remains pending. AIR-02 stays `in_progress`; chat research is not enabled.**

## Implemented behavior

- Added a Brave Web Search adapter using the existing pinned HTTPCore/AnyIO dependencies. No second model service, browser application, vector database or new package is required.
- Added the internal service-authenticated `POST /voltForge-ai/api/v1/model/research/search` endpoint. It accepts the AIR-01 research contract only. No arbitrary URL command, private project snapshot, user memory or chat history is accepted as an additional input.
- Explicit modes require a reviewed public query and a real boolean consent value. Queries are limited to 500 characters and 75 words. Off makes no external request. Unsupported document retrieval still fails closed.
- Only the reviewed query is sent to the fixed Brave endpoint. Web-only results and disabled spelling correction preserve technical part identifiers. Result normalization deduplicates URLs and removes unsupported destinations. Known original technical publishers are preferred; that ranking does not certify their claims.
- Provider credentials are held as a Pydantic secret, taken from configuration/environment and sent only to the fixed API origin/path. Redirects cannot forward credentials to another host. No full query, source text or credential is logged by this feature.
- HTTP connections resolve DNS once, reject any non-public address in the answer, and connect to the validated IP. HTTPCore retains the original hostname for TLS/SNI and certificate verification. Local/private/link-local/multicast/reserved addresses, IPv4-mapped IPv6 and translation/tunnel ranges are blocked. Hostname validation alone is not treated as sufficient protection against DNS rebinding.
- Fetches allow public HTTPS on port 443 only, without URL credentials or environment proxies. Redirects are rejected, not followed. JavaScript is not executed. Compressed responses are rejected before decompression. Unsupported types, oversized declared bodies and oversized streamed/chunked bodies fail explicitly.
- Source extraction creates bounded text excerpts with URL, title, retrieval timestamp and hash. A failed page fetch retains a search snippet identified as a snippet; it never becomes a claimed fetched page. Partial/no-result responses are explicit. All excerpts remain `untrusted_reference_only`; the model has not consumed them yet.
- Scoped connection pools close on success, errors, deadline and cancellation. HTTP client disconnect cancels the retrieval task. Requests have concurrency, rate, per-page and overall limits. At most one retry is allowed for transient upstream/network failure; retries also reserve a rate slot. Provider 429s trigger a 60-second local cooldown.

## Default operational limits

| Setting | Default |
| --- | --- |
| `VOLTFORGE_AI_WEB_SEARCH_ENABLED` | `false` |
| `VOLTFORGE_AI_WEB_TIMEOUT_SECONDS` | 20 seconds overall |
| `VOLTFORGE_AI_WEB_PAGE_TIMEOUT_SECONDS` | 5 seconds per HTTP request, including DNS/connect/read |
| `VOLTFORGE_AI_WEB_MAX_RESULTS` | 5 results |
| `VOLTFORGE_AI_WEB_MAX_PAGES` | 3 fetched pages |
| `VOLTFORGE_AI_WEB_MAX_RESPONSE_BYTES` | 524,288 bytes per response |
| `VOLTFORGE_AI_WEB_MAX_CONCURRENT` | 2 attempts per process |
| `VOLTFORGE_AI_WEB_MIN_SEARCH_INTERVAL_SECONDS` | 1.1 seconds |

These are bounded configuration values, not promises of provider quota or measured production capacity. Multi-process deployments sharing a provider key need a coordinated quota policy before release; AIR-07 owns deployment/performance acceptance.

## Credential setup and acceptance command

1. Provision a Brave Web Search API credential with an appropriate subscription/quota using the provider's account. Bootstrap can install dependencies, but cannot create a subscription or credential.
2. Supply `VOLTFORGE_AI_BRAVE_API_KEY` to the AI process through your private environment/deployment secret mechanism. `BRAVE_API_KEY` is an alias. Do not commit the key to `config.json` or paste it into chat. This application's settings loader does not automatically load `.env`.
3. Set `VOLTFORGE_AI_WEB_SEARCH_ENABLED=true` in the same environment. Restart the AI process to load environment changes.
4. From the AI repository, run:

   ```powershell
   rtk proxy .venv\Scripts\python.exe scripts\verify_web_research.py --live-search
   ```

The script uses a fixed public Arduino query and page. It prints only status and size/count measurements, not credential values or source content. Exit 0 requires provider results and at least one fetched source page. Exit 2 means provider verification remains incomplete (the RTK wrapper may surface that as a generic nonzero exit); exit 1 means the public-page check failed.

`GET /voltForge-ai/api/v1/model/research/capabilities` reports adapter configuration separately from chat availability. Configuration is not a health probe: `liveConnectivity` remains `not_checked` instead of inferring a valid subscription from a nonempty key. Existing BL capability proxying remains compatible. Chat research remains unavailable until AIR-03 integrates retrieval, streaming and persisted citations.

## Verification actually executed

- `.venv\Scripts\python.exe -m unittest discover -s tests`: **64 passed**.
- `.venv\Scripts\python.exe -m unittest test_health_status test_security_va`: **5 passed**.
- `.venv\Scripts\python.exe -m compileall -q api engine scripts`: passed.
- `git diff --check`: passed.
- `scripts/verify_web_research.py --live-search`: real HTTPS retrieval and extraction of the official Arduino Uno page passed: **415,973 downloaded bytes and 2,173 excerpt characters**. Provider phase reported `WEB_DISABLED` and was not performed. Loaded settings were then checked without displaying values: enabled=false, provider credential configured=false.

The 17 new tests cover URL and DNS restrictions, mixed public/private DNS answers, IP pinning with original-host TLS verification, redirect/credential isolation, response and decompression restrictions, chunked body limits, extraction bounds, source provenance, publisher preference, no results, malformed provider JSON, errors/retries/cooldown, disabled/no-key no-network behavior, concurrency, total deadline, cancellation, HTTP disconnect, service authentication, strict consent and endpoint payload constraints.

Testing caught a disconnect-monitor issue: polling `Request.is_disconnected()` could consume task cancellation through its internal cancel scope. The endpoint now waits directly for the ASGI disconnect event after request-body parsing; the disconnect regression test and full suite pass.

Existing test-client deprecation and Python `nul` resource warnings remain. No new Python lint command is configured in this repository. UI/BL files were not changed in AIR-02, so their builds/browser suites were not rerun or claimed as newly verified.

## Changed files

- `engine/web_research.py`: provider, transport, extraction, limits and retrieval orchestration.
- `api/research.py`: strict consent, provider query limit and accurate capabilities.
- `api/routes.py`: private search endpoint and disconnect cancellation.
- `config.py`: bounded settings and secret provider credential.
- `tests/test_web_research.py`: 17 targeted regression tests.
- `scripts/verify_web_research.py`: repeatable live acceptance probe.
- `README.md`, plan, status tracker and this report.

No database writes, migrations, model downloads or additional dependencies were required. Existing worktree changes were preserved. No running development service was restarted by this step.

## Remaining boundaries

- **AIR-02 completion gate:** real provider search with a configured key/quota, including normalization and fetched-page evidence. This gate has not passed.
- **AIR-03:** UI consent/query controls, evidence token budgeting, model generation, real streaming stages, citations and conversation persistence. The new endpoint is not a substitute for that end-to-end integration.
- Redirected, compressed, PDF, login/paywall and JavaScript-only pages are currently unsupported. HTML/plain text is decoded as UTF-8 with replacement for unsupported bytes; arbitrary charset handling and PDF ingestion are not claimed.
- Retrieval alone does not make factual claims reliable. Evidence must remain untrusted data in AIR-03, and citation/engineering quality must be evaluated in AIR-06/07.

## Primary implementation references

- [Brave Web Search API](https://api-dashboard.search.brave.com/api-reference/web/search/get): request parameters, provider authentication, response contract and query limits.
- [Brave authentication](https://api-dashboard.search.brave.com/documentation/guides/authentication): separately provisioned subscription key.
- [HTTPCore network backends](https://www.encode.io/httpcore/network-backends/) and [async streaming](https://www.encode.io/httpcore/async/): custom transport and scoped stream handling.
- [OWASP SSRF prevention](https://cheatsheetseries.owasp.org/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html): public destination validation and redirect/DNS protections.
