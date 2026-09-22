# VoltForge AI security and privacy threat model

Status: approved for VFAI-030  
Approved: 2026-08-31  
Scope: Voltforge_AI, the Spring AI gateway in Voltforge_BL, and the
proposal/review boundary in Voltforge_UI

## Security decision

VoltForge AI is a project-owned local model service. It has no hosted model
provider, hosted inference fallback, or remote training loop. Internet access,
when explicitly enabled, is a separate evidence tool and is never a generation
dependency.

The Python service is a private backend component. All model routes, including
health, diagnostics, generation, export, simulation, memory, feedback, and
retrieval routes, inherit the service-token dependency. Development and test
may omit the token for local compatibility. Production configuration fails
closed unless it has:

- VOLTFORGE_AI_ENVIRONMENT=production
- a 32-character-or-longer VOLTFORGE_AI_API_TOKEN
- explicit VOLTFORGE_AI_ALLOWED_ORIGINS values
- a memory database path beneath VOLTFORGE_AI_RUNTIME_DIRECTORY

The Spring backend is the browser-facing trust boundary. Its security filter
chain requires an authenticated JWT for non-public API requests. Chat binds
the JWT subject, checks project access and the current project revision, and
forwards only server-derived identity headers to the Python service. The UI
never receives or supplies the private service token.

## Assets and owners

| Asset | Security property | Owner |
| --- | --- | --- |
| Model weights, tokenizer, manifest, registry, signing trust store | Authenticity, integrity, compatible loading | AI runtime owner |
| Project IDs, revisions, canvas/code context | Confidentiality and scope isolation | Backend owner |
| Chat prompts, responses, memory, feedback | Data minimization, bounded retention, no hidden reasoning | Privacy/data owner |
| Retrieval query and evidence | Fixed egress, provenance, untrusted-content labeling | Retrieval owner |
| Generated circuit/code proposals | No implicit mutation; safety evidence retained | UI and engineering-tools owners |
| Service token and database credentials | Confidentiality and rotation | Operations owner |
| Availability and resource budgets | Bounded request, model, stream, and retrieval work | Operations and backend owners |

## Trust boundaries

1. **Browser to Spring**: the browser is untrusted. JWT authentication,
   project authorization, request validation, and revision checks happen at
   the backend. Client-provided identity and memory are not authoritative.
2. **Spring to Python**: the backend is the only approved caller. A private
   X-Voltforge-AI-Token is injected by the server-side WebClient. Python
   validates the token before route execution and derives memory scope from
   gateway headers.
3. **Request context to local reasoning**: prompt, canvas, code, diagnostics,
   memory, and retrieved text are untrusted inputs. Deterministic engineering
   tools remain authoritative; model or retrieved text cannot override a
   critical finding.
4. **Python to local files**: model and tokenizer files are read from explicit
   manifests. Memory is SQLite under the configured runtime directory.
5. **Python to internet**: only the checked DuckDuckGo Instant Answer JSON
   endpoint is allowed. Returned titles/snippets/citations are untrusted
   evidence, not instructions or training data.
6. **Proposal to editor mutation**: the UI shows a typed diff and requires
   per-item approval, a matching project revision, one atomic commit, and
   guarded undo.
7. **Training workspace to release**: governed corpus and held-out evaluation
   data are separate. A live request, memory entry, feedback item, or web
   response cannot update model weights.

## Threat register

| ID | Threat and impact | Current mitigations | Residual risk / owner |
| --- | --- | --- | --- |
| T-01 | A caller reaches the Python service directly and invokes generation, export, health, or memory routes. This could consume resources or inspect service state. | Router-wide require_service_token; constant-time comparison; production token is mandatory and validated at startup; loopback default; production docs/OpenAPI are disabled; Spring SecurityConfig requires JWT for browser APIs. | A deployment can still misconfigure network ACLs. Run the service account on loopback/private network and firewall the port. Operations, VFAI-035. |
| T-02 | A user supplies another project ID, revision, context object, or memory identity and receives cross-project data. | Spring canAccessProject and current-revision checks; recursive structured-context project/revision checks; server-derived identity headers; Python hashed user/project/session scopes; memory owner/project predicates and revision staleness filtering. | Stateless generation endpoints accept caller-supplied content because they do not read persisted project data. Future endpoints that read a project must use the same binding helper. Backend owner, VFAI-032. |
| T-03 | Prompt injection in user context, memory, or retrieved web content causes instruction following, secret disclosure, or unsafe actions. | Memory rejects instruction-like content, hidden-reasoning terms, snapshots, and secret-only content; retrieval sanitizes markup/control characters and rejects injection patterns; evidence is labeled untrusted; engineering checks are authoritative; proposal apply is explicit and typed. | A local model can still produce a persuasive unsafe explanation. Safety tools and human review remain required. AI and engineering-tools owners, VFAI-014/VFAI-019. |
| T-04 | A modified or substituted model artifact executes code, lies about compatibility, or causes resource exhaustion. | Explicit approved registry; schema-2 Ed25519 registry verification; per-file SHA-256; artifact ID/runtime/release/lineage checks; path traversal rejection; NPZ allow_pickle=False; Gen1 checkpoint weights_only=True; architecture, tensor name/shape/dtype, parameter, tokenizer, and allocation checks; no random initialization on missing artifacts. GHSA-63cw-57p8-fm3p affects the pinned Torch runtime, so ModelRuntimeService rejects checkpoint-backed serving before native import under versions below 2.10.0. | The pinned runtime is limited to trusted local training/testing until a patched Windows wheel passes VFAI-FU-003. A correctly signed artifact may still contain poor learned behavior or consume its declared resource budget. Release evaluation, signing-key custody, and OS resource limits remain required. AI runtime and operations, VFAI-FU-003/VFAI-033. |
| T-05 | Unsafe deserialization or path traversal in JSON, tokenizer, checkpoint, or memory files. | JSON-only manifests/configs; safe NPZ; weights-only PyTorch state-dict loading only for trusted local work under the current pin; affected-runtime serving block; relative manifest paths must remain beneath the approved root; SQLite parameterized statements; bounded Pydantic request models. | PyTorch 2.8.0 remains affected by GHSA-63cw-57p8-fm3p. Do not load untrusted checkpoints or activate neural serving until a verified >=2.10.0 runtime replaces it. Legacy retired experiment files remain quarantined. AI runtime owner, VFAI-FU-003/VFAI-033. |
| T-06 | Training-data poisoning, evaluation leakage, or private project data enters a future model. | Source/shard provenance and checksums; held-out collision checks; governed corpus admission; web policy forbids training use; memory metadata declares trainingUseAllowed=false; live chat has no weight-update path; raw conversation persistence is disabled in the active chat path. | The legacy optional DatabaseManager contains retired raw persistence methods but no request handler calls them; remove or isolate that code before enabling any feedback/retraining pipeline. Data owner, VFAI-034. |
| T-07 | SSRF reaches localhost, cloud metadata, private IPs, an attacker-controlled host, or an unsafe redirect. | Caller can submit only bounded search text and result count, never a URL; exact provider/domain/path/query allowlists; DNS answers must be global public addresses; HTTPS/port checks; no environment proxies, redirects, retries, arbitrary result fetching, binary content, or oversized bodies. | DNS/provider availability and public provider behavior can change. Keep egress firewall allowlisting aligned with the checked policy. Retrieval owner, VFAI-031/VFAI-035. |
| T-08 | Resource exhaustion through oversized JSON, context, history, files, token output, streams, retrieval, or concurrent streams. | 2 MB request-body cap; bounded Pydantic fields and aggregate context; 15-second default/120-second maximum chat deadline; bounded retrieval response/cache; SSE event/delta/stream byte and count limits; cancellation registry; Spring per-user/per-project non-blocking stream admission and request-size policy. | There is no deployment-wide IP rate limiter or durable quota. Add centralized rate/capacity telemetry and abuse controls before exposing beyond a trusted local network. Operations/backend owner, VFAI-031/VFAI-032. |
| T-09 | Logs, errors, health responses, or feedback expose prompts, code, credentials, database details, or upstream exception text. | Request logs use IDs/outcomes; model startup logs identity only; backend errors log exception type without message/stack; user-visible backend fallbacks are generic; feedback accepts bounded metadata and returns storedRawContent=false; database health/error text is generic; telemetry error text is bounded and redacted. | Platform access logs and legacy optional persistence are outside this service’s complete retention control. Configure redaction, access control, and retention at the deployment boundary. Operations/data owner, VFAI-031/VFAI-034. |
| T-10 | Memory becomes a covert transcript or crosses user/project/session boundaries. | Explicit per-project enablement; authenticated gateway identity required; hashed scope keys; project/session predicates; bounded entry count/bytes/TTL; redaction; no raw content in public metadata; recent turns store topic/category summaries only; stale revisions excluded from model context; clear/delete/correction controls. | Users can intentionally remember sensitive text that is not recognized by current patterns. Treat memory as user-controlled project data and provide deletion/retention controls. Data owner, VFAI-034/VFAI-035. |
| T-11 | Model-proposed circuit or firmware changes are silently or incorrectly applied. | VFAI-029 typed proposal planner; visible review items; allowlisted actions; invalid selection rejects the entire transaction; source revision/fingerprint checks; explicit approval; atomic commit; guarded undo; direct generated-code mutation removed; critical engineering findings cannot be overridden by the model. | A user can approve an unsafe suggestion, and a compromised browser can attempt UI actions. Backend authorization and deterministic validation must remain authoritative. UI/engineering-tools owners, VFAI-032. |
| T-12 | Service-token theft or credential leakage allows a trusted internal caller to impersonate the gateway. | Token is environment-only, never in browser DTOs or model context; constant-time comparison; backend-only WebClient header; production minimum length; generic errors; no credential in health output. | Rotation, revocation, secret-manager integration, and multi-instance key rollout are not yet implemented. Operations owner, VFAI-033/VFAI-035. |
| T-13 | Internet evidence silently becomes a generation or training dependency. | Retrieval is opt-in and local-first; disabled/degraded states are explicit; one provider policy is checksum-pinned; generation metadata declares networkRequired=false; failures return no inferred facts and local deterministic flow continues; web content is not admitted to corpus. | Provider outage reduces evidence quality, not local generation availability. Product owner must preserve this separation. |
| T-14 | Health/contract metadata is used as an information oracle. | Metadata is content-free and route protected in production; model health exposes IDs/checksums/state, not weights, prompts, or secrets; docs disabled in production. | Availability/status timing can still reveal coarse operational state to an authenticated caller. Acceptable for an authenticated gateway; do not expose the Python port publicly. Operations owner. |

## Control ownership and review gates

- **AI runtime owner**: artifact registry, checkpoint loading, tokenizer/model
  compatibility, request and context boundaries, local-only generation.
- **Backend owner**: JWT authentication, project authorization, revision binding,
  request admission, token injection, response/error boundary.
- **UI owner**: proposal diff, explicit selection, atomic apply, undo, and
  stale-editor rejection.
- **Data/privacy owner**: corpus provenance, memory retention, feedback
  admission, deletion, redaction, and training exclusion.
- **Operations owner**: OS account/ACLs, loopback or private firewalling,
  egress allowlist, secret storage/rotation, log retention, process/container
  limits, backups, and incident response.

VFAI-031 owns privacy-safe telemetry and rate/capacity visibility. VFAI-032
owns the cross-repository security pipeline. VFAI-033 owns signing-key and
service-token rotation plus release/rollback controls. VFAI-034 owns approved
feedback/retraining admission. VFAI-035 owns the deployment and incident
runbook. These are residual risks, not reasons to weaken the current local-only
boundary.

## Production deployment checklist

1. Set VOLTFORGE_AI_ENVIRONMENT=production in both services and inject the
   same random service token through a secret manager; never commit it.
2. Bind the Python service to loopback or a private interface, firewall the
   port so only the Spring service can reach it, and run it as a non-root
   account.
3. Give the AI account read access to the approved model registry/trust store
   and read/write access only to the runtime directory. Keep model artifacts
   immutable at runtime.
4. Keep internet retrieval disabled unless needed. If enabled, permit egress
   only to the approved provider and retain the no-proxy/no-redirect policy.
5. Keep database persistence disabled unless its schema, credentials, ACLs,
   retention, and raw-content policy are separately approved.
6. Supply explicit browser origins. Do not use *.
7. Treat health, logs, memory, feedback, web evidence, and generated proposals
   as security-relevant data. Apply platform log access and retention controls.
8. Before a model activation, verify the signed registry, artifact checksums,
   tokenizer lineage, release scorecard, resource budget, rollback artifact,
   full cross-repository test receipt, and that the installed PyTorch release is
   not blocked by a published checkpoint-loading advisory.

## Accepted residual risk

VFAI-030 closes the currently actionable application-boundary gaps and records
the ownership of residual deployment and lifecycle work. It does not claim
that a local language model is intrinsically safe, that a human cannot approve
an unsafe proposal, that a public internet provider is always available, or
that OS/container controls are supplied by Python code. Those risks are
explicitly assigned above and remain release gates for the later M6 backlog.
The pinned PyTorch 2.8.0 runtime is additionally accepted only for trusted local
training/testing; checkpoint-backed serving is blocked until VFAI-FU-003 proves
a patched Windows runtime through every import, correctness, and performance
gate.
