# AIR-04: authorized documents and lexical retrieval

Date: 2026-10-22. Work spans the existing UI, BL and AI applications.

AIR-04 is complete for bounded, authorized uploads and lexical retrieval. Real development database/local-model acceptance passed after the fixes below. AIR-02/03 live public-search acceptance still requires a configured Brave credential. Document-only retrieval does not need that credential.

## Data and authorization

- Inspected the configured **dev/development MySQL `voltforge_db`** before writing. It contained conversation/message/preference storage and no document/source/chunk tables.
- Created and applied the versioned additive script `Voltforge_BL/docs/schema/VF_DOCUMENT_001.sql`. New tables: `ai_documents`, `ai_document_versions`, `ai_document_chunks`. No existing table/data was rewritten. Schema snapshot: `Voltforge_BL/target/document-schema-before-20261022123858.sql`.
- Verified insert, update, read and owner/document/version deletion cascades with synthetic records in a transaction, then rolled back. MySQL DDL commits implicitly, so schema changes themselves are not described as transactional.
- Owner and project foreign keys, version/checksum uniqueness, chunk ordering and ready-state constraints are enforced. New versions and their chunks publish in one transaction; extraction happens before the write transaction. A failed/interrupted upload cannot expose a partial index. Retry is checksum-idempotent; old versions remain readable by their owner until document deletion.
- All browser endpoints derive identity from the JWT. Reads, uploads, replacement, deletion and retrieval recheck ownership. Project documents require an owned matching project and explicit project-context sharing when used in chat. General references can be selected without a project.
- BL ignores browser-provided `authorizedDocumentEvidence`, loads authorized chunks itself, and sends them only over the existing private service-authenticated AI connection. Python validates document/version/page provenance, selected IDs and attempt binding. The public-query object sent to the web adapter never contains private document IDs or text.
- Source reads recheck current ownership/access. Deleted references become unavailable; previously saved chat excerpts remain in their owner’s transcript. Deleting a document does not silently erase conversations.

## Supported behavior and limits

- Chat options > **Reference documents** supports upload, title search, pagination, explicit selection, new versions and confirmed deletion. Source buttons open an authenticated excerpt viewer with version/page information. Ordinary chat keeps its existing uncluttered presentation.
- PDF, UTF-8 TXT and Markdown uploads: 2 MiB each, 80 PDF pages, 200,000 extracted characters, 256 chunks, 50 documents per account and 10 versions per document. Files are read as text; no JavaScript, attachments or links are executed. Empty/scanned/encrypted/restricted/invalid PDFs fail clearly. OCR and URL ingestion are not implemented in this stage.
- PDF parsing uses Apache PDFBox 3.0.8 in a child JVM with a 128 MiB heap, 96 MiB metaspace cap and a 20-second parent deadline; one parser is admitted per backend process. Temporary input/output files are removed after completion/failure. Packaging compatibility is checked separately below. These bounds are resource controls, not a claim that parsing arbitrary documents is risk-free.
- Page-aware chunks overlap by 160 characters. Retrieval ranks keyword matches in current versions of at most 10 selected authorized documents, with additional weight for identifiers containing digits. It returns at most five chunks. This is bounded lexical retrieval, not semantic similarity or guaranteed exact-part disambiguation; AIR-05 owns semantic/hybrid quality improvements.
- Web Off and document selection are independent. No matching text is an actionable error; the app does not pretend to have read useful evidence. Mixed retrieval interleaves document and web candidates before packing them. The default prompt budget is 384 additional tokens and at most three references, including record overhead; only references actually supplied to the model appear in response metadata. `research_source_tokens` and `research_source_count` are bounded settings. Document excerpts start near the strongest matching query term instead of unrelated page headers.
- Source IDs, document/version IDs, content hash, page and the supplied excerpt persist through the existing conversation metadata. Native generation uses the existing local model and SSE, not an additional AI service.
- A live test exposed a gateway idle timeout during long native prompt processing. The AI stream now sends SSE comment keepalives every ten seconds while awaiting retrieval/model output; these are not fabricated tokens or thinking messages. Disconnect cancels the pending generator and requests runtime cancellation. A native model call already in progress stops at its next token boundary; Python cannot safely kill that call mid-prefill.
- The next live attempt exceeded the existing three-minute backend limit. The resulting performance fix reduced the evidence budget from up to 1,200 tokens/five sources to 384 tokens/three sources and centered excerpts on query matches. Overall execution limits were not increased to hide the failure.

## Verification actually executed

- AI managed Python discovery: **72 tests passed**, including document-only no-search streaming, scope/attempt/source validation, page provenance, keepalive cancellation and the compact evidence-budget regression.
- AI health/security: **5 tests passed**.
- Packaged executable-JAR PDF worker: passed on the official Arduino document, **14 pages / 14,822 bytes of bounded extracted output**. This separately verifies Spring Boot's nested-JAR loader path.
- BL policy/gateway/conversation/document suite: **43 tests passed** before PDF child-process hardening; the five document integration tests passed again after hardening and the compact query-window change. Includes multipart HTTP upload, unauthorized access, two-owner isolation, project consent, checksum deduplication, immutable old-source reads, replacement/deletion, PDF page extraction and invalid/scanned inputs.
- UI TypeScript/production build/bundle budget: passed. Lint passed with 13 existing warnings outside this work.
- Browser chat suite: **14 groups passed**, including actual multipart encoding, upload/select with web off, source read/reload, safe text rendering, version replacement, mobile width and deletion confirmation. Browser APIs/auth are controlled fixtures; they do not represent a new Keycloak login verification.
- Downloaded the official Arduino UNO R3 PDF for a real development smoke test: **334,245 bytes**. The first native-stream attempt failed at the gateway’s idle timeout; it is not counted as a passing acceptance test. Retest after keepalives is recorded below.
- A subsequent run with compact evidence returned `AI_GENERATION_TIMEOUT`; the five document integration tests in that invocation passed, but the combined Maven invocation failed because of the native test. A later isolated general-chat runtime probe passed (4.48-second load, 13.64-second generation). Final document-specific acceptance is tracked separately below; a passing general-chat probe is not a substitute.
- Shared VPN database audit: all five current SQL scripts verified, 13 foreign-key relationships without orphans, schema visible from a fresh connection. Existing user/project/chat counts preserved; temporary reference documents removed. See [shared database verification](SHARED_DB_VERIFICATION.md).
- **Final live document acceptance passed:** `DevelopmentDocumentSmokeTest` used the real shared development MySQL connection, real BL-to-AI authenticated HTTP and the configured native GGUF. Extracted 14 pages / 16 chunks; received 65 native streaming chunks; persisted 2 messages and 3 version-bound references; reopened a page-4 excerpt and verified retry replay without a duplicate message. First generated chunk arrived after **25,676 ms**. Maven: 1 test, 0 failures, BUILD SUCCESS. Only synthetic account/document/conversation records were removed afterward. The test supplies a synthetic authenticated identity at the service boundary; it does not claim a new live Keycloak/browser login test.
- Earlier native timeouts under the then-current machine load remain documented above. The successful final run establishes this flow, not a concurrency or first-token latency guarantee. AIR-07 retains load/performance acceptance; AIR-05 owns retrieval-quality benchmarks.

## Files and APIs

- BL `ai/document/`: document/version/chunk entities, repositories, extractor/worker, service and controller. `POST /api/v1/ai/documents` (multipart), `GET /api/v1/ai/documents` (summaries/search/page), `DELETE /api/v1/ai/documents/{id}`, `GET /api/v1/ai/documents/{id}/versions/{version}/chunks/{chunk}`. Replacement uploads pass `documentId` and create immutable versions.
- BL `ConversationStore`, AI DTO/policy/gateway/controller, multipart limits and upload error handling; `DocumentIntegrationTest`, `DevelopmentDocumentSmokeTest`, existing policy test; SQL and `scripts/document_schema.py`.
- AI `api/research.py`, `api/schemas.py`, `api/routes.py`, `engine/llm.py`, document retrieval/contract tests.
- UI `DocumentDialog.tsx`, `documentApi.ts`, `ResearchSources.tsx`, `AiChatPanel.tsx`, citation types, workspace styles and browser scenarios.

No embedding model, vector database or additional provider was introduced. PDFBox is the only new runtime library for this stage. Existing user worktree changes were preserved.

## Primary references

- [Apache PDFBox 3 migration/loading API](https://pdfbox.apache.org/3.0/migration.html).
- [Apache PDFBox release downloads](https://downloads.apache.org/pdfbox/).
- [Official Arduino UNO R3 datasheet used for live verification](https://docs.arduino.cc/resources/datasheets/A000066-datasheet.pdf).
