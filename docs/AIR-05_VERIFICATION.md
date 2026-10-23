# AIR-05: local hybrid retrieval and repeatable embedding setup

Date: 22 October 2026. Scope: existing AI and BL applications. The existing document UI and conversation/source contracts are preserved.

Status: complete for the scoped local hybrid implementation and recorded acceptance. AIR-06/07 retain broader quality/load work; live internet search remains independently blocked on provider configuration.

## Model and benchmark decision

Selected MiniLM-L6-v2, 384 dimensions, Q8 GGUF, 25,008,064 bytes, Apache-2.0. Conversion publisher `second-state/All-MiniLM-L6-v2-Embedding-GGUF`, revision `544f204f2eaa2d71361ffc74d6df7170285b286a`, artifact SHA-256 `263215c3cadd6e16740741a7624ab4cbb6c8e777688bd5331ecfbf5681c2f8ed`. Upstream: `sentence-transformers/all-MiniLM-L6-v2`. The conversion's README is also pinned and checked. It uses the existing llama-cpp-python 0.2.90 runtime, two CPU threads and mean/L2-normalized embeddings. No additional runtime dependencies remain.

The initial ONNX candidate was not adopted: runtime 1.31.0 crashed on native import on this Windows host. Alternative runtime downloads also encountered timeouts. A later alternative installed, but the existing GGUF runtime already passed the native benchmark and avoided a second inference dependency. Experimental packages were removed; `pip check` passed. This is an observed host compatibility result, not a claim that ONNX generally fails.

Measured in a fresh Python 3.12 environment on this Windows machine, using [the recorded synthetic evaluation](AIR-05_BENCHMARK.json):

| Measure | Result |
| --- | --- |
| Exact identifier top result | 5/5 for lexical and hybrid |
| Paraphrase top result | Lexical 5/12; hybrid 11/12 |
| Hybrid expected result in top three | 17/17 |
| Encoder load | 0.297 s |
| Index 20 short reference fixtures | 0.234 s |
| Query encoding p95 | 16 ms |
| Batch of 32 long reference excerpts | 5.438 s |
| Peak standalone process working set | 88.13 MiB |

These are small synthetic retrieval fixtures, not engineering validation, independent quality guarantees or production latency targets. An initial reciprocal-rank blend diluted useful semantic matches; final fusion preserves complete part identifiers, leads with cosine similarity, and uses a bounded 0.02 keyword contribution. The production Java algorithm is checked against native-generated fixture vectors. Unrelated hotel/cake questions and an unknown part identifier are negative cases. Broader held-out quality and concurrent-load evaluations remain AIR-06/07.

## Data flow and limits

BL rechecks owner and matching project/context consent before loading selected current-version chunks. It sends only those chunks over its existing service-authenticated AI connection. There is no user-provided vector, model ID or owner ID in the browser API. The private encoder rejects missing authentication, mismatched model identity and oversized batches/text. It has one admitted encoder operation at a time; busy/unavailable states are explicit.

Vectors live in `ai_document_embeddings`, keyed uniquely by chunk/model ID. Model identity hashes the pinned manifest, including dimensions, pooling, token limit and runtime identity. Model changes therefore build new rows instead of interpreting old vectors as compatible. Source documents remain authoritative. Old model vectors remain harmless derived data tied to their source chunk; deleting a source cascades to all its vectors. No independent vector service or cross-user answer cache was added.

Missing vectors build in 16-chunk batches, committed separately from the chat transaction. Per-owner write locking prevents duplicate inserts. A bounded 20-second indexing window plus a maximum 10-second in-flight encoder call may return a clear retryable warming error for a large cold selection; completed batches survive and the next attempt resumes. No response is presented as generated when preparation fails. Selection is limited to the existing 10 documents / 2,560 chunks. Retrieval computes exact cosine scores over authorized rows, not an approximate global index.

The encoder uses at most 256 word pieces of each chunk and 2,000 query characters. Long chunk tails can therefore be missed semantically. Exact identifier matching considers the full chunk; identifiers must start with a letter so measurements such as `5V` are not misclassified as part IDs. Similarity below 0.2 without an explicit identifier match produces no candidate; this threshold is a relevance heuristic, not confidence or proof. Lexical mode remains an explicit backend configuration option; failures do not silently claim semantic success. No cross-encoder reranker was added without evidence justifying its cost.

## Setup and database verification

- `bootstrap.py` reuses the existing downloader for the optional embedding manifest. Revision, byte length and SHA-256 are checked before use; download locks, resume, atomic replacement and offline reuse are shared with the chat-model downloader. Startup reports embedding availability separately and does not disable ordinary chat for an embedding setup failure.
- Fresh `.venv-air05-check` dependency setup passed with unchanged requirements. Running bootstrap again reused the verified environment/artifacts. Native benchmark inference passed from that new environment. This is fresh-environment verification on the existing Windows host, not a new OS/container certification.
- `scripts/verify_embedding_setup.py` downloaded a clean temporary artifact set in 4.91 s, repeated verification, loaded/encoded offline, rejected a corrupted temporary artifact and checked model-identity invalidation. It never corrupted the application's working model.
- Inspected `dev/development` VPN MySQL first. Added and applied `VF_EMBEDDING_001.sql` to `voltforge_db`; snapshot `Voltforge_BL/target/embedding-schema-before-20261022205353.sql`. Reapplication passed. Vector JSON write/read and owner-to-vector deletion cascade were tested with synthetic records and rolled back. Existing records were not rewritten.
- Read-only audit now verifies **six** SQL scripts and **14** foreign-key relationships with no orphans. Existing 7 users, 5 projects, 2 conversations, 30 messages and 3 memories remain. Test document/vector rows were cleaned up; zero embeddings after cleanup is expected.

## Executed checks

- AI: 76 unit/integration tests and 5 health/security tests passed. Includes artifact corruption/offline/revision checks, encoder authentication, bounds, identity binding and unavailable state.
- BL: 38 existing conversation/gateway/policy tests passed; final hybrid/source run passed all 9 tests (4 hybrid, 5 document). Includes actual-vector fusion parity, negative queries, project consent, owner isolation, cache reuse, model rebuild, version replacement and deletion.
- Real shared-DB + BL-to-AI + native model acceptance passed: 14-page Arduino PDF, 16 chunks indexed, stored vectors verified and reused, 35 native answer chunks, 3 persisted references, 2 messages, a page-6 source reopened, retry replay without duplicate messages. First answer token: 24,685 ms. The authenticated service test uses a synthetic account; it does not claim a fresh Keycloak browser login.
- UI source was unchanged in this stage. AIR-04 browser/build evidence remains historical; no new browser verification is claimed. Container/GPU execution remains unverified.
- Backend Maven packaging passed after the final code changes; dependency consistency and diff checks passed. Restart BL and AI to load the updated implementation; the shared database migration is already applied.

## Main changed files

AI: `embedding_manifest.json`, `embedding_model.py`, `engine/embeddings.py`, `api/embeddings.py`, configuration/startup/bootstrap/Docker packaging, benchmark/setup scripts and tests. BL: `DocumentHybridSearch`, encoder client, embedding entity/repository/store, `DocumentService`, configuration, migration/helper and integration tests. Existing chat provider and generation paths were reused.

Primary references: [upstream model and pooling/token limits](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [GGUF conversion publisher](https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF), [publisher embedding integration](https://llamaedge.com/docs/inference-sdk/embedding-app).
