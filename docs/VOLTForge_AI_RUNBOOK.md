# VoltForge AI operations and development runbook

Version: `vfai035-runbook-v1`  
Owner: VoltForge AI/runtime, with backend, UI, data, and operations owners  
Scope: the project-owned VoltForge AI service and its three-repository release
boundary

This is the operational source of truth for building, checking, serving,
backing up, rolling back, and retiring VoltForge AI. It describes the current
implementation honestly. A command that writes a corpus, tokenizer, model,
registry, or release record is an intentional change and requires review.

## 1. Support boundary and release truth

The [owned Gen2 foundation design](LLM_FOUNDATION_DESIGN.v1.md) defines the
`vfdlm-g2` family trained from random initialization on an owner-controlled
private server. The Gen2 design contract does not approve model training or activation.
Its base, instruction and integrated-assistant stages have separate evidence
gates. Gen1 policies and history remain unchanged; task 018 owns the new frozen
evaluation suite and thresholds. Verify the design before implementing a new
foundation task:

```powershell
rtk proxy .\.toolchains\gen1\Scripts\python.exe tools/verify_llm_foundation.py
```

Task 018 now supplies the [independent evaluation harness](LLM_FOUNDATION_EVALUATION.v1.md),
900 frozen cases and an additive design/policy binding. The original task-017
design verifier retains its historical pending-threshold result; use the binding
verifier below for the current threshold state. The evaluation baseline grants no model release or activation approval.
Acceptance is checksum locked, with separate confidential custody still required
before candidate training. Preserve its material outside all training inputs.

```powershell
rtk proxy .\.toolchains\gen1\Scripts\python.exe tools/evaluate_llm_foundation.py verify
rtk proxy .\.toolchains\gen1\Scripts\python.exe tools/verify_foundation_evaluation_binding.py
```

Task 019 adds the [training-source inventory](LLM_TRAINING_SOURCE_INVENTORY.v1.md).
Four source definitions and four existing task shards pass current permission
checks; twelve legacy datasets remain quarantined. The 227 eligible synthetic
records measure 112,418 tokens with the existing tokenizer as a planning proxy.
The source inventory does not admit a Gen2 corpus or authorize training.
Seven external sources remain proposals pending exact content and permission
review. Preserve held-out records and distinct source, shard and release gates.

```powershell
rtk proxy .\.toolchains\gen1\Scripts\python.exe tools/inventory_llm_sources.py verify --recompute
```

Task 020 adds [reproducible corpus ingestion](LLM_CORPUS_INGESTION.v1.md).
The offline builder stages 227 approved records, excludes 23 held-out records,
and records immutable raw/normalized shards, extraction review and memory evidence.
Ingestion staging does not authorize training or a Gen2 corpus release.
Exact deduplication is implemented; task 021 still owns near-duplicates and family splits.
The build-only dependency is pinned in `requirements-corpus.txt`; runtime serving
does not import the ingestion or PDF parser modules.

```powershell
rtk proxy .\.toolchains\gen1\Scripts\python.exe tools/ingest_llm_corpus.py verify --release corpus/ingestion/v1/4eb23ca6950fb9d7fa9fd87bfbfcbe29aadd92d008a1d9d1e45229231078948d --recompute
```

Task 021 adds [family partitions and leakage controls](LLM_CORPUS_PARTITIONS.v1.md).
All 227 staged records are quarantined by the stricter protected-content/family
checks; the new train/validation/test files are empty. Positive and negative
fixtures separately verify populated partitions and deliberate leakage rejection.
Partition verification does not establish corpus adequacy or authorize training.
Task 022 adds a separate owned-domain candidate release with pre-render
reservations: 67 records across 27 families, 51 independent calculation checks,
and 24 exact Uno/Mega compiler checks. The frozen policy admits 17 records to
candidate partitions (12 train, 0 validation, 5 test) and quarantines 50,
including all 12 firmware examples. Domain corpus readiness remains false.
Owned domain candidates do not authorize corpus release or model training.
Task 023 subsequently completed source admission, independent validation and
pilot/input corpus release gates. See [owned domain data](LLM_OWNED_DOMAIN_DATA.v1.md).

Task 023 supplies an [admitted pilot/input corpus](LLM_PILOT_INPUT_CORPUS.v1.md):
42 train / 11 validation / five test / eight quarantine records, with 919,275
unique train and 34,462 validation proxy tokens. Source, family, quality,
finite-budget and real-input resource gates pass. All older exclusions remain
excluded. Admitted pilot inputs do not approve a training run or production model release.
Production corpus/mixture acceptance remains in tasks 031/032. The older
[candidate artifacts](LLM_PRETRAINING_CORPUS_CANDIDATE.v1.md) remain historical.
Pretraining corpus candidates do not authorize tokenizer fitting or model training.

```powershell
rtk proxy .\.toolchains\gen1\Scripts\python.exe -B tools/build_pretraining_corpus.py verify --release corpus/pretraining-candidates/v1/8f55313b779ffa8c8beebc19cbb3761a354a0e219fdf794865882d4728dcf852 --recompute
```

```powershell
rtk proxy .\.toolchains\gen1\Scripts\python.exe -B tools/build_owned_domain_data.py verify --release corpus/owned-domain/v1/8ac1756fd8c8402d032ccedc94d58d44faf5ec67659754e2b6770fce3c130a25 --recompute
```

```powershell
rtk proxy .\.toolchains\gen1\Scripts\python.exe tools/partition_llm_corpus.py verify --release corpus/partitions/v1/be29e68619449efec0348e5747944b2a1861b78b82a93642df923de2424813c6 --recompute
```

VoltForge AI is a local, project-specific system. Response generation is
owned by the VoltForge code and approved local artifacts. There is no hosted LLM, third-party generation API, pretrained model dependency, or remote-device fallback. Internet access is an optional, bounded evidence lookup only; web
content is untrusted, citation-only context and is never training data.

The following rules are release invariants:

- The production model never changes weights from live chats. The runtime has
  no training or download path.
- A model is served only from the signed registry's active, release-approved,
  activation-eligible artifact. File presence, a package report, or the word
  `approved` in a package manifest is not sufficient for activation.
- The current checked-in registry intentionally has
  `activeArtifactId: null`, state `no-approved-artifact`, and three
  experimental inactive packages. In this state deterministic engineering
  tools and safe fallbacks remain available, while neural chat is unavailable.
  Do not activate an experimental checkpoint to make a health check green.
- Memory and feedback are bounded runtime data. Memory has
  `trainingUseAllowed: false`; feedback must pass project access, review,
  held-out, consent, and leakage gates before it can become a candidate input.
- Structured project changes remain proposal-only and require explicit review
  and user confirmation in the backend/UI transaction boundary.

Use these owners when an incident crosses a boundary:

| Area | Owner | First evidence |
| --- | --- | --- |
| Local runtime, artifact compatibility, generation gates | AI/runtime | AI health, registry status, signed manifest |
| JWT, project access, revision binding, AI token | Backend | Spring logs by request ID, backend tests, actuator health |
| Proposal review, apply, undo, stale editor state | UI/backend | browser console-free status, UI tests, project revision |
| Corpus, memory, feedback, retention, deletion | Data/privacy | source/shard/policy hashes, bounded database metadata |
| Account, firewall, secrets, backups, service manager | Operations | service status, ACLs, secret-manager audit, backup manifest |

## 2. Clean-machine setup

### Prerequisites

For the verified Windows development path install Git, `uv`, CPython 3.12,
JDK 17, Maven or the backend Maven wrapper, and Node.js 20 or newer. A clean
machine must obtain dependencies from the pinned manifests and the approved
PyTorch CPU index; it must not need a hosted generation service.

From the workspace root, obtain the three sibling repositories:

```powershell
Set-Location D:\Project\voltforge
Test-Path .\Voltforge_AI
Test-Path .\Voltforge_BL
Test-Path .\Voltforge_UI
```

The quality runner expects exactly these sibling names. The root is not a Git
repository; Git status and history commands must be run in each child
repository.

### Create the pinned AI environment

Run these commands from `Voltforge_AI`. The `.toolchains` directory is ignored
and must never be committed:

```powershell
Set-Location D:\Project\voltforge\Voltforge_AI
uv venv .toolchains\gen1 --python 3.12
uv pip install --python .toolchains\gen1\Scripts\python.exe `
  --index-url https://download.pytorch.org/whl/cpu `
  -r requirements-gen1.txt
uv pip install --python .toolchains\gen1\Scripts\python.exe `
  -r requirements.txt -r requirements-dev.txt
.toolchains\gen1\Scripts\python.exe --version
.toolchains\gen1\Scripts\python.exe -c "import torch; print(torch.__version__)"
```

The native Gen1 contract requires Python ABI compatibility and PyTorch `2.8.0`.
If the pinned interpreter or framework is unavailable, stop. Do not install a
different framework version and relabel the artifact as compatible.

Copy `.env.example` to a local `.env` only when environment loading is
configured by the operator. Never commit `.env`, a service token, a database
password, or the Ed25519 private signing key. In production set
`VOLTFORGE_AI_ENVIRONMENT=production`, an explicit origin list, a random token
of at least 32 characters, and a runtime directory with the required ACLs.

### First verification

Before starting a service or writing any generated output, run the no-write
runbook and repository checks:

```powershell
.toolchains\gen1\Scripts\python.exe tools/verify_runbook.py
.toolchains\gen1\Scripts\python.exe tools/scan_quality_boundaries.py --check
.toolchains\gen1\Scripts\python.exe -m pytest -q
```

The normal three-repository development gate is run from the workspace root:

```powershell
Set-Location D:\Project\voltforge
.\run_quality_pipeline.ps1 -Lane fast
```

The release gate additionally requires clean child worktrees and the model
lane:

```powershell
.\run_quality_pipeline.ps1 -Lane all -Release
```

## 3. Start and verify the local service

### AI service alone

The service can run without MySQL and without internet retrieval. Development
defaults to loopback and no token for local compatibility. Production must be
token-protected and must not expose port `2002` publicly.

```powershell
Set-Location D:\Project\voltforge\Voltforge_AI
$env:VOLTFORGE_AI_ENVIRONMENT = "development"
$env:VOLTFORGE_AI_HOST = "127.0.0.1"
$env:VOLTFORGE_AI_PORT = "2002"
$env:VOLTFORGE_AI_INTERNET_RETRIEVAL_ENABLED = "false"
.toolchains\gen1\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 2002
```

In a second terminal, check content-free health and contract responses:

```powershell
Invoke-RestMethod http://127.0.0.1:2002/voltForge-ai/api/v1/model/health
Invoke-RestMethod http://127.0.0.1:2002/voltForge-ai/api/v1/model/system/health
Invoke-RestMethod http://127.0.0.1:2002/voltForge-ai/api/v1/model/contract
```

For production, send `X-Voltforge-AI-Token` or a bearer token from the secret
manager. Never put that value in a browser bundle, command history, issue, or
runbook. A normal current response reports `NO_APPROVED_MODEL_ARTIFACT`; that
is expected until a complete signed release is activated.

### Backend and UI integration

The supported request path is browser → Spring backend → private Python AI
service. The backend supplies the AI service token and server-derived
user/project/session identity; the browser does not call the Python port with
credentials.

```powershell
Set-Location D:\Project\voltforge\Voltforge_BL
.\mvnw.cmd spring-boot:run

Set-Location D:\Project\voltforge\Voltforge_UI
npm install
npm run dev
```

The backend listens on `http://127.0.0.1:2001/voltForge-app` by default and
the UI on `http://localhost:3000`. Confirm `app.ai.model.url` points to the
private AI service and that `VOLTFORGE_AI_API_TOKEN` is identical on both
services when production authentication is enabled. Backend tests must pass
before diagnosing a UI symptom as an AI defect.

The Dockerfile is a deterministic service packaging path, not a way to
download or train a model. An approved neural artifact requires the pinned
Gen1 runtime and a separately mounted, immutable artifact store; the current
registry intentionally has no serving artifact.

Build the deterministic container from `Voltforge_AI` when Docker is part of
the deployment environment. The image must still be reached only through the
private backend network:

```powershell
docker build --tag voltforge-ai:local .
docker run --rm --publish 127.0.0.1:2002:2002 --env VOLTFORGE_AI_ENVIRONMENT=development voltforge-ai:local
```

## 4. Data and corpus lifecycle

The task-023 admitted input release binds exact source-use decisions, retained
notices, checked math/electronics validation references and all source/split
gates. Use its verifier and train-only handoff; source permission alone is
insufficient. Source-input admission does not authorize corpus fitting or model training.

```powershell
rtk proxy .toolchains\gen1\Scripts\python.exe -B tools/release_pilot_corpus.py verify --release corpus/pilot-input/v1/04830f1057093c4a71930e9f49940c6507c59496425926679e5cb789897c0f94 --recompute
```

```powershell
.toolchains\gen1\Scripts\python.exe -B tools/expand_pretraining_sources.py verify-admission
```

The task-023 [leakage precision review](LLM_CORPUS_LEAKAGE_PRECISION.v1.md)
distinguishes direct identity/literal/code evidence from lexical matches needing
review. It preserves every current corpus exclusion and exports counts only.
Leakage precision review does not authorize source use or corpus readmission.

```powershell
.toolchains\gen1\Scripts\python.exe -B tools/review_corpus_leakage.py controls
```

File presence never grants training or retrieval approval. Every accepted
source, shard, and task record is checksum-bound to its producer, schema,
license/privacy evidence, and allowed use.

### Curated electronics corpus

Verify first; write only after a reviewed source or builder revision:

```powershell
.toolchains\gen1\Scripts\python.exe tools/build_electronics_corpus.py
.toolchains\gen1\Scripts\python.exe tools/build_electronics_corpus.py --write
```

The corpus is project-owned normalized facts, not copied web prose. Exact
variant ambiguity and missing evidence are preserved. Do not edit generated
packs by hand.

### Synthetic training release

The approved VFAI-009 pipeline is deterministic and compiler-backed:

```powershell
.toolchains\gen1\Scripts\python.exe tools/build_verified_synthetic_data.py --check
.toolchains\gen1\Scripts\python.exe tools/build_verified_synthetic_data.py --recompile
.toolchains\gen1\Scripts\python.exe tools/build_verified_synthetic_data.py --write
```

`--check` is the normal no-write check. `--recompile` requires the locally
pinned Arduino toolchain and validates retained receipts. `--write` recompiles
and replaces the checked-in release outputs only after a deliberate data
review. Never add project snapshots, chats, live web results, or generated
model responses to these shards.

After a source revision, inspect impact before removal:

```powershell
.toolchains\gen1\Scripts\python.exe tools/audit_data_governance.py --check
.toolchains\gen1\Scripts\python.exe tools/source_removal_impact.py SOURCE_ID
```

## 5. Tokenizer lifecycle

Gen2 tokenizer fixtures do not authorize corpus fitting, model training or serving.

Task 024 now supplies a [corpus-trained owned tokenizer](LLM_TRAINED_GEN2_TOKENIZER.v1.md)
with 16,384 entries, immutable 0.1.0 lineage and actual decoder/context
version checks. The older [fixture preparation](LLM_GEN2_TOKENIZER_PREPARATION.v1.md)
remains historical. The trained tokenizer does not approve model training, serving or external-retention acceptance.

```powershell
rtk proxy .toolchains\gen1\Scripts\python.exe -B tools/train_gen2_tokenizer.py verify --release model/tokenizers/gen2/releases/f2bf8a23c7d251252bf120ea3262875152e3dede6a4549e3ea147bc867a09d65 --recompute
```

```powershell
.toolchains\gen1\Scripts\python.exe -B tools/build_gen2_tokenizer.py fixtures
```

This offline command fits only public implementation fixtures, verifies the
resulting byte contracts and reports actual sizes. It grants no corpus or model
release approval. The separate corpus-trained release uses the admitted-input verifier.

The released tokenizer is the project-owned byte-level BPE contract. Its
vocabulary, merges, split, trainer code, and evaluation report are immutable
dependencies of a compatible model package.

```powershell
.toolchains\gen1\Scripts\python.exe tools/build_tokenizer.py --check
.toolchains\gen1\Scripts\python.exe tools/build_tokenizer.py --write
.toolchains\gen1\Scripts\python.exe -m pytest -q tests/test_tokenizer_release.py
```

Use `--check` during normal operation. `--write` is a release change and must
be accompanied by a new tokenizer version, a new model compatibility record,
and a fresh release evaluation. Never replace tokenizer files in an existing
artifact directory.

## 6. Model training and resume

### Supported Gen1 training path

`tools/train_gen1.py` is the governed PyTorch path. It loads only approved
training data and the released tokenizer, records source/config/runtime hashes,
and checkpoints optimizer, scheduler, data cursor, and random states at safe
points. Training is a development/research operation; it does not approve,
package, activate, or replace the production model.

Create a new run with a unique run ID:

```powershell
.toolchains\gen1\Scripts\python.exe tools/train_gen1.py `
  --run-directory training_runs\gen1-review-001 `
  --run-id gen1-review-001 `
  --max-steps 10 `
  --warmup-steps 2 `
  --validation-interval 5 `
  --checkpoint-interval 5 `
  --device cpu
```

Resume only from the checksum-verified latest checkpoint and the unchanged
run manifest:

```powershell
.toolchains\gen1\Scripts\python.exe tools/train_gen1.py `
  --run-directory training_runs\gen1-review-001 `
  --resume
```

If resume reports changed code, data, tokenizer, configuration, model, or
checkpoint bytes, preserve the run for audit and create a new run. Do not
delete a checkpoint to force resume.

The older `model/train.py`, `model/train_chunks.py`, and retired experiment
paths are not release training paths. They may not write a catalog artifact or
consume unapproved data. A feedback retraining manifest is currently only a
verified candidate input: there is no automatic executor that merges it into a
training shard or changes live weights.

## 7. Evaluation and inference tuning

Run evaluation before packaging any candidate:

```powershell
.toolchains\gen1\Scripts\python.exe tools/check_evaluation_leakage.py
.toolchains\gen1\Scripts\python.exe tools/run_release_evaluation.py --require-pass
.toolchains\gen1\Scripts\python.exe tools/evaluate_generation_quality_gates.py evaluate
```

The scorecard must pass with empty critical failure lists and content-free
hashes. A failed quality gate is a release stop, not a reason to rewrite the
held-out suite or lower a threshold.

The measured optimization profile is part of the signed artifact contract.
Verify it and run the offline smoke before making a new package:

```powershell
.toolchains\gen1\Scripts\python.exe tools/benchmark_gen1_optimizations.py verify
.toolchains\gen1\Scripts\python.exe tools/smoke_gen1_optimization.py
.toolchains\gen1\Scripts\python.exe -m pytest -q tests/test_gen1_optimization.py
```

Only a measured, portable FP32 profile is released. A changed thread count,
attention implementation, quantization format, context length, or generation
limit requires new benchmark evidence and a new immutable artifact. Tuning an
environment variable cannot silently alter a signed artifact's behavior.

## 8. Package, approve, activate

Package operations create immutable artifacts and signed manifests; they do
not make an artifact active. Verify the package that was actually built:

```powershell
.toolchains\gen1\Scripts\python.exe tools/package_gen1_artifact.py verify
.toolchains\gen1\Scripts\python.exe tools/package_gen1_runtime_artifact.py verify
.toolchains\gen1\Scripts\python.exe tools/package_gen1_optimized_artifact.py verify
.toolchains\gen1\Scripts\python.exe tools/manage_model_registry.py verify
```

The normal release sequence is:

1. Build and verify the artifact from approved data, tokenizer, checkpoint,
   evaluation, and compatibility evidence.
2. Run `run_quality_pipeline.ps1 -Lane all -Release` from clean worktrees.
3. Create a signed candidate record bound to the exact scorecard:

   ```powershell
   .toolchains\gen1\Scripts\python.exe tools/manage_model_release.py candidate ARTIFACT_ID --scorecard SCORECARD.json
   ```

4. Collect the required bounded canary receipt and promote only if the
   artifact, scorecard, compatibility, and canary policy pass:

   ```powershell
   .toolchains\gen1\Scripts\python.exe tools/manage_model_release.py stable model\registry\releases\ARTIFACT_ID.candidate.json --canary CANARY.json
   ```

5. Activate with the expected registry revision. This is an operator action,
   never a chat-side effect:

   ```powershell
   .toolchains\gen1\Scripts\python.exe tools/manage_model_release.py activate model\registry\releases\ARTIFACT_ID.stable.json --expected-revision REVISION
   ```

6. Recheck health, artifact identity, compatibility, and canary metrics after
   the pointer switch. Keep the previous stable artifact and signed history.

Stable promotion requires at least 100 observations, zero contract failures,
zero safety failures, error rate at most 2%, and p95 latency at most 5 seconds.
The active pointer is switched atomically and revision-guarded. A failed
signature, checksum, compatibility, scorecard, canary, or expected revision
leaves the prior pointer unchanged.

## 9. Search and internet evidence

Local retrieval is the default authority. Verify its immutable index without
writing:

```powershell
.toolchains\gen1\Scripts\python.exe tools/build_local_retrieval_index.py verify
.toolchains\gen1\Scripts\python.exe tools/evaluate_local_retrieval.py evaluate
.toolchains\gen1\Scripts\python.exe tools/evaluate_local_retrieval.py verify
```

If the curated source or builder changes, rebuild only through the reviewed
write path:

```powershell
.toolchains\gen1\Scripts\python.exe tools/build_local_retrieval_index.py write
```

Internet evidence is disabled by default. If an operator enables it, use only
the configured `duckduckgo-instant-answer-v1` provider and retain the checked
no-proxy, no-redirect, HTTPS-only, bounded-response policy:

```powershell
$env:VOLTFORGE_AI_INTERNET_RETRIEVAL_ENABLED = "true"
.toolchains\gen1\Scripts\python.exe tools/evaluate_internet_retrieval.py evaluate
```

Result URLs are citations only. Never paste a web result into a training
shard, corpus pack, feedback example, model artifact, or release report.
Provider failure must degrade to a content-free state while local tools and
local generation continue.

## 10. Memory and feedback operations

### Bounded memory

Memory is disabled until explicitly enabled for a user/project scope. It
stores bounded, redacted entries with hashed identity keys, project revisions,
expiration, stable eviction, and delete/correct controls. It is not a model
training mechanism.

The current policy allows at most 1,200 characters per entry, 50 project
entries, 12 session entries, 24 KiB per project, and a maximum TTL of 2,160
hours. Recent-turn summaries default to 168 hours. Inspect or clear memory
through the authenticated backend path; do not query the SQLite file to expose
user content.

```powershell
.toolchains\gen1\Scripts\python.exe tools/evaluate_bounded_memory.py evaluate
.toolchains\gen1\Scripts\python.exe tools/evaluate_bounded_memory.py verify
```

For a deletion request, authenticate the exact user/project/session scope and
use the backend memory delete operation. Record only the request ID, scope
hash, deletion count, and policy version in operational evidence. Never copy
the deleted content into a ticket or backup.

### Governed feedback and retraining candidates

Feedback requires gateway-authenticated user/project headers and a matching
project ID. It starts as `pending-review`; useful feedback is not a regression
case. Incorrect/unsafe feedback requires expected behavior, and the evidence
must be explicitly approved. Raw transcripts, project snapshots, secrets, and
instruction-like material are rejected.

```powershell
.toolchains\gen1\Scripts\python.exe tools/manage_feedback.py health
.toolchains\gen1\Scripts\python.exe tools/manage_feedback.py review FEEDBACK_ID --decision approve-heldout --reviewer-id REVIEWER_ID
.toolchains\gen1\Scripts\python.exe tools/manage_feedback.py approve-training FEEDBACK_ID --example TRAINING_EXAMPLE.json --reviewer-id REVIEWER_ID
.toolchains\gen1\Scripts\python.exe tools/manage_feedback.py schedule --feedback-id FEEDBACK_ID --base-artifact-id ARTIFACT_ID --base-registry-revision REVISION
.toolchains\gen1\Scripts\python.exe tools/manage_feedback.py verify-run RUNTIME_RUN_JSON
```

Scheduling writes candidate JSONL and a reproducibility manifest only. It does
not run training, load weights, change the registry, or mutate a live chat.
Every future training candidate must repeat held-out leakage, model-quality,
artifact-integrity, candidate-release, stable-canary, and signed-activation
gates. There is no supported command that bypasses this sequence.

## 11. Incident diagnosis

Start with the smallest content-free receipt and keep the request ID, error
code, artifact ID, registry revision, and policy/report hashes. Do not capture
prompts, project snapshots, generated text, search queries, secrets, or stack
traces in incident tickets.

### AI unavailable or degraded

```powershell
.toolchains\gen1\Scripts\python.exe tools/manage_model_registry.py status
.toolchains\gen1\Scripts\python.exe tools/manage_model_registry.py verify
Invoke-RestMethod http://127.0.0.1:2002/voltForge-ai/api/v1/model/health
Invoke-RestMethod http://127.0.0.1:2002/voltForge-ai/api/v1/model/system/health
```

Interpretation:

- `NO_APPROVED_MODEL_ARTIFACT`: expected for the current checked-in registry;
  deterministic fallbacks remain the safe behavior.
- Signature/checksum/compatibility failure: stop rollout, do not edit the
  artifact or registry by hand, and compare the signed hashes with the release
  receipt.
- Loading/runtime failure: verify the pinned Python/PyTorch ABI, device policy,
  exact artifact file set, and read-only artifact permissions.
- Slow or cancelled generation: inspect bounded observability metrics and
  configured request/context/stream ceilings before changing model parameters.

### Retrieval or internet evidence degraded

```powershell
.toolchains\gen1\Scripts\python.exe tools/build_local_retrieval_index.py verify
.toolchains\gen1\Scripts\python.exe tools/evaluate_local_retrieval.py verify
.toolchains\gen1\Scripts\python.exe tools/evaluate_internet_retrieval.py evaluate
```

An unavailable or stale local index must return no authoritative result. An
internet provider error must not trigger retries, arbitrary URL fetching, or
invented component facts. Keep internet retrieval disabled while isolating a
network incident.

### Memory, feedback, or tool path degraded

```powershell
.toolchains\gen1\Scripts\python.exe tools/evaluate_bounded_memory.py verify
.toolchains\gen1\Scripts\python.exe tools/manage_feedback.py health
.toolchains\gen1\Scripts\python.exe tools/evaluate_engineering_tools.py evaluate
```

If a database is unavailable, fail closed for the affected stateful operation;
do not switch to raw file logging or an unscoped fallback. A deterministic
engineering tool failure must remain visible as a tool failure, not be replaced
with a guessed model claim.

### Backend or UI path degraded

```powershell
Set-Location D:\Project\voltforge\Voltforge_BL
.\mvnw.cmd -q test

Set-Location D:\Project\voltforge\Voltforge_UI
npm test
npm run build
```

Check backend AI URL, private token injection, JWT/project authorization,
request-size and stream admission limits, and the current project revision.
The UI must show local AI unavailable/degraded status and preserve editing when the AI service is down. Never work around a stale proposal by applying its payload manually.

### Containment and recovery

1. Stop new AI traffic at the backend or service manager; keep the UI and
   project database available when safe.
2. Preserve content-free health, registry, release, and request-ID evidence.
3. If a registry file is damaged, copy it to a restricted incident directory
   before recovery, then verify the newest signed history snapshot:

   ```powershell
   .toolchains\gen1\Scripts\python.exe tools/manage_model_registry.py recover
   .toolchains\gen1\Scripts\python.exe tools/manage_model_registry.py verify
   ```

4. If a stable artifact violates a rollback trigger, stop rollout and perform
   an explicit signed, expected-revision rollback using the prior stable
   release record. Do not delete the current artifact during rollback.
5. Rotate a suspected service token or signing key through the secret/trust
   owner. Never regenerate a missing trusted private key silently.
6. Re-run the complete release and cross-repository gates before reopening
   traffic. Record the incident code and hashes, not private content.

## 12. Backup, retention, rollback, and retirement

### Backup set and handling

Backups are encrypted, access-controlled, outside the source worktree, and
tested by restoring into a disposable directory. At minimum retain:

- the signed `model/registry/active_model.json` and `model/registry/history/`;
- the public trust store and signed release records;
- each active and previous-stable immutable artifact directory;
- the exact scorecard, canary, compatibility, tokenizer, retrieval-index, and
  data-lineage receipts that authorize the artifact;
- operational policy versions and content-free incident receipts.

Do not back up `.env` into the repository. A private signing key belongs in an
approved secret manager or encrypted offline escrow with a tested recovery
procedure; its public trust record alone cannot sign a new release.

Example Windows backup outline; choose a restricted destination approved by
Operations and Data/Privacy before running it:

```powershell
$backupRoot = "D:\Voltforge-backups\YYYYMMDD-HHmm"
New-Item -ItemType Directory -Force -Path $backupRoot | Out-Null
Copy-Item -LiteralPath "D:\Project\voltforge\Voltforge_AI\model\registry\active_model.json" -Destination $backupRoot
Copy-Item -LiteralPath "D:\Project\voltforge\Voltforge_AI\model\registry\history" -Destination $backupRoot -Recurse
Copy-Item -LiteralPath "D:\Project\voltforge\Voltforge_AI\model\registry\trust" -Destination $backupRoot -Recurse
Get-ChildItem -LiteralPath $backupRoot -Recurse -File | Get-FileHash -Algorithm SHA256
```

Memory and feedback databases contain governed project data even though their
contents are bounded. Back them up only when the data owner approves the
retention, encryption, ACL, and deletion policy. Never use a general log or
model-artifact backup as a second copy of those databases.

### Retention and deletion

Release records and registry history are immutable evidence. Keep at least one
previous stable history entry and its artifact until the next approved stable
release has passed its canary window and the retention owner signs off. Do not
delete artifacts merely because they are not active.

Memory follows its policy TTL and explicit deletion controls. Feedback follows
the project/data retention decision and must preserve only bounded hashes and
review metadata required for audit. Platform logs need independent redaction,
ACL, and retention configuration; this Python service does not claim to control
all host logs.

### Rollback

Before rollback, verify the stable release record and capture the current
registry revision. Use the release lifecycle command so compatibility, signed
artifact identity, and the expected revision are checked:

```powershell
.toolchains\gen1\Scripts\python.exe tools/manage_model_release.py rollback model\registry\releases\ARTIFACT_ID.stable.json --expected-revision REVISION
```

If the live registry file itself is damaged, use `manage_model_registry.py
recover` only after preserving the damaged file. Rollback changes a signed
registry revision; it does not erase artifacts or history.

### Artifact retirement

Retirement is a signed catalog operation, not filesystem deletion. Verify the
artifact is neither active nor the current rollback target, preserve its
backup, and use a revision guard:

```powershell
.toolchains\gen1\Scripts\python.exe tools/manage_model_registry.py status
.toolchains\gen1\Scripts\python.exe tools/manage_model_registry.py retire ARTIFACT_ID `
  --expected-revision REVISION `
  --operator-id OPERATOR_ID `
  --reason "Superseded after retention review."
```

The command marks the catalog entry `retired`, disables activation, records a
bounded reason and operator identity, writes signed history, and keeps package
bytes intact. Physical deletion requires a later approved retention/data
decision, a verified backup, and an incident-safe change window. Never use
recursive deletion against the workspace or artifact root as a retirement
shortcut.

## 13. Current capability matrix and planned work

| Capability | Current state | Safe operator action |
| --- | --- | --- |
| Deterministic circuit/electronics tools | Supported locally | Verify/evaluate; preserve typed failures |
| Local neural generation | Runtime implemented but current artifact is inactive and release-blocked | Do not activate experimental packages |
| Local retrieval | Supported from the signed curated index | Verify; rebuild only after corpus review |
| Internet search | Optional untrusted evidence, disabled by default | Enable only for evidence intent and policy-approved egress |
| Memory | Explicit bounded project/session storage | Enable per scope; inspect/correct/delete through authenticated APIs |
| Feedback | Governed review and candidate scheduling | Keep held-out-first and consented; no live training |
| Model training | Reproducible Gen1 create/resume path | Train isolated runs; evaluate and package separately |
| Owned Gen2 foundation | Verified immutable ingestion and family/contamination checks; current 227 records yield no independent training partition | Continue with LLM-TASK-022 independent domain families and pre-render reservations; broad corpus/training remain open |
| Feedback-driven model training executor | Not implemented | Keep scheduled inputs isolated; do not merge manually |
| Signed release and rollback | Implemented and operator-triggered | Use release records, canary, expected revisions |
| Automated backup/incident service | Deployment responsibility | Use encrypted external backups and this runbook |

The following are explicitly planned, not silently promised by this runbook:

- a model-specific executor that consumes only verified feedback candidates,
  produces an immutable artifact, and repeats all release gates;
- deployment-specific service-manager, quota, backup restore-test, and key/
  token rotation automation;
- broader data consent workflows, if the product owner ever approves private
  project/chat training as a separate governed feature.

## 14. Change-control checklist

Before merging or deploying an AI change, an operator or reviewer must be able
to answer yes to all applicable questions:

1. Is the change inside the project-owned local AI boundary, with no
   third-party generation model or hidden network dependency?
2. Does it preserve fail-closed artifact, data, project-scope, memory,
   feedback, retrieval, and proposal boundaries?
3. Are new generated files, hashes, manifests, and release records reproducible
   from the pinned toolchain?
4. Were held-out leakage, security, API, backend, UI, and build checks run?
5. If a model or tokenizer changed, are compatibility, quality, packaging,
   candidate, canary, signing, and rollback evidence bound to the exact bytes?
6. Are backup, retention, deletion, incident owner, and rollback implications
   recorded without copying private content?
7. Is the distinction between supported, experimental, release-blocked, and
   planned work visible to operators and users?

The final standard is reproducibility with an honest boundary: a clean machine
can rebuild the supported local service and its verification evidence, while
an unavailable or unsafe model remains unavailable instead of being replaced
by an untracked or remote dependency.
