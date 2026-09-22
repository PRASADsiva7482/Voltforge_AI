# VoltForge owned LLM foundation design

Contract: `voltforge-owned-foundation-v1` / **1.0.0**. Frozen design: **2026-09-06**. Owner task: **LLM-TASK-017**.

[Machine contract](../foundation/contract.v1.json) | [Release policy](../foundation/release-policy.v1.json) | [Compatibility mapping](../foundation/compatibility-map.v1.json) | [Build backlog](../../VOLTFORGE_LOCAL_LLM_BUILD_BACKLOG.md)

## 1. Decision and current implementation

VoltForge will train its own model from random initialization. It may reuse reviewed open-source libraries and correct project-owned model mathematics. It imports no pretrained weights or third-party tokenizer. Training source rights and permitted uses are separate from weight ownership; publicly available text is not automatically approved training data. Raw chats, secrets and unreviewed external-model outputs remain excluded.

The target is private serving on an owner-controlled server. Workstation GPU availability does not limit the architecture. Server training, evaluation and deployment need finite measured compute/storage budgets before allocation; this document does not start a paid job or select hardware.

LLM-TASK-016 removed fabricated runtime identity, activation shortcuts and rule-as-neural fallback. The registry remains inactive. This task adds a design, structural release policy and consistency verifier. It implements no Gen2 decoder, trainer, tokenizer, model service or trained artifact, and grants no model acceptance.

Normative decisions are in the JSON contract and release policy. This document explains them. Follow-up mappings preserve prior evidence and statuses. An incompatible design change requires a new contract version and explicit migration; historical receipts retain their original contract hashes.

## 2. Family, stages and identity

The new family namespace is **vfdlm-g2**, using the existing `vfdlm-g{generation}-{profile}-v{semanticVersion}` identity envelope. Its architecture ID is **vfdlm-gen2-decoder-v1**. The existing branding slug stays `vfdlm`; generation 2, architecture and lineage distinguish the new foundation. No Gen1 package is renamed, resized, re-signed or declared a Gen2 parent.

| Release | Reserved example identity | Meaning |
| --- | --- | --- |
| Base model | `vfdlm-g2-server-v0.1.0-base.1` | A pretrained owned checkpoint with held-out language/code/domain and context evidence. It is not accepted for product chat. |
| Instruction model | `vfdlm-g2-server-v0.1.0-instruction.1` | An independently evaluated instruction-trained child of an accepted owned Gen2 base. It still requires assistant integration acceptance. |
| Integrated assistant | `vf-assistant-g2-v0.1.0` | An immutable bundle referencing the exact instruction artifact, runtime, context, tools, retrieval, memory, API, gateway, UI and acceptance receipts. The bundle ID is not a model artifact ID. |

These examples reserve names only; they do not identify existing files. The explicit manifest field `releaseKind` establishes stage, rather than inferring readiness from a name. The existing deployment profile `server` describes intended placement, not parameter count or capability. Report `artifact.artifactId` as the loaded model identity; retain the existing API `model` routing label for compatibility.

The owned tokenizer gets namespace **vfdlm-g2-byte-bpe**, initial version **0.1.0**. Vocabulary size, special-token IDs and merge rules are frozen in task 024 from approved data. Existing tokenizer 1.0.0/1.1.0 releases retain their own identities and evidence.

## 3. Decoder and training boundary

Choose a causal decoder-only Transformer with pre-norm RMSNorm, rotary positions, grouped-query causal attention, SwiGLU feed-forward layers and tied input/output embeddings. Reuse Gen1 mathematics only after independent numerical checks. A new Gen2 config and allocation policy replace Gen1-specific limits; changing defaults in old files is not the migration.

The model consumes token IDs, masks, positions and optional KV state, and returns logits and KV state. It does not retrieve evidence, choose canned responses, call application APIs or compile firmware. Query-head count must divide model width; KV heads must divide query heads; rotary dimensions must be valid. Exact parameter accounting is derived from tensors and checked against the signed architecture config.

Pretraining predicts next tokens with explicit padding/document boundary handling. Instruction training masks user/system/context tokens and learns assistant responses and valid tool-call targets. Packing and loss-mask implementations are tasks 026/027/037 and need independent tests, including document boundaries and no loss on padding. Tool results are trusted only according to their recorded authority and evidence, not because they appear in a conversation.

The initial context design is **4096 total tokens**, with input plus reserved output fitting within that total. Task 046 sets the actual prompt reservations using the selected tokenizer. **8192 tokens** is a separately trained and evaluated extension; changing a config value or interpolating historical weights cannot establish context ability.

Pilot 100–300M and candidate 1–3B parameter ranges remain planning hypotheses from the backlog. They are neither release minima nor promises. Tasks 024–026 select tokenizer and model configurations; tasks 031/032 use learning curves, data exposure and server measurements to select scale. Task 028 pins the training/runtime environment and validates precision/security. This contract neither relaxes the Gen1 runtime pin nor bypasses its checkpoint security block.

## 4. Initialization, continuation and recovery lineage

The root initialization receipt records method `random`, seed/RNG algorithm, source/config/environment digests, initial tensor digest and `imported-checkpoint=null`. Architecture-defined constants such as normalization scales are declared initialization, not imported learned tensors. Record per-tensor initialization rules with the eventual config.

Base pretraining starts from this root. A resumed run must recover the same lineage, optimizer/scheduler, RNG per rank, data cursor, precision scaler, world size and step/token counts. Changed data, tokenizer, config or distributed topology requires an explicit validated migration or a new run, not an undocumented resume.

Instruction training starts from the exact accepted owned Gen2 base checkpoint. Domain/context continuation may use an owned Gen2 parent only when its chain resolves to the recorded random root. An assistant bundle references an accepted instruction checkpoint. No imported or Gen1 weight tensors become a Gen2 parent. Code reuse and checkpoint reuse are different operations.

Every stage binds the contract, release policy, run ID, code snapshot, environment lock, architecture, tokenizer, immutable data release, split manifest, initialization receipt, checkpoint, evaluation policy/suite and report by digest. Unique approved tokens and repeated training exposures are separate counters. Validation/acceptance families cannot enter tokenizer fitting, pretraining, SFT or feedback training. Missing or inconsistent lineage blocks a candidate.

## 5. Module interfaces

These are implementation contracts for later tasks, not claims that the proposed modules exist. Existing paths listed as reuse candidates remain subject to their original policies.

| Module | Input → output | Implementation owner |
| --- | --- | --- |
| data | SourceReceipt: origin, rights, allowed uses, revision, digest, source family. → CorpusRelease: immutable shards, tokenizer-bound unique counts, exposure counts, split/dedup lineage. | LLM-TASK-019, LLM-TASK-020, LLM-TASK-021, LLM-TASK-022, LLM-TASK-023 |
| tokenizer | Approved training split and tokenizer config; no held-out fitting. → TokenizerRelease: version, special IDs, merges/vocab, file hashes and byte-roundtrip receipt. | LLM-TASK-024 |
| model | Gen2Config plus token IDs, attention mask, positions and optional KV state. → Logits and KV state; actual tensor-derived parameter count. | LLM-TASK-025, LLM-TASK-026 |
| trainer | Immutable run plan, masks, initialization or validated owned parent, finite compute/storage budget. → Loss/token metrics, initialization lineage, checkpoints and complete resume state. | LLM-TASK-027, LLM-TASK-028, LLM-TASK-029, LLM-TASK-030, LLM-TASK-031, LLM-TASK-032, LLM-TASK-033, LLM-TASK-034, LLM-TASK-035, LLM-TASK-037, LLM-TASK-038 |
| inference | Verified release bundle, prompt token IDs, generation bounds, cancellation and deadline. → Actual token IDs/deltas, usage counts, finish reason, verified loaded identity. | LLM-TASK-041, LLM-TASK-042, LLM-TASK-043, LLM-TASK-044, LLM-TASK-045 |
| context | Authenticated user/project/session, authoritative project revision, code, history and evidence. → Bounded PromptEnvelope: exact tokenizer/config binding, source IDs, input/output reservations and trusted/untrusted sections. | LLM-TASK-046, LLM-TASK-047, LLM-TASK-048 |
| tools | Typed operation, explicit operands, exact variant and source revision. → ToolResult: status, units, assumptions, evidence/compile/simulation receipts and authority. | LLM-TASK-022, LLM-TASK-049 |
| application | Existing bounded authenticated chat request and current project revision. → Compatible response/SSE with inference source, artifact identity and reviewable proposals. | LLM-TASK-044, LLM-TASK-045, LLM-TASK-050, LLM-TASK-051, LLM-TASK-052, LLM-TASK-053 |
| evaluation-release | Frozen independent suite, exact checkpoint/bundle and all lineage bindings. → Separate numerical, raw-neural, tool-assisted, integration, browser and server receipts. | LLM-TASK-018, LLM-TASK-034, LLM-TASK-039, LLM-TASK-040, LLM-TASK-041, LLM-TASK-052, LLM-TASK-053, LLM-TASK-054, LLM-TASK-055 |

Model and inference modules cannot invoke deterministic response routing. Context and tools are explicit inputs surrounding neural generation. The API orchestrates them and attributes their source; a missing model may use a declared deterministic-only mode, but never earn neural credit. Runtime training/downloads are excluded.

## 6. Product scope and API compatibility

Version 1 targets English electronics explanations, embedded C/C++ firmware, project circuit/simulation assistance, exact hardware lookup and numerical tool use. Greetings, clarification, corrections and helpful unsupported-topic responses are part of the instruction scope. Additional languages need approved data and separate evaluation before support claims.

An unknown component family or board variant remains unknown until exact evidence exists. Ask for missing electrical operands and variant details; do not substitute guessed defaults or claim hardware measurements, compilation or simulation that did not occur. Unsupported requests get a brief scope explanation and a useful next step.

Preserve Python **/voltForge-ai/api/v1/model** and Spring **/api/v1/ai**, including existing chat, streaming, memory, hardware-coverage and cancellation routes. Public request/SSE contracts remain schema **1**, contract **1.0.0**, including the existing model routing label **voltforge-local-engine-v1**. The new family identity fits the current model identity envelope; task 041/042 still must implement a verified Gen2 loader and release adapter.

Internal PromptEnvelope, ToolResult and release metadata do not silently become new public fields. Public incompatible changes need an explicit versioned migration and coordinated API/gateway/UI tests. Existing request size, event and deadline limits remain effective until separately versioned.

The server-authenticated user/project/session and authoritative source revision bind context, memory, tools and results. Preserve `projectRevision` on every event and `sourceProjectRevision` on proposals. A stale project cannot accept an old proposal; review, explicit apply and undo remain required. Model text never directly mutates a saved project.

The model runtime emits actual token IDs and incremental deltas under a bounded generation contract. API deltas remain provisional until final validation. Emit exactly one terminal event, `complete` or `error`. Cancellation, deadline or mid-stream failure must not append a deterministic response and report neural success. Retain public modes `neural-quality-gated`, `deterministic-fallback` and `unavailable`.

Service liveness, model readiness, retrieval availability and tool availability are independent. A loaded, verified artifact establishes model identity/readiness; a reachable HTTP address does not. `VOLTFORGE_AI_HOST` remains the Python bind host. Task 043 defines a separate private model endpoint, resolved-address checks and authentication. Browser clients continue through Spring; they receive no model-server credentials.

## 7. Release acceptance policy

Structural gates in **voltforge-owned-release-policy-v1** are frozen here. Task **LLM-TASK-018** must create and freeze the independent suite, thresholds, denominators, confidence intervals, seeds and grading policy **before training and candidate evaluation**. The expected path is `evaluation/foundation/acceptance-policy.v1.json`; it does not exist yet. Missing/unfrozen evidence means **blocked**, not a pass.

| Stage | Required acceptance beyond common immutable lineage |
| --- | --- |
| base-model | Owned random root; real training loss/token exposure; recovery reproducibility; held-out language/code/domain quality; context use; observed weights/logits/token invocation. |
| instruction-model | Accepted owned base parent; governed instruction/tool data; correct loss masks; unseen/paraphrased/value-change/multi-turn cases; structured-output/safety checks; raw-neural scoring without rule substitution. |
| integrated-assistant | Exact accepted instruction/runtime/context/tools bundle; negative readiness cases; true streaming/cancel/deadline; exact hardware/compiler evidence; authenticated revision-bound gateway; browser review/apply/undo; measured server capacity; independent backup/restore/rollback and final release review. |

Independent lanes cover numerical/training correctness, raw neural quality, neural with tools, HTTP integration, browser behavior, and server performance/rollback. The task-016 test receipts are cleanup evidence only. Template output, echoed prompts, word chunk rate, a binary header, helper tests or reduced loss alone cannot satisfy model or product acceptance.

Each stage progresses experimental → candidate → accepted, or is rejected. Records and history are immutable; accepted/rejected records are not rewritten. Activation is a separate signed, expected-registry-revision operation requiring an accepted assistant bundle, compatible runtime and the reviewed rollout/canary gates. Keep the prior complete bundle for rollback. A design verifier, checkpoint package or base acceptance does not authorize product activation.

## 8. Existing follow-up mapping

Statuses below are observed history, not changed by this task. A task mapping carries relevant obligations forward; it does not mark the old follow-up complete or authorize work outside the selected backlog item.

| Existing follow-up | Observed status | New task links | Decision |
| --- | --- | --- | --- |
| VFAI-FU-001 | done | LLM-TASK-022, LLM-TASK-046, LLM-TASK-049, LLM-TASK-052 | Reuse exact board facts and evidence; expand only through independently validated corpus releases. |
| VFAI-FU-002 | done | LLM-TASK-022, LLM-TASK-023, LLM-TASK-036, LLM-TASK-049, LLM-TASK-052 | Reuse immutable exact-FQBN compiler receipts and tokenizer lineage; do not treat them as base-pretraining completion. |
| VFAI-FU-003 | promoted | LLM-TASK-028, LLM-TASK-029, LLM-TASK-042, LLM-TASK-054 | Keep checkpoint serving blocked and the verified Gen1 pin until clean native imports, numerical parity and server runtime checks establish a new pinned environment. |
| VFAI-FU-004 | accepted_for_later | LLM-TASK-027, LLM-TASK-028, LLM-TASK-029, LLM-TASK-030 | Validate mixed precision and scaler/optimizer/RNG resume on the selected private training server. |
| VFAI-FU-005 | accepted_for_later | LLM-TASK-024, LLM-TASK-025, LLM-TASK-026, LLM-TASK-031, LLM-TASK-032 | Replace Gen1 vocabulary/scale sweep limits only for Gen2; freeze corpus exposure and compare controlled owned tokenizers/configurations. |
| VFAI-FU-006 | accepted_for_later | LLM-TASK-019, LLM-TASK-020, LLM-TASK-021, LLM-TASK-022, LLM-TASK-023, LLM-TASK-031, LLM-TASK-033 | Replace edge-only source/token targets for Gen2 with diverse language/code/domain corpus and measured pilots; preserve rights, unique/exposure distinction, dedup and sealed splits. |
| VFAI-FU-007 | accepted_for_later | LLM-TASK-042, LLM-TASK-045, LLM-TASK-054 | Measure the selected private server's real token throughput, memory and concurrency. Apple Silicon and workstation parity are optional additional profiles, not prerequisites. |
| VFAI-FU-008 | accepted_for_later | LLM-TASK-024, LLM-TASK-026, LLM-TASK-027, LLM-TASK-030, LLM-TASK-033, LLM-TASK-035, LLM-TASK-046 | Train context from owned random-root lineage; use a new tokenizer-bound compiler policy and prove context occupancy and input/output budgets. Never stretch or relabel old 128-token weights. |
| VFAI-FU-009 | accepted_for_later | LLM-TASK-036, LLM-TASK-037, LLM-TASK-038, LLM-TASK-048, LLM-TASK-055 | Retain consent, isolation and offline candidate-only feedback rules. A general feedback executor remains a deferred follow-up; this mapping does not authorize live-chat learning. |
| VFAI-FU-010 | accepted_for_later | LLM-TASK-041, LLM-TASK-054, LLM-TASK-055 | Retain owner/operations backup, secret rotation and independent restore evidence before deployment acceptance. |
| VFAI-FU-011 | accepted_for_later | LLM-TASK-022, LLM-TASK-046, LLM-TASK-049, LLM-TASK-052 | Retain exact manufacturer/revision evidence and unknown generic variants; expand reviewed component coverage. |
| VFAI-FU-012 | accepted_for_later | LLM-TASK-046, LLM-TASK-049, LLM-TASK-052, LLM-TASK-053 | Retain geometry/source revision mapping and real browser fixtures; circuit artwork cannot establish electrical or mechanical correctness. |
| VFAI-FU-013 | accepted_for_later | LLM-TASK-023, LLM-TASK-024, LLM-TASK-027, LLM-TASK-030, LLM-TASK-033, LLM-TASK-034, LLM-TASK-041 | Replace fixed Gen1 tokenizer 1.1.0 and legacy stage namespace only for Gen2. Preserve fresh run identity, immutable split/shard lineage and independent multi-seed evaluation. |
| VFAI-FU-014 | accepted_for_later | LLM-TASK-022, LLM-TASK-029, LLM-TASK-036, LLM-TASK-049, LLM-TASK-054 | Retain checksum-pinned toolchains, exact FQBN, clean-host provisioning and compiler-cache provenance. |
| VFAI-FU-015 | accepted_for_later | LLM-TASK-019, LLM-TASK-020, LLM-TASK-023, LLM-TASK-024, LLM-TASK-030, LLM-TASK-041, LLM-TASK-054, LLM-TASK-055 | Retain immutable content-addressed data/tokenizer history and owner backup/restore requirements; repository-side preservation does not complete external retention. |

## 9. Policy precedence and preserved history

All replacements below apply **only to vfdlm-g2**. Old Gen1 policies, artifacts and receipts remain unchanged and govern their own family. The new design selects different experiment constraints; future implementing tasks must supply new versioned data, tokenizer, config, training and runtime policies before allocation. The compatibility map records the reviewed historical hashes.

| Historical policy | Gen2 replacement |
| --- | --- |
| gen1_decision/policy.v1.json | Gen2 has no inherited 150M ceiling or workstation gate; 100-300M pilots and 1-3B candidates remain planning hypotheses until server measurements. Owner: LLM-TASK-025, LLM-TASK-026, LLM-TASK-031, LLM-TASK-032. |
| corpus_expansion/policy.v1.json | Gen2 freezes a new broad language/code/domain mixture and scale policy; source rights, semantic dedup, split integrity and unique-vs-repeated accounting remain mandatory. Owner: LLM-TASK-019, LLM-TASK-020, LLM-TASK-021, LLM-TASK-022, LLM-TASK-023, LLM-TASK-031. |
| gen1_sweep/joint-tokenizer-model-policy.v1.json | Gen2 tokenizer and architecture sweeps use their own immutable pilot plan and new tokenizer namespace. Owner: LLM-TASK-024, LLM-TASK-025, LLM-TASK-026, LLM-TASK-031, LLM-TASK-032. |
| gen1_training/context-revision-policy.v1.json | Gen2 starts with a 4096-token total trained budget; 8192 is a separate experiment. Exact tokenizer counting, evidence-backed context and immutable history remain required. Owner: LLM-TASK-026, LLM-TASK-030, LLM-TASK-035, LLM-TASK-046. |
| gen1_training/tokenizer-v1.1-revision-policy.v1.json | Gen2 uses vfdlm-g2-byte-bpe with fresh random initialization and fresh run/checkpoint namespaces; no inherited optimizer or weight tensors. Owner: LLM-TASK-023, LLM-TASK-024, LLM-TASK-027, LLM-TASK-030, LLM-TASK-033. |
| gen1_optimization/accelerator-policy.v1.json | Gen2 deployment acceptance targets the chosen private server; additional workstation/Apple profiles are optional. Preserve historical CPU receipts and actual-token measurement. Owner: LLM-TASK-029, LLM-TASK-042, LLM-TASK-045, LLM-TASK-054. |
| model/release-policy.v1.json | Gen2 separates base, instruction and assistant acceptance; task 018 must freeze numerical acceptance before candidates. Signed immutable records, compatibility, canary/rollback and revision-guarded activation remain required. Owner: LLM-TASK-018, LLM-TASK-041, LLM-TASK-054, LLM-TASK-055. |

Retain separately governed source rights, exact hardware/compiler evidence, no private-data export, no hosted generation, no live-chat training, checkpoint safety, immutable signed histories, finite allocation budgets and independent backup/restore evidence. In particular, the existing VFAI-FU-015 owner restore/retention requirement remains open. No edit to an inactive checkpoint or old approval policy can stand in for new acceptance.

## 10. Verification and continuation

From Voltforge_AI:

```powershell
rtk proxy .\.toolchains\gen1\Scripts\python.exe -B tools/verify_llm_foundation.py --check-preserved-history --report evaluation/reports/llm-task-017-foundation-contract.json
rtk proxy .\.toolchains\gen1\Scripts\python.exe -B -m pytest -q tests/test_llm_foundation_contract.py tests/test_model_identity.py tests/test_runbook_contract.py
rtk proxy .\.toolchains\gen1\Scripts\python.exe -B tools/verify_runbook.py
rtk proxy .\.toolchains\gen1\Scripts\python.exe -B tools/verify_llm_backlogs.py
```

The verifier checks structural decisions, stage separation, identity compatibility, route declarations, follow-up coverage and required design sections without importing Torch, allocating tensors, contacting services or approving a release. The optional --check-preserved-history check proves that task 017 preserved the reviewed baseline files and statuses. Omit that historical check for later design validation after an independently governed registry or follow-up update; new release evidence must prove those changes separately. Mutation tests challenge weakened ownership, parent/activation rules, API drift, missing modules and omitted mappings. These checks validate the design contract, not downstream implementations or model ability.

**Next: LLM-TASK-018**, the independent evaluation harness and frozen acceptance thresholds. Training, tokenizer/model implementation, runtime migration, server execution and browser acceptance remain assigned to their later dependency-ready tasks.
