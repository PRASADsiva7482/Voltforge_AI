# VFDLM Gen1 reproducible training

This package performs real next-token optimization of the project-owned Gen1
decoder. It consumes only checksum-approved VFAI-009 task shards through the
released VFAI-010 tokenizer. The tokenizer's immutable 138-record training and
16-record evaluation split is reused as the model training/validation split;
all shard, manifest, record-ID, compiled-corpus, and tokenizer hashes are
recomputed before a model is allocated.

Documents are framed with `[BOS]` and `[EOS]`, concatenated in stable record-ID
order, and packed into fixed context blocks. Adjacent blocks overlap by one
token, ensuring every stream transition contributes exactly once to next-token
loss. Training batches use a private deterministic shuffler whose permutation,
position, epoch, generator state, and counters are checkpointed.

The optimizer is PyTorch AdamW with decay excluded from one-dimensional norm
weights. The pipeline supports token-weighted gradient accumulation, global
gradient clipping, linear warmup plus cosine decay, complete validation passes,
and CUDA mixed precision when the selected hardware supports it. CPU `auto`
uses float32; unsupported explicit precision requests fail instead of silently
changing arithmetic.

Every safe-point checkpoint contains:

- strict Gen1 model checkpoint and architecture manifest;
- AdamW tensor state;
- scheduler position and learning rate;
- GradScaler state, including its disabled CPU state;
- packed-data cursor and private shuffle RNG;
- Python, NumPy, Torch CPU, and Torch CUDA RNG states;
- global step, tokens seen, metrics, and accumulated runtime;
- checksums binding every checkpoint file.

The mutable run manifest records the exact training/model configs, approved
dataset and tokenizer lineage, source-file hashes plus Git revision state,
hardware/runtime details, timing, real train/validation metrics, and every
checkpoint checksum. Resume refuses changed code, data, tokenizer, config,
model, or checkpoint content. Checkpoints occur only after a completed optimizer
safe point, so no uncheckpointed partial accumulation is claimed as resumable.

Mixed-precision overflow is also handled at that boundary. A skipped GradScaler
update backs off the scaler, restores the exact packed-data cursor and Python,
NumPy, Torch CPU, and Torch CUDA RNG states, clears gradients, and retries the
same optimizer step. Recovery is bounded to eight retries and each completed
step records its retry count and before/after gradient scale. The CPU-safe
synthetic regression verifies this state machine, but it is not CUDA acceptance
evidence.

VFAI-FU-004 remains open until owner-controlled CUDA hardware is available.
Generate or verify the content-free capability receipt without loading the
governed corpus:

```powershell
.toolchains/gen1/Scripts/python.exe `
  tools/evaluate_cuda_mixed_precision_resume.py `
  evaluate --generated-on 2026-08-31
.toolchains/gen1/Scripts/python.exe `
  tools/evaluate_cuda_mixed_precision_resume.py verify
```

Passing the BF16/FP16 capability smokes only authorizes the controlled local
FP32/BF16/FP16 training and resume scorecard. It does not complete the follow-up
or change the auto-precision release policy.

Use the pinned environment documented in `model/gen1/README.md`:

```powershell
.toolchains/gen1/Scripts/python.exe tools/train_gen1.py `
  --run-directory training_runs/gen1-smoke `
  --run-id gen1-smoke `
  --max-steps 10 `
  --warmup-steps 2 `
  --validation-interval 5 `
  --checkpoint-interval 5
```

Resume from the checksum-verified latest pointer:

```powershell
.toolchains/gen1/Scripts/python.exe tools/train_gen1.py `
  --run-directory training_runs/gen1-smoke `
  --resume
```

`training_runs/` is ignored. A run becomes a release artifact only through the
later VFAI-014 to VFAI-016 evaluation and registry gates.

## Context-capable revision follow-up

[`context-revision-policy.v1.json`](context-revision-policy.v1.json) freezes the
VFAI-FU-008 readiness and experiment boundary. It predeclares 1,024, 2,048,
4,096, and 8,192-token candidates, identical raw-record exposure, three seeds,
context-specific packing, long-position coverage, full quality gates, runtime
measurements, fallback, and signed rollback. It does not authorize training or
activation.

The frozen VFAI-020 compiler evidence uses tokenizer 1.0.0, while current Gen1
training uses the approved 1.1.0 tokenizer. A separately identified compiler
policy and synthetic content-free snapshot must therefore prove exact 1.1.0
accounting before a context candidate can run. The historical compiler policy,
snapshot, current 128-token artifacts, and their weights remain immutable.

Generate or verify the readiness receipt without loading weights, training
data, prompts, or project content:

```powershell
.toolchains/gen1/Scripts/python.exe `
  tools/evaluate_context_revision_readiness.py `
  evaluate --generated-on 2026-08-31
.toolchains/gen1/Scripts/python.exe `
  tools/evaluate_context_revision_readiness.py verify
```

## Tokenizer 1.1 Gen1 revision follow-up

[`tokenizer-v1.1-revision-policy.v1.json`](tokenizer-v1.1-revision-policy.v1.json)
freezes the VFAI-FU-013 start and evaluation boundary. A new run must use a new
immutable plan, run ID, checkpoint namespace, project-owned random
initialization, tokenizer 1.1.0, content-addressed corpus lineage, at least three
independent seeds, and an owner-approved finite compute/storage budget. Existing
tokenizer 1.0.0 runs and packages cannot be relabeled or overwritten.

Training is not currently authorized. VFAI-FU-006 has no approved corpus stage
or owner budget, and VFAI-FU-015 immutable release retention is incomplete. The
readiness evaluator hashes current tokenizer/shard inputs and historical package
manifests without parsing training examples, loading weights, allocating a run,
or changing the registry:

```powershell
.toolchains/gen1/Scripts/python.exe `
  tools/evaluate_gen1_v1_1_revision_readiness.py `
  evaluate --generated-on 2026-09-01
.toolchains/gen1/Scripts/python.exe `
  tools/evaluate_gen1_v1_1_revision_readiness.py verify
```

After the start gates are satisfied, the same receipt advances sequentially
through immutable-plan creation, inactive three-seed training, the complete
quality/runtime/reproducibility scorecard, and independent release review. It
never authorizes automatic packaging or registry activation.
