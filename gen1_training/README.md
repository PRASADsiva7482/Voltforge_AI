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
