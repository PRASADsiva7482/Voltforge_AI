# Gen1 local inference runtime

VFAI-017 implements local inference for the signed `pytorch-gen1-v1` artifact
format. It does not contain a trainer, downloader, hosted model client, remote
device, retrieval client, or network fallback.

## Current release boundary

`vfdlm-g1-edge-v0.1.1-runtime` is a runtime-format revision of the immutable
step-256 bootstrap package. Its model weights and tokenizer content are
unchanged. Canonical checkpoint/tokenizer filenames and an offline generation
policy were added in a new signed artifact because the VFAI-016 parent is
immutable.

The runtime artifact remains `experimental`, `activationEligible: false`, and
`servingEnabled: false`. It may be loaded only with the explicit
`allow_experimental=True` offline API or the smoke tool. The process supervisor
loads only a signed active approved artifact, so normal service startup leaves
the current registry unavailable and does not import PyTorch.

## Runtime behavior

`LocalGen1Runtime` owns at most one tokenizer/model instance and validates the
registry trust key, artifact signature, exact file set, checksums,
compatibility, canonical model/tokenizer manifests, tensor contract, vocabulary,
context, and generation policy before use.

Generation provides:

- CPU, CUDA, or MPS local-device selection (`auto` prefers an available local
  accelerator and otherwise uses CPU).
- KV-cached autoregressive decoding under `torch.inference_mode()`.
- Deterministic greedy decoding and seed-reproducible temperature/top-k/top-p
  sampling.
- Signed input, output, and total-context token ceilings.
- Incremental UTF-8-safe text streaming.
- Cooperative cancellation and monotonic deadlines while queued and between
  decoding steps.
- Explicit unload and reproducible same-artifact restart.

The lifecycle states are `unavailable`, `loading`, `ready`, `degraded`, and
`failed`. Approved active artifacts may become `ready`; an explicitly loaded
experimental artifact is always `degraded`, remains `ready: false`, and cannot
be obtained from the service's production generation accessor.

## Commands

Use the pinned native runtime:

```powershell
.toolchains/gen1/Scripts/python.exe tools/package_gen1_runtime_artifact.py verify
.toolchains/gen1/Scripts/python.exe tools/smoke_gen1_runtime.py --device cpu
.toolchains/gen1/Scripts/python.exe -m pytest tests/test_gen1_runtime.py -q
```

Set `VOLTFORGE_AI_MODEL_DEVICE` to `auto`, `cpu`, `cuda`, or `mps` for future
approved service activation. There is no remote-device value.

The smoke receipt stores hashes, token counts, timings, finish reasons, and
boolean checks. It stores neither the raw prompt nor generated text. Runtime and
startup logs contain artifact IDs, states, devices, revisions, and safe error
codes only.

## VFAI-018 optimization profile

The immutable `vfdlm-g1-edge-v0.1.2-optimized` revision adds the signed
VFAI-018 optimization policy and complete benchmark evidence without changing
weights or tokenizer content. The measured selection is FP32 SDPA with the GQA
KV cache and four CPU threads. The signed fallback is FP32 manual attention with
the KV cache.

Profile selection is artifact-controlled. An explicit profile override is
accepted only by the offline experimental API and is rejected for approved
service loading. If a selected optimization is locally unavailable, the runtime
rematerializes the signed fallback; artifact trust, checksum, compatibility, or
model-shape failures never use that fallback.

See `gen1_optimization/README.md` for the full before/after decision, rejected
quantization probes, portability boundary, and reproduction commands.

## VFAI-019 generation boundary

All future runtime text is untrusted until it passes the strict VFAI-019 JSON,
domain, evidence, citation, safety, confidence, action, repetition, and
completion gates. A rejected candidate cannot produce typed actions and routes
to the deterministic electronics engine with visible SSE reason metadata. The
current experimental artifacts remain disconnected from chat. See
[`../GENERATION_QUALITY.md`](../GENERATION_QUALITY.md) for the policy and
reproduction commands.

## VFAI-020 project context boundary

Future generation receives project data only through the deterministic,
versioned project context compiler. The compiler canonicalizes the complete
request, frames untrusted project sources through `task-record-v1`, and counts
the resulting prompt with the exact project-owned tokenizer before any runtime
access. Task, safety evidence, active selection, relevant nets, and active
source boundaries are mandatory; lower-priority sections are omitted only at
complete typed boundaries.

The signed VFAI-017/VFAI-018 artifacts declare a 128-token context. This is
below the compiler policy minimum of 768 and below the measured 459-token
legacy typed prompt before reserving output. They therefore fail closed with
`CONTEXT_WINDOW_TOO_SMALL` and remain inactive. Existing weights or manifests
must not be relabeled. `VFAI-FU-008` tracks training and evaluation of a new
owned artifact with a measured context window. See
[`../../context_compiler/README.md`](../../context_compiler/README.md).
