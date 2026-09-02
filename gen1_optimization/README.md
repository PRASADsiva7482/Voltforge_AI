# Gen1 inference optimization

VFAI-018 measures every optimization against the explicit
`vfai018-reference-fp32-manual-no-kv` profile in a fresh process with outbound
network access denied. Each candidate re-verifies the signed artifact, evaluates
all 9,027 targets in the immutable `vf-tokenizer-split-v1` validation stream,
and repeats four bounded generation probes derived from the frozen VFAI-005
suite. Reports contain prompt/output hashes, never raw prompt or output text.

## Measured CPU decision

The selected profile is `vfai018-fp32-sdpa-kv`: FP32 weights, PyTorch SDPA,
unexpanded GQA KV caches, four Torch threads, no memory mapping, and one request
per generation batch. It preserved validation loss `5.9367194`, top-1 accuracy
`0.019386286`, and exact greedy output tokens. Median measured generation for
four eight-token probes improved from 774.1903 ms for the unoptimized reference
to 365.5 ms.

The safe fallback is `vfai017-safe-fp32-manual-kv`. Runtime loading falls back
to it only when the selected local optimization is unavailable; corrupt or
untrusted artifacts still fail closed.

Dynamic batching at four requests measured 225.193367 tokens/second, but it is
not enabled for serving because a cancellation/deadline-aware scheduler belongs
to the later API work. Local int8 and int4 probes reduced resident model tensors
from 56,793,856 bytes to 19,829,248 and 12,732,928 bytes respectively, but were
slower and CPU-only. They are neither released weight formats nor selected
profiles. Memory mapping and alternate thread counts did not clear the material
improvement gate on this host.

The benchmark does not improve the undertrained checkpoint's absolute quality,
approve model release, or approve service activation. CUDA and MPS remain
conditional unmeasured tiers; only portable FP32 weights are packaged.

## Commands

```powershell
.toolchains/gen1/Scripts/python.exe tools/benchmark_gen1_optimizations.py verify
.toolchains/gen1/Scripts/python.exe tools/package_gen1_optimized_artifact.py verify
.toolchains/gen1/Scripts/python.exe tools/smoke_gen1_optimization.py
.toolchains/gen1/Scripts/python.exe -m pytest tests/test_gen1_optimization.py -q
```

Regenerating the benchmark intentionally takes longer than verification because
all ten candidates run in isolated processes:

```powershell
.toolchains/gen1/Scripts/python.exe tools/benchmark_gen1_optimizations.py benchmark
```

## Owned accelerator follow-up

`accelerator-policy.v1.json` freezes the VFAI-FU-007 CUDA and Apple Silicon
measurement boundary without changing the historical CPU evidence. It requires
the same immutable artifact, complete 9,027-target validation split, exact greedy
outputs, three FP32 reference/selected/fallback profiles, conditionally supported
BF16/FP16 profiles, fresh processes, warmups, 30 measured generations per
profile, thermal cycles, synchronized timing, memory evidence, and signed
fallback/rollback.

Generate and verify the content-free readiness receipt:

```powershell
.toolchains/gen1/Scripts/python.exe tools/evaluate_accelerator_inference_readiness.py evaluate --generated-on 2026-08-31
.toolchains/gen1/Scripts/python.exe tools/evaluate_accelerator_inference_readiness.py verify
```

The historical `benchmark.py` remains checksum-bound CPU evidence and is not
silently repurposed as an accelerator harness. The readiness evaluator records
that its checkpoint loading and benchmark tensors are CPU-bound. A separate
device-aware harness and real owner-controlled hardware are required before any
CUDA or MPS result report can be assigned in the accelerator policy.
