# VoltForge AI runtime baselines

VFAI-004 establishes a measured `cpu-development-reference` tier for the
current local runtime. The benchmark runs with database persistence and
internet retrieval disabled, replaces outbound sockets with a rejecting local
guard, and emits JSON conforming to `runtime-baseline.schema.json`.

Run the standard baseline from `Voltforge_AI`:

```powershell
python tools/benchmark_runtime.py
```

For a quicker diagnostic run that does not replace the checked-in reference:

```powershell
python tools/benchmark_runtime.py --cold-start-runs 1 --warmup-requests 0 --measured-requests 1 --concurrency 1 --skip-accelerator-probe --output benchmarks/reports/diagnostic.json
```

The checked-in report is `reports/cpu-reference.json`. It captures:

- CPU, total RAM, storage, OS, Python, NumPy, PyTorch and framework versions.
- CUDA/MPS and NVIDIA GPU/VRAM capability when locally detectable.
- Fresh-process service/model-health startup and peak working set.
- Local deterministic first-SSE-delta latency and character throughput.
- Request latency, request throughput, failures and memory at concurrency 1, 2 and 4.
- Regression budgets derived from those measurements with explicit headroom.

## Metric boundary

The approved-artifact registry currently reports no neural model. Therefore
`model.neuralGeneration` truthfully contains null first-token latency,
tokens/second and neural peak-memory values. The deterministic engine is
measured in characters/second, never relabelled as tokens/second. Neural
startup and generation measurements become release gates only after an
approved artifact and the VFAI-017 local inference runtime exist.

## Reference-tier policy

The CPU tier is required and may run on any matching or stronger local
development machine. An accelerated tier is documented only when the probe
finds a locally usable Torch CUDA or MPS runtime. A physically present GPU is
not treated as usable merely because its product name suggests performance.

Budgets are generated from the report's observed values, not a model name,
parameter label, or theoretical hardware claim. A later hardware tier must
produce its own report and must not reuse these numeric budgets.

## Recorded CPU reference — 2026-08-28

Benchmark `vfai-runtime-20260828T012253Z` ran on Windows 11 with an Intel
Core i5-8265U (8 logical cores), 15.813 GiB RAM, Python 3.14.4 and NumPy
2.5.1. Torch 2.13.0 was installed but could not initialize (`WinError 1114`),
and no CUDA, MPS or NVIDIA device was detected, so this machine does not define
an accelerated reference tier.

Measured CPU results:

| Measurement | Result |
| --- | ---: |
| Fresh-process launch-to-ready p95 / max | 1,388.962 / 1,403.481 ms |
| Cold-start peak working set | 74.117 MiB |
| Deterministic first-delta p95 | 6.491 ms |
| Deterministic median throughput | 151,015.449 characters/second |
| Concurrency-4 latency p95 | 22.919 ms |
| Concurrency-4 request throughput | 164.902 requests/second |
| Concurrency-4 failures | 0 of 8 |

The generated CPU regression budget is 1,760 ms maximum cold start, 12 ms
maximum first delta, 105,710 minimum characters/second, 105 MiB maximum
working set, and at concurrency 4 a 37 ms maximum p95 with at least 115.431
requests/second and zero failures. These limits apply only to the current
deterministic runtime on the declared reference tier; they are not neural-model
release budgets.
