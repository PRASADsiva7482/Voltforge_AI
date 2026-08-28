# VoltForge AI frozen release evaluation

VFAI-005 freezes release suite `vfai-release-suite-v1` at version `1.0.0`.
It contains 26 held-out cases—two for each required capability—and 13 governed
metrics. The immutable hashes are recorded in `frozen-manifest.v1.json`.

Covered tasks:

- Domain electronics chat.
- Circuit validation, wiring, and board-pin correctness.
- Firmware review, generation, and compiler repair.
- Simulation interpretation and search grounding.
- Cross-session/project memory isolation.
- Out-of-domain refusal, malformed input, and adversarial safety.

Every metric in `metrics.v1.json` declares an owner, scoring method, minimum
threshold, and critical-failure rule. Unsupported capabilities, execution
errors, missing neural output, and generic fallback responses receive zero
release credit. Aggregate scores cannot override critical safety, privacy,
grounding, or input-validation failures.

## Commands

From `Voltforge_AI`, verify that the suite has not changed:

```powershell
python tools/freeze_evaluation_suite.py
```

Check all retained training and retrieval files for exact or semantic leakage:

```powershell
python tools/check_evaluation_leakage.py
```

Record a baseline without requiring it to pass:

```powershell
python tools/run_release_evaluation.py
```

Use the release-blocking form in CI or artifact approval:

```powershell
python tools/run_release_evaluation.py --require-pass
```

The final command returns exit code 2 whenever any release gate is blocked.
`python model/evaluate.py` is a compatibility alias for the same truthful
runner; the former print-only “all checks passed” evaluator no longer exists.

## Leakage boundary

The manifest reserves normalized SHA-256 hashes and token-trigram semantic
fingerprints for fixture inputs and oracle assertions. The gate currently scans
`dataset.txt` and every retained JSONL corpus under `model/artifacts`. Dataset
generators, chunked training ingestion, tokenizer corpus loading, and the
runtime `dataset.txt` retriever also invoke the same exclusion registry.

An intentional fixture change requires a new suite version, review of metric
thresholds, regeneration of the manifest with
`python tools/freeze_evaluation_suite.py --write`, a zero-collision leakage
audit, and a newly recorded current-system baseline. Do not rewrite a locked
manifest merely to make a failing implementation pass.

## Current system baseline

`reports/current-system-baseline.json` records the local deterministic runtime
with no approved neural artifact and all network access blocked. The baseline
is intentionally blocked: 17 of 26 cases pass, the metric average is 0.673077,
six generic fallback cases receive zero credit, and two compiler-repair cases
are explicitly unsupported. Domain chat, compiler repair, simulation
interpretation, refusal, and adversarial safety remain below their frozen
thresholds. The leakage audit passes with zero collisions across 17,814
records in 12 files.
