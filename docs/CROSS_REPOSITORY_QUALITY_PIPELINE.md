# Cross-repository quality pipeline

VFAI-032 provides one local command for the three sibling repositories:

```powershell
Set-Location D:\Project\voltforge
.\run_quality_pipeline.ps1 -Lane fast
```

The runner uses the pinned AI interpreter at `Voltforge_AI/.toolchains/gen1/Scripts/python.exe`, a checked-in `quality_pipeline.v1.json` manifest, and no hosted or third-party generation provider.

## Lanes

`fast` is the normal development and CI gate. It runs the selected Python contract/security/evaluation tests, leakage and governance checks, artifact/provider boundary scanning, Spring tests and gateway receipt, UI tests, the production build plus bundle budget, and whitespace checks in all three repositories.

`model` is intentionally separate and expensive:

```powershell
.\run_quality_pipeline.ps1 -Lane model
```

It runs the Gen1 model tests, offline runtime and optimization smokes, and the release scorecard. The current registry has no approved stable neural artifact, so a model release must remain blocked until a candidate passes the existing release gates. This is an honest release boundary, not a reason to silently activate an experimental artifact.

`all` runs both lanes. A release invocation must use `model` or `all` and clean worktrees:

```powershell
.\run_quality_pipeline.ps1 -Lane all -Release
```

## Evidence and reproducibility

The runner writes `evaluation/reports/cross-repository-quality-v1.json`. The report contains pass/fail status and bounded durations, while the deterministic `scorecardSha256` is calculated from the manifest hash, repository input fingerprints, Git heads, tool versions, selected check definitions, exit codes, and statuses. Raw child stdout/stderr, prompts, project content, and generated responses are never stored in the report.

Build products, caches, and generated evaluation receipts are excluded from source fingerprints so a report cannot change the inputs it measures. Release automation should archive the report, scorecard hash, immutable manifest, and the three repository revisions together.

The workspace is intentionally a sibling-repository layout rather than one Git repository. CI should check out `Voltforge_AI`, `Voltforge_BL`, and `Voltforge_UI` beneath the same workspace directory and invoke the root wrapper. The runner returns non-zero on malformed manifests, provider/credential findings, unsafe web policy, evaluation leakage, schema drift, artifact integrity failure, test/build failure, or any release preflight failure.
