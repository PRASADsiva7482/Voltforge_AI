# Gen1 architecture sweep

VFAI-013 selects a bootstrap Gen1 decoder configuration using measured quality and
runtime cost. It does not train or approve a release model.

The historical immutable plan is `plan.v1.json`. Its candidates used the same
approved VFAI-009/VFAI-010 corpus and tokenizer, 8,128 predicted training tokens,
128-token packed blocks, optimizer schedule, seed, CPU precision, Torch thread
limits, and complete frozen validation split. Candidate processes varied only the
declared architecture and regularization dimensions. The plan remains evidence;
it is not executable against the newer tokenizer v1.1.0 corpus contract.

Run each candidate in a fresh pinned-runtime process:

```powershell
.toolchains/gen1/Scripts/python.exe tools/run_gen1_sweep.py --list
.toolchains/gen1/Scripts/python.exe tools/run_gen1_sweep.py --candidate edge-deep-gqa
```

After all listed candidates complete, assemble and verify the evidence:

```powershell
.toolchains/gen1/Scripts/python.exe tools/run_gen1_sweep.py --assemble
.toolchains/gen1/Scripts/python.exe tools/run_gen1_sweep.py --check
```

The scorecard is written to
`benchmarks/reports/gen1-architecture-sweep-v1.json`; the selected bootstrap
configuration is written to `model/gen1/configs/gen1-selected-v1.json`.

Selection is restricted to the measured Pareto frontier. A core profile remains
eligible only when it materially improves proxy validation loss over the best edge
profile. Parameter count receives no positive score. The report also records why
energy and an alternate vocabulary cannot be measured honestly on this sweep.

## Joint tokenizer/model readiness

`joint-tokenizer-model-policy.v1.json` freezes the VFAI-FU-005 comparison
contract before expensive work begins. It predeclares project-owned 2,048,
3,072, and 4,096-ID byte-BPE candidates, fixed special IDs, identical
record/raw-byte exposure, explicit token/context/embedding/compute accounting,
three-seed model comparisons, and Pareto-only selection. The policy does not
approve the two untrained alternative vocabularies or authorize a release.

Generate and verify the content-free readiness receipt:

```powershell
.toolchains/gen1/Scripts/python.exe tools/evaluate_tokenizer_model_sweep_readiness.py evaluate --generated-on 2026-08-31
.toolchains/gen1/Scripts/python.exe tools/evaluate_tokenizer_model_sweep_readiness.py verify
```

The evaluator reads checksum-approved local lineage to count unique packed
training predictions. It never exports records, trains candidates, runs a model
sweep, accesses the network, or claims VFAI-FU-005 complete. Joint execution
remains blocked until VFAI-FU-006 meets its 100M unique-token governance and
diversity contract and every predeclared tokenizer candidate is independently
approved.
