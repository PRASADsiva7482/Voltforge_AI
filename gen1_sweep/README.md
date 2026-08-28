# Gen1 architecture sweep

VFAI-013 selects a bootstrap Gen1 decoder configuration using measured quality and
runtime cost. It does not train or approve a release model.

The immutable plan is `plan.v1.json`. Every executable candidate uses the same
approved VFAI-009/VFAI-010 corpus and tokenizer, 8,128 predicted training tokens,
128-token packed blocks, optimizer schedule, seed, CPU precision, Torch thread
limits, and complete frozen validation split. Candidate processes vary only the
declared architecture and regularization dimensions.

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
