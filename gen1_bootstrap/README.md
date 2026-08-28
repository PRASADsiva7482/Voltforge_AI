# Gen1 bootstrap training

VFAI-014 trains the VFAI-013 `edge-wide-gqa` configuration from random
initialization using only the approved VFAI-009 shards and VFAI-010 tokenizer.
It produces an experimental best checkpoint; it cannot approve or activate a
release model.

The immutable `plan.v1.json` fixes the optimizer, seed, data packing, CPU
precision, 256-step ceiling, 32-step evaluation/checkpoint interval, six
held-out selection objectives, stopping rules, and all VFAI-005 global gates.
Training loss is recorded but is not a checkpoint-selection objective.

Run with the pinned local runtime:

```powershell
.toolchains/gen1/Scripts/python.exe tools/train_gen1_bootstrap.py --show-plan
.toolchains/gen1/Scripts/python.exe tools/train_gen1_bootstrap.py --run
.toolchains/gen1/Scripts/python.exe tools/train_gen1_bootstrap.py --check-artifacts
```

Local checkpoint bytes are retained under
`training_runs/vfai014-gen1-bootstrap-v1/`, which is ignored until VFAI-016
packages an approved artifact. Checksum-bound scorecards and the selected
checkpoint descriptor are written under `evaluation/reports/` and
`model/gen1/configs/`.

## Selection and release boundary

Every 32 steps, the complete frozen 16-record validation split is measured for
full next-token loss/accuracy and output-only loss/accuracy, macro task loss,
and worst task loss. The best checkpoint must be on the six-objective Pareto
frontier and then have the lowest unweighted mean objective rank; ties choose
the earlier step. Training loss cannot affect the decision.

The 26-case VFAI-005 suite is also rerun and every one of its 13 correctness,
grounding, privacy, refusal, input-validation, and safety metrics appears in the
held-out report. Raw decoder weights receive zero neural release credit because
the approved generation runtime and output gates belong to later backlog items.
