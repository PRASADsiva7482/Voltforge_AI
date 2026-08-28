# Gen1 bootstrap limitations

The VFAI-014 checkpoint is an experimental training result, not a usable or
active VoltForge assistant model. Its release state is `releaseApproved=false`.

## What was demonstrated

- The selected 14,198,464-parameter `edge-wide-gqa` decoder can be optimized
  reproducibly from random initialization using only approved VoltForge data.
- Step 256 was selected from the measured held-out Pareto frontier by six frozen
  metrics. Training loss was not a selection input.
- Full validation loss improved from 8.108122 to 5.936719 (26.78%). Output-only
  held-out loss improved from 8.121738 to 6.033155 (25.72%).
- All nine validation task strata improved in output-only loss, by 23.49% to
  28.01%, and the restored checkpoint reproduces the recorded metrics exactly.

## What was not demonstrated

- Top-1 token accuracy is only 1.939% on the complete validation stream and
  2.086% on output-only targets. This is far below usable generation quality.
- Training contains only 138 records and 71,410 unique next-token transitions.
  The run stopped after 129,866 exposures (1.8186 corpus passes); further
  repetition is not additional knowledge.
- Validation contains only 16 records across nine synthetic task strata. It
  cannot establish broad hardware, firmware, safety, Unicode, conversational,
  or real-project generalization.
- The 128-token raw decoder is not connected to a generation runtime, project
  context compiler, deterministic output gate, retrieval index, internet
  evidence pipeline, memory system, backend activation path, or UI.
- Simulation interpretation, internet/search grounding, memory isolation,
  malformed-input rejection, adversarial safety, and runtime refusal behavior
  are unsupported neural capabilities at this stage.
- All 13 VFAI-005 global metrics assign this raw checkpoint zero neural release
  credit. The separately measured deterministic system remains release-blocked.
- Training was measured only on the CPU reference host in float32. Accelerator,
  mixed-precision, calibrated energy, quantization, and deployment portability
  remain unproven.

## Required next decisions

VFAI-015 must classify the result as data-, optimization-, architecture-, or
capacity-limited and decide whether to revise data/training or stop. VFAI-016
may package a checkpoint only after that decision. No runtime may activate this
checkpoint before the later inference, generation-quality, context, retrieval,
and release gates are complete.

The checksum-bound measurements are in
`evaluation/reports/gen1-bootstrap-heldout-v1.json`; the selected checkpoint
descriptor is `model/gen1/configs/gen1-bootstrap-best-v1.json`.
