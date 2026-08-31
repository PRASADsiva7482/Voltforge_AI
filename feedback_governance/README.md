# VoltForge governed feedback and retraining

VFAI-034 keeps feedback outside the live generation and model-weight paths.
The `/voltForge-ai/api/v1/model/feedback` route requires gateway-authenticated
user/project headers and a matching project ID. It accepts only bounded
regression evidence; raw prompts, raw model replies, project snapshots,
secrets, and instruction-like material are rejected rather than persisted.

Feedback starts as `pending-review`. A reviewer may reject it or admit it to
the private, runtime-held-out regression set. Only a held-out regression with
explicit training consent may receive a separately authored training example.
The training example is checked against both the frozen release suite and all
admitted feedback regressions before it can be scheduled.

The scheduler writes a candidate JSONL input and a reproducibility manifest to
the runtime directory. The manifest binds feedback and held-out hashes, the
base artifact and registry revision, current runtime/API/data/tokenizer/index
compatibility, the deterministic seed, hyperparameters, and dependency
checksums. It does not run training, load weights, change the active registry,
or mutate live chats. A retrained artifact must repeat VFAI-032 and VFAI-033
quality, packaging, candidate, canary, and signed activation gates.

Operator commands use the pinned Python environment:

```powershell
.toolchains/gen1/Scripts/python.exe tools/manage_feedback.py health
.toolchains/gen1/Scripts/python.exe tools/manage_feedback.py review FEEDBACK_ID --decision approve-heldout --reviewer-id reviewer-local-v1
.toolchains/gen1/Scripts/python.exe tools/manage_feedback.py approve-training FEEDBACK_ID --example training-example.json --reviewer-id reviewer-local-v1
.toolchains/gen1/Scripts/python.exe tools/manage_feedback.py schedule --feedback-id FEEDBACK_ID --base-artifact-id ARTIFACT_ID --base-registry-revision REVISION
.toolchains/gen1/Scripts/python.exe tools/manage_feedback.py verify-run RUNTIME_RUN_JSON
```

The current checked-in model registry has no active production artifact. No
feedback is automatically accepted for training and no online self-training
is available.
