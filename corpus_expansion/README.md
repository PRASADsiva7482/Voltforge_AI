# Governed corpus expansion

VFAI-FU-006 is a data-governance and evidence program, not a request to inflate
the token counter. `policy.v1.json` freezes the 1M, 10M, and 100M release stages,
task balance, source ownership, immutable lineage, held-out coverage, manual
sampling, deterministic receipts, budget approval, and three-seed learning-curve
gates before high-volume generation begins.

The counting unit is one checksum-bound packed next-token prediction under the
stage tokenizer. Repeated epochs, exact duplicates, semantic near-duplicates,
and mechanical paraphrases never count as unique coverage.

Generate and verify the content-free readiness receipt with the pinned local
runtime:

```powershell
.toolchains/gen1/Scripts/python.exe tools/evaluate_corpus_expansion_readiness.py evaluate --generated-on 2026-08-31
.toolchains/gen1/Scripts/python.exe tools/evaluate_corpus_expansion_readiness.py verify
```

The evaluator reads approved local manifests and reports counts, checksums, task
balance, and missing gates. It does not export corpus text, generate records,
train a model, access the network, or authorize a release.

Current private-project and feedback training remains disabled. Naming a data
source is not consent: the data-governance private-consent workflow, per-record
authorization, redaction, review, retention, and deletion contracts must all be
implemented and approved before any such record can enter a candidate shard.

No stage may begin until an owner records finite storage and compute budgets and
VFAI-FU-015 makes new shard paths immutable and restore-tested. Each later stage
also requires the prior stage's independent three-seed learning curve to improve
held-out and global quality without a critical regression.
