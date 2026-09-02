# Immutable synthetic releases

VFAI-FU-015 retains every governed synthetic release below
`synthetic_data/releases/<pipeline-version>/<content-key>`. Each write-once
directory contains an exact mirrored `content/` tree and a self-checksummed
`release-manifest.json`. The content key binds every retained file path, byte
count, and SHA-256 value. Existing directories are verified and never repaired
or overwritten.

`CURRENT.json` is a mutable pointer only. The synthetic pipeline preserves the
previous current payload before changing its lock, receipts, shards, manifests,
or generation report, then publishes and verifies the new release before its
normal release check can pass.

```powershell
.toolchains/gen1/Scripts/python.exe tools/manage_synthetic_releases.py verify-all
.toolchains/gen1/Scripts/python.exe tools/manage_synthetic_releases.py verify-current
```

Tokenizer v1.1.0 remains byte-for-byte unchanged. Training resolves its four
declared shard and manifest hashes to the retained 1.2.0 release and validates
the snapshotted registry, policy, source origins, producer lock, leakage gate,
and task schema. The ordinary tokenizer `--check` retrains against that retained
lineage without rewriting the approved artifact.

Tokenizer v1.0.0 remains inactive and explicitly non-reconstructable from the
trusted bytes currently retained. Its four historical hashes must never be
replaced with current shards. Exact owner-backup recovery may add a new retained
release, but may not edit the historical tokenizer manifest.

Repository retention is not an external backup. FU-015 remains gated until the
owner supplies a finite encrypted-backup/retention policy and an independent
backup/restore receipt covering every retained release.
