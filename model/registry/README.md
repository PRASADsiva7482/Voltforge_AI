# VoltForge local model registry

Production inference resolves exactly one release-approved artifact through
`active_model.json`. It never scans `model/artifacts` and never initializes
random weights when the active artifact is absent or invalid.

The current registry intentionally has no active artifact. Existing files under
`model/artifacts` are retained as legacy audit evidence and classified in
`model/artifact_inventory.json`; they are not production candidates.

Future approved artifacts must live beneath `model/`, declare every required
file and SHA-256 checksum, use a supported runtime, match tokenizer/model tensor
dimensions exactly, and report the exact tensor parameter count.

Each model must declare `tokenizerManifest` in its required files. Activation
verifies the approved VFDLM byte-BPE algorithm, fixed special-token IDs,
manifest integrity hash, vocabulary/merge/config hashes, vocabulary size, and
exact agreement between model and tokenizer source/shard lineage. A copied,
stale, development-only, or differently trained tokenizer therefore cannot be
paired silently with model weights.

Every approved entry must also declare `dataLineage.sourceIds` and
`dataLineage.shardIds`. Missing lineage fails activation; these IDs let the
VFAI-006 source-removal analyzer identify artifacts that must be deactivated,
deleted, and retrained.

Artifact IDs follow `vfdlm-g{generation}-{edge|core|server}-v{semanticVersion}`.
The identity is deployment-oriented rather than parameter-marketing-oriented;
exact parameter count, context length, quantization, checksum, and release
status are reported separately by health and the registry manifest.
