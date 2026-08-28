# VoltForge data governance

This directory is the fail-closed authority for data entering VoltForge-owned
model training or retained retrieval. File presence is never approval.

`source-registry.v1.json` records each source's origin, immutable revision and
checksum, license evidence, privacy class, allowed/prohibited uses,
preprocessing version, retention policy, and deletion procedure. The default
policy denies unknown sources, unlicensed web/scraped content, live web results,
private user projects, and user chats for training. A private-data consent
workflow is intentionally not implemented, so those classes cannot be approved
by a local flag.

Each generated JSONL shard receives a manifest containing its content checksum,
record count, source revision snapshots, generator/preprocessor versions and
checksums, eligible uses, and the exact v1 task-schema checksum. Every record
inherits that manifest's lineage. The governed writer validates the complete
batch against `task_schema/task-record.schema.json` and semantic invariants
before atomically creating a shard. Training and retrieval validate the sidecar
before reading any record. A stale source, changed generator, modified shard,
task-schema mismatch, missing manifest, or quarantined status stops ingestion
with a `DATA_*` error.

The 12 pre-governance corpus files are retained as audit evidence. Their
quarantine manifests truthfully preserve current hashes and likely producers,
but none is training- or retrieval-eligible because immutable generation-time
lineage was not retained. VFAI-008 instead creates the first approved source:
49 project-authored electronics knowledge records in eight schema-validated,
checksum-bound packs. These packs are eligible for exact runtime lookup and as
lineage for synthetic task records; they do not retroactively approve legacy
data and are not themselves task-training shards. VFAI-009 creates the first
approved training release: 154 native task-record examples in four shards,
bound to the deterministic pipeline lock, held-out leakage gate, and 24 pinned
Arduino compiler receipts. The release and its no-write verification procedure
are documented in `synthetic_data/README.md`.

Run the corpus audit from `Voltforge_AI`:

```powershell
python tools/audit_data_governance.py --check
```

Refresh generated quarantine manifests and the checked-in report only after a
deliberate corpus review:

```powershell
python tools/audit_data_governance.py
```

Before removing a source, generate a read-only impact report:

```powershell
python tools/source_removal_impact.py vf-src-project-domain-generator-v1
```

Future approved model registry entries must include a `dataLineage` object with
the exact `sourceIds` and `shardIds` used for training. The impact analyzer
matches both direct source references and shard references. Existing legacy
weights/tokenizers have unknown dataset lineage, so they are conservatively
reported as potentially affected and remain quarantined.

Source approval procedure:

1. Register the exact origin, owner, revision, checksum, license evidence,
   privacy classification, allowed uses, and deletion procedure.
2. Keep `approval.status` non-approved until all evidence is present. Never
   approve scraped or private data by assumption.
3. Generate a new shard through the governed writer. Do not edit an approved
   shard or manifest in place.
4. Validate held-out leakage and the VFAI-007 task schema before training.
5. Bind source and shard IDs into every training run and model artifact.

To revise a source, create a new source ID or revision, update its checksum, and
regenerate dependent shards. Do not overwrite lineage and call it the same
immutable revision.
