# VoltForge model release lifecycle

VFAI-033 separates package integrity from operational release approval. The
existing signed artifact manifest uses `experimental` and `approved` for
package-level readiness. The new signed release record uses the operational
states `experimental`, `candidate`, `stable`, and `rejected`; `approved` in an
artifact manifest means only that the immutable package is eligible for a
future stable decision.

## Release sequence

1. Build and verify an immutable artifact, then register it without changing
   the active pointer.
2. Run the VFAI-032 model release scorecard and save only its hashes, decision,
   and failure lists as release metadata.
3. Create a signed candidate record:

   ```powershell
   .toolchains/gen1/Scripts/python.exe tools/manage_model_release.py candidate ARTIFACT_ID --scorecard scorecard.json
   ```

4. Run the required canary and bind its content-free metrics to the scorecard.
   Promote the signed candidate to stable only when the artifact is approved,
   compatible, the scorecard passes, and the canary meets policy:

   ```powershell
   .toolchains/gen1/Scripts/python.exe tools/manage_model_release.py stable model/registry/releases/ARTIFACT_ID.candidate.json --canary canary.json
   ```

5. Activate the stable record with an expected registry revision. The registry
   manager verifies the artifact again and atomically replaces the signed
   active pointer:

   ```powershell
   .toolchains/gen1/Scripts/python.exe tools/manage_model_release.py activate model/registry/releases/ARTIFACT_ID.stable.json --expected-revision REVISION
   ```

6. Keep the previous stable artifact. If a rollback trigger fires, stop new
   rollout traffic, verify the prior stable record, and perform an explicit
   revision-guarded rollback:

   ```powershell
   .toolchains/gen1/Scripts/python.exe tools/manage_model_release.py rollback model/registry/releases/ARTIFACT_ID.stable.json --expected-revision REVISION
   ```

Every pointer switch is a signed registry revision with an immutable history
snapshot. A failed checksum, signature, expected revision, artifact package,
runtime contract, scorecard, or canary leaves the active registry unchanged.

## Compatibility and migration rules

The release record binds the exact runtime interface, Python ABI, framework,
architecture, checkpoint format, API contract/schema, task-record data schema,
tokenizer ID/contract, and curated retrieval index ID/version/digest. API major
changes, data-schema changes, tokenizer contract changes, and retrieval-index
changes require an explicit migration release. There is no implicit in-place
index or tokenizer migration, and the current migration policy requires the
model and retrieval index to switch as one verified bundle.

## Canary and rollback

Stable promotion requires at least 100 observations, zero contract failures,
zero safety failures, error rate at most 2%, and p95 latency at most 5 seconds.
`rollback-reasons` evaluates the same bounded metrics and returns deterministic
reason codes. Rollback is intentionally operator-triggered and revision
guarded; live chats never mutate model weights or release records.

The current checked-in registry remains `no-approved-artifact` with all three
packages experimental. No current artifact is promoted or activated by this
backlog item.
