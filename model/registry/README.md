# VoltForge signed local model registry

`active_model.json` is the only production model pointer. Registry schema 2 is
signed with Ed25519, revisioned, and switched with one atomic file replacement.
Inference must never scan legacy directories, initialize random weights, or use
an artifact merely because it exists in the catalog.

## Current state

The registry preserves `vfdlm-g1-edge-v0.1.0-bootstrap`, catalogs
`vfdlm-g1-edge-v0.1.1-runtime`, and catalogs
`vfdlm-g1-edge-v0.1.2-optimized`, its measured VFAI-018 runtime-profile
revision. All three contain the same 14,198,464-parameter step-256 VFAI-014
FP32 weights and approved tokenizer content. All are deliberately `experimental`,
`activationEligible: false`, and inactive. `activeArtifactId` must remain null
until later data, evaluation, output-gate, and release work explicitly approves
a serving artifact.

The local packages are stored under `artifacts/` and excluded from Git. Their
signed manifests are mirrored under `manifests/`, and complete packaging
receipts are under `reports/`. Build or verify them with the pinned runtime:

```powershell
.toolchains/gen1/Scripts/python.exe tools/package_gen1_artifact.py package
.toolchains/gen1/Scripts/python.exe tools/package_gen1_artifact.py verify
.toolchains/gen1/Scripts/python.exe tools/package_gen1_runtime_artifact.py package
.toolchains/gen1/Scripts/python.exe tools/package_gen1_runtime_artifact.py verify
.toolchains/gen1/Scripts/python.exe tools/package_gen1_optimized_artifact.py package
.toolchains/gen1/Scripts/python.exe tools/package_gen1_optimized_artifact.py verify
```

The private Ed25519 key is generated once under the ignored
`.toolchains/registry-signing/` directory. Only its public key and fingerprint
are stored in `trust/trusted-keys.json`. Back up the private key securely if the
local artifact must remain signable; replacing it requires an explicit trust
rotation, not silent key regeneration.

## Package contract

Each immutable artifact directory contains model weights/configuration, the
approved tokenizer, generation policy, evaluation and scale-decision evidence,
compatibility versions, provenance, and `artifact-manifest.json`. The signed
manifest declares every file's relative path, byte size, role, SHA-256, and the
whole file-inventory checksum. Verification rejects missing, extra, modified,
path-escaping, unsigned, untrusted, and incompatible content.

Compatibility is exact for the Gen1 runtime interface, Python ABI, PyTorch
version, architecture, checkpoint format, tokenizer ID, and tokenizer contract.
VFAI-017 may implement inference for `pytorch-gen1-v1`; the existing NumPy
production loader continues to reject that runtime.

## Activation and rollback

Use `tools/manage_model_registry.py` for read-only status/verification and for
future explicitly approved lifecycle changes:

```powershell
.toolchains/gen1/Scripts/python.exe tools/manage_model_registry.py status
.toolchains/gen1/Scripts/python.exe tools/manage_model_registry.py verify
.toolchains/gen1/Scripts/python.exe tools/manage_model_registry.py activate ARTIFACT_ID --expected-revision REVISION
.toolchains/gen1/Scripts/python.exe tools/manage_model_registry.py rollback --expected-revision REVISION
.toolchains/gen1/Scripts/python.exe tools/manage_model_registry.py recover
```

Activation verifies the registry signature, expected revision, approval state,
activation eligibility, artifact signature, exact file set, checksums,
compatibility, generation policy, and catalog-to-package identity before the
atomic switch. Every successful switch first writes an immutable signed history
snapshot. Rollback is another signed revision; recovery can restore the newest
valid signed history snapshot if the live registry file is damaged.

No checked-in experimental artifact can activate. The manager returns
`MODEL_ARTIFACT_NOT_APPROVED` without changing the registry.

VFAI-033 adds the signed operational release layer described in
[`../../docs/MODEL_RELEASE_LIFECYCLE.md`](../../docs/MODEL_RELEASE_LIFECYCLE.md).
Use `tools/manage_model_release.py` to bind a passing VFAI-032 scorecard and
canary receipt to `candidate` and `stable` release records before activation.
The package-level `approved` value remains a readiness prerequisite and does
not by itself make an artifact stable.

## Observability and privacy

Health exposes the signed registry revision/checksum and exact catalog artifact
identity while continuing to report `ready: false` and
`NO_APPROVED_MODEL_ARTIFACT`. Startup logs contain only registry state, artifact
IDs, revision, and checksum. Prompts, chat text, search queries, generated code,
and model outputs are excluded from these logs.

Artifact IDs follow
`vfdlm-g{generation}-{edge|core|server}-v{semanticVersion}`. Exact parameter
count, context length, quantization, checksum, and release state are reported as
separate facts rather than size marketing.
