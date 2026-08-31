# Registry trust anchors

`trusted-keys.json` contains public Ed25519 keys allowed to verify local
VoltForge model registries and artifact manifests. Private keys must never be
stored in this directory or committed to source control.

Trust-key addition, replacement, or revocation is a release-security operation.
Do not edit a public key merely to make a failed signature pass; diagnose the
artifact, registry, private-key backup, and expected signer first.
