# Deployment continuity boundary

`VFAI-FU-010` now has an application-side, content-free verification boundary.
It inventories the signed registry, public trust records, immutable registered
artifacts, release policy, registry history, and signed release records. The
inventory is not a backup and records that no encryption or restore was run.

The application intentionally does not choose a deployment target, copy backup
data, provide encryption, schedule host tasks, access a secret manager, or
rotate credentials. Operations and the data owner must first approve the
target, retention, RPO, RTO, availability, encryption, network, account, and
secret-manager controls in `deployment-continuity-policy.v1.json`.

The owner approval is itself checksum-bound and content-free. It records finite
RPO, RTO, availability, and retention values plus distinct opaque Operations
and data-owner approval identifiers:

```powershell
python tools/manage_continuity.py verify-objectives --receipt <receipt.json>
```

Generate and verify the safe source inventory:

```powershell
python tools/manage_continuity.py inventory --generated-on YYYY-MM-DD
python tools/manage_continuity.py verify-inventory
```

An external backup job can emit a checksum-bound
`vfai-fu-010-encrypted-backup-v1` receipt. Validate it without reading backup
contents or keys:

```powershell
python tools/manage_continuity.py verify-backup --receipt <receipt.json>
```

After an external system restores into a clean disposable directory, verify
every inventoried checksum, the signed registry, all registered artifacts, and
any signed release records, then write a content-free restore receipt:

```powershell
python tools/manage_continuity.py verify-restore `
  --restored-root <clean-directory> `
  --restore-target-id <opaque-target-id> `
  --restore-scheduler-sha256 <sha256> `
  --verified-on YYYY-MM-DD `
  --output <restore-receipt.json>
```

External token and signing-trust rotation jobs must emit separate
`vfai-fu-010-secret-rotation-v1` receipts. Each receipt proves that the old
credential was rejected, the new credential was accepted, rollback was tested,
and trusted release history remained valid. Raw credentials are forbidden.

```powershell
python tools/manage_continuity.py verify-rotation `
  --secret-kind service-token --receipt <receipt.json>
python tools/manage_continuity.py verify-rotation `
  --secret-kind signing-trust --receipt <receipt.json>
```

The readiness evaluator fails closed at the first missing external gate:

```powershell
python tools/evaluate_deployment_continuity_readiness.py evaluate `
  --generated-on YYYY-MM-DD
python tools/evaluate_deployment_continuity_readiness.py verify
```

Do not put `.env`, private keys, runtime databases, raw tokens, secret-manager
identifiers that reveal secret values, or a restore destination inside the live
source worktree into these receipts.
