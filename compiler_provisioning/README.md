# Compiler provisioning and offline-cache readiness

VFAI-FU-014 separates the already verified installed compiler identity from the
stronger claim that a clean Windows host can reconstruct it. The current five
profiles and 88 receipts remain immutable inputs; this folder does not replace
or relabel them.

`policy.v1.json` freezes the complete provisioning-input, owner-budget,
content-addressed cache, clean-host, offline-compile, and cache-restore gates.
Cache keys include the exact profile, FQBNs, host, CLI command, package-index
hashes, dependency archives, and source/submodule revisions. Missing, changed,
extra, partial, or stale cache content fails closed.

Generate or verify the content-free readiness receipt without downloading,
exporting a cache, provisioning a host, or compiling firmware:

```powershell
.toolchains/gen1/Scripts/python.exe `
  tools/evaluate_compiler_provisioning_readiness.py `
  evaluate --generated-on 2026-09-01
.toolchains/gen1/Scripts/python.exe `
  tools/evaluate_compiler_provisioning_readiness.py verify
```

The current local downloads directory is observation evidence only. It is not a
portable cache export until a complete provisioning-input manifest and owner
storage/bandwidth/retention receipt exist. Clean-host and offline cache-restore
receipts must be created on the approved target host; this workstation cannot
self-certify that acceptance gate.
