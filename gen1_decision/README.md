# Gen1 scale decision

VFAI-015 decides whether to release edge, scale to core/server, or keep no
neural release. The decision is derived from checksum-bound VFAI-013 and
VFAI-014 evidence using `policy.v1.json`; parameter count is never treated as
quality evidence.

```powershell
.toolchains/gen1/Scripts/python.exe tools/make_gen1_scale_decision.py --show-policy
.toolchains/gen1/Scripts/python.exe tools/make_gen1_scale_decision.py --make
.toolchains/gen1/Scripts/python.exe tools/make_gen1_scale_decision.py --check
```

The policy requires each profile to reach its approved-data target and all 13
global metrics with zero critical failures, an approved local generation
runtime, a release-approved checkpoint, and a safe activation path. Core or
server additionally requires at least three independent seeds, a 95% confidence
analysis, material critical-domain improvement with no critical regression, a
demonstrated smaller-profile plateau, and separately measured target hardware.

The current result selects `no-neural-release`. The edge checkpoint is retained
as an experimental inactive baseline for development and VFAI-016 packaging.
The active registry remains empty; deterministic engineering tools stay
authoritative, approved local retrieval may continue, and internet access is
evidence retrieval only—not a model or factual authority by itself.
