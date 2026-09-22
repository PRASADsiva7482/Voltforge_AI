# Authoritative engineering tools v1

VFAI-021 defines the deterministic engineering boundary used by VoltForge AI.
These tools run locally, do not call a model or network service, and return one
strict `EngineeringAuthorityReport` for a single project revision. The checked
JSON Schema is generated from the executable Pydantic contract.

## Authority contract

Every finding has a stable ID, severity, blocking state, affected project IDs,
hashed or curated evidence, and a reviewable fix. Critical findings are always
blocking and cannot permit a model override. Unknown hardware facts fail closed
when the policy marks them as safety relevant. Proposed edits remain
`proposal-only`, require confirmation, cite deterministic evidence, and bind to
the exact source project revision.

The seven versioned tools are:

| Category | Local tool | Current boundary |
| --- | --- | --- |
| Circuit/netlist | `circuit-netlist-validator` | Connectivity conflicts and conservative electrical hazards |
| Board/pin | `board-pin-capability-checker` | Exact curated board and pin capabilities |
| Firmware | `firmware-static-analyzer` | Bounded source-pattern analysis; not a compiler |
| Compiler | `compiler-diagnostic-interpreter` | Curated diagnostic classification without invented repairs |
| Simulation | `simulation-state-interpreter` | Interpretation of a client-supplied snapshot; not a simulator rerun |
| Units | `units-calculation-checker` | Decimal unit parsing and consistency calculations |
| Components | `supported-component-checker` | Exact curated component/variant support |

`policy.v1.json` pins the tool versions, limits, tolerances, privacy rules, and
critical non-override behavior. `engineering-report.schema.json` pins the
wire contract.

## Model and API boundary

`engineering-authority-index` is the compact mandatory context event. It
contains the highest-priority blocking findings and evidence hashes even when
individual tool events are omitted by a token budget. A future neural answer
must present those blockers as safety warnings and cite the index. Omission or
contradiction is rejected by the generation quality gate. While blockers are
active, deterministic chat returns the authoritative report before any legacy
free-form explanation is considered.

`POST /engineering/check` returns the complete typed report. Existing circuit,
wiring, code-review, and chat routes consume the same report, while health and
stream-start metadata expose content-free tool readiness. The backend and UI
DTOs expose authority state and typed findings; no finding is silently reduced
to an ungrounded string.

## Deliberate limitations

Static firmware analysis is review guidance and cannot claim compilation.
Compiler diagnostics are interpreted only when supplied by a trusted local
build. Simulation input is reported by the client and is not independently
recomputed. Board, pin, and component conclusions are authoritative only for
exact records in the curated corpus; unknown variants remain explicit rather
than being guessed. Existing follow-ups VFAI-FU-001 and VFAI-FU-002 track wider
hardware and compiler-target coverage.

## Reproduce the evidence

Use the pinned Python 3.12 environment:

```powershell
.toolchains/gen1/Scripts/python.exe tools/check_engineering_tools.py verify
.toolchains/gen1/Scripts/python.exe tools/evaluate_engineering_tools.py evaluate
.toolchains/gen1/Scripts/python.exe tools/evaluate_engineering_tools.py verify
.toolchains/gen1/Scripts/python.exe -m pytest tests/test_engineering_tools.py -q
```

The committed evaluation receipt contains synthetic fixture IDs, check results,
and hashes. It stores no raw firmware, compiler output, simulation state, model
prompt, or response text.
