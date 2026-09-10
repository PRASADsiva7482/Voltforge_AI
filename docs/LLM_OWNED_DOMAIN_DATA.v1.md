# Owned electronics domain candidates — LLM-TASK-022

This adds an immutable, versioned candidate source for VoltForge's model trained
from random weights. It does not train a model or change chat serving. No
pretrained weights, third-party page bodies, customer projects or hardware
measurements were acquired for this dataset. The source policy accurately records
coding-assistant authorship; calculation review is algorithmic, not human review.

## Result and acceptance boundary

The [release manifest](../corpus/owned-domain/v1/8ac1756fd8c8402d032ccedc94d58d44faf5ec67659754e2b6770fce3c130a25/manifest.json)
binds 67 candidate records in 27 authored families, the exact generator and
verifier bytes, all source evidence, split reservations, local compiler/core
fingerprints, and the unchanged protected registry.

| Evidence | Verified result |
| --- | --- |
| Parameterized calculation problems | 51 across 17 families |
| Independent negative controls | 153 rejected: wrong answer, wrong unit, nonfinite input |
| New firmware reference projects | 6 families, each on Uno R3 and Mega 2560 Rev3 |
| Exact compiler executions | 24: 12 fixed sources accepted, 12 intentionally broken sources rejected |
| Entity uncertainty examples | 4; false board identity, LED/OLED distinction, generic display and relay ratings |
| Records surviving the frozen partition policy | 17 families: train 12, validation 0, test 5 |
| Quarantined records | 50: 15 cross-reservation bridges and 35 duplicates |
| Surviving cross-split/protected collisions | 0 under the lexical/structural policy |
| Released Gen2 training tokens | 0; every candidate and shard has `trainingAllowed=false` |

Candidate examples cover analog, digital, timing, buses, power, ADC, control,
memory and firmware debugging. The surviving pool covers eight of these nine
domains: **all 12 firmware records remain quarantined**, along with three ring
occupancy variants connected by declared ancestry. There is no surviving
validation partition. `domainCorpusReady=false` is an intentional acceptance
finding; this small candidate release is insufficient for pretraining.

The frozen matcher finds a protected exact board-name substring in the Mega
question and lexical similarity in shared firmware question wording. That
connects families assigned to different pre-render partitions. Shared generic
assumption text also connects the entity cases. These exclusions were retained;
no protected prompt was copied, split reassigned, or threshold reduced to admit
them. The [leakage report](../corpus/owned-domain/v1/8ac1756fd8c8402d032ccedc94d58d44faf5ec67659754e2b6770fce3c130a25/leakage.json)
retains guard IDs and reasons, without reproducing protected evaluation text.
Zero collisions describe only the 17 surviving records, not all 67 candidates or
arbitrary semantic paraphrases. Parameter variants are not counted as new
independent families.

## Source and release structure

The new adapter is `synthetic_data/foundation_domain/`. It does not rewrite the
frozen task-019 source inventory, task-020 ingestion pipeline, task-021 partition
code, earlier synthetic shards or compiler receipts.

- `recipes.py` defines supplied parameters, problem text, derivations and answers.
- `calculations.py` independently checks circuit current/power balances, exact
  rational arithmetic, state tables, enumeration and monotone bisection. Each
  receipt retains units, finite parameter bounds and numeric tolerances.
- `firmware.py` contains runnable reference sketches and a single undeclared-call
  mutation per sketch. Source pairs are compiled for exact targets with pinned
  Arduino AVR core 1.8.6 and bundled Wire 1.0. The environment manifest hashes
  every retained core/library file in addition to compiler identity.
- `entities.py` queries the existing checked electronics corpus; no substring
  alias matching or unspecified electrical ratings are introduced.
- `release.py` writes the recipe/source-bound reservation before rendering or
  compilation, validates every receipt, applies the frozen protected matcher,
  preserves reserved splits, and quarantines cross-reservation bridges.

The content-addressed release contains `candidates.jsonl`, four partition shards,
`calculation-review.jsonl`, `compiler-receipts.json`, `environment.json`,
`reservation.json`, `source-manifest.json`, `decisions.jsonl`, `leakage.json` and
`coverage.json`. Partition shards contain the same source records, not additional
unique data. `normalizedText` preserves code and line breaks. Board/component
identities and hash-bound source receipts remain attached to each applicable
record. All publications are write-once; per-case build caches are disposable
working evidence and are not the released source of truth.

## Reference-project limits

The projects are newly authored examples, not collected private user designs.
Their supplied resistor/capacitor values, ideal devices and hypothetical I2C
address/register are assumptions, not manufacturer ratings. The floor-bin ADC
model explicitly defines its transfer convention; it does not assign that
convention to every physical ADC. Generic LED, OLED and relay labels do not
identify exact electrical variants.

The six compiler projects cover bounded command parsing, interrupt-counter
snapshots, an ADC window average, cooperative switch debounce, a reserved-slot
serial queue and an I2C register transaction. The intentionally broken examples
currently exercise one compiler-error class: an undeclared function call.
Compiler acceptance proves build compatibility; it does not prove runtime,
electrical, deadline, noise, or hardware behavior. In particular, Wire blocking
and recovery, serial throughput, interrupt loss and real ADC behavior need later
tests. Only two exact AVR targets are newly compiled by this release.

The pinned local core is authoritative for the APIs used here. Reference links
are citation-only and are not imported training documents: [Arduino AVR core
1.8.6](https://github.com/arduino/ArduinoCore-avr/tree/1.8.6), [Arduino interrupt
reference](https://github.com/arduino/reference-en/blob/master/Language/Functions/External%20Interrupts/attachInterrupt.adoc),
and [TI precision ADC fundamentals](https://www.ti.com/lit/SLYY192).

## Reproduction

From `Voltforge_AI`, with the existing pinned local toolchains:

```powershell
rtk proxy .\.toolchains\gen1\Scripts\python.exe -B tools/build_owned_domain_data.py build
rtk proxy .\.toolchains\gen1\Scripts\python.exe -B tools/build_owned_domain_data.py verify --release corpus/owned-domain/v1/8ac1756fd8c8402d032ccedc94d58d44faf5ec67659754e2b6770fce3c130a25 --recompute
rtk proxy .\.toolchains\gen1\Scripts\python.exe -B tools/build_owned_domain_data.py verify --release corpus/owned-domain/v1/8ac1756fd8c8402d032ccedc94d58d44faf5ec67659754e2b6770fce3c130a25 --recompute --recompile
rtk proxy .\.toolchains\gen1\Scripts\python.exe -B -m pytest -q -p no:cacheprovider tests/test_owned_domain_data.py
```

`--recompute` verifies the exact local environment and reproduces all source,
calculation, entity, partition and coverage bytes using validated retained
compiler receipts. `--recompile` additionally runs every compiler case freshly
and compares its source/target/outcome identity; diagnostic-output hashes can
differ with compiler cache conditions and are not claimed byte-identical.
No build command downloads toolchains. Missing or altered inputs fail closed.
The public `require_use` gate permits candidate review only.

## Task-023 handoff

Admit this new source version explicitly; existing source permissions do not
automatically cover newly authored bytes. Retain the original candidate source,
reservations and exclusions. Before broad corpus release, provide independent
new families, a populated validation set, and firmware data that passes a reviewed
policy. A future versioned matcher/adapter must distinguish protocol or factual
identity boilerplate from answer/code contamination using independently authored
positive and negative controls, without stripping genuine board-specific answers,
changing sealed evaluation bytes, or tailoring source text to evade guards.

The existing task-023 indexed matching/recall/resource gate remains mandatory at
larger scale. Expand firmware defect classes and runtime checks as appropriate;
do not grant those claims to compiler-only receipts. General-language/code source
approval, measured mixture/token budgets, Gen2 tokenizer fitting, model training,
confidential evaluation custody and owner backup/restore evidence remain open.
