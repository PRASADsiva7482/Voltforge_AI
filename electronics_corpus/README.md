# VoltForge curated electronics corpus

This package is the project-owned, offline factual authority introduced by
VFAI-008. It contains no model-generated or copied source prose. Records keep
only normalized facts, exact variant identity, effective revision, and bounded
references to manufacturer, maintainer, toolchain, or VoltForge engineering
evidence.

The v1.1 corpus has eight independently checksummed JSONL packs: boards, pin
maps, components, wiring recipes, firmware APIs, compiler diagnostics,
simulation behavior, and safety constraints. `catalog.v1.json` binds each pack,
the executable builder, and `knowledge-record.schema.json` by SHA-256.

The current release contains 61 records. It promotes exact Arduino Nano Every
(ABX00028), Leonardo (A000057), and Micro (A000053) variants while retaining
explicit variant-required records for generic Nano, ESP32, ESP32-S3, ESP8266,
and STM32 labels. The UI hardware coverage report is a separate parity check:
selectable artwork is not equivalent to verified electrical support.

Lookup is exact and deterministic. A supported exact variant returns `found`;
a family name matching conflicting variants returns `ambiguous`; an unsupported
subject, missing evidence, or a deliberately generic component returns
`unknown`. The store never substitutes Arduino UNO, a generic I2C device, or
any other convenient default.

Generate the checked-in corpus intentionally:

```powershell
python tools/build_electronics_corpus.py --write
```

Verify it without changing files:

```powershell
python tools/build_electronics_corpus.py
```

VFAI-022 now builds ranked lexical retrieval exclusively from these validated
packs. It preserves record IDs, source and effective revisions, evidence,
conflicts, and unknown results as exact typed chunks, and rejects the complete
index if any corpus or index boundary becomes stale. See
[`../local_retrieval/README.md`](../local_retrieval/README.md).

The VFAI-023 internet evidence cache is deliberately outside this package.
Web results cannot be copied into this corpus by the retrieval service;
admission requires a separate provenance, license, governance, and checksum
review.
