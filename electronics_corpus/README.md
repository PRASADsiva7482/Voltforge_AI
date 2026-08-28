# VoltForge curated electronics corpus

This package is the project-owned, offline factual authority introduced by
VFAI-008. It contains no model-generated or copied source prose. Records keep
only normalized facts, exact variant identity, effective revision, and bounded
references to manufacturer, maintainer, toolchain, or VoltForge engineering
evidence.

The v1 corpus has eight independently checksummed JSONL packs: boards, pin
maps, components, wiring recipes, firmware APIs, compiler diagnostics,
simulation behavior, and safety constraints. `catalog.v1.json` binds each pack,
the executable builder, and `knowledge-record.schema.json` by SHA-256.

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

Ranked lexical retrieval is intentionally deferred to VFAI-022. That future
index must consume these validated packs and preserve their record IDs,
revisions, evidence, conflicts, and unknown results.
