# Shared development database verification

Checked 22 October 2026 against the configured VPN endpoint `100.73.197.23:3306/voltforge_db` in `dev/development`.

All six SQL scripts currently present in the BL/UI/AI checkout are reflected in the live database:

| Script | Verified change |
| --- | --- |
| VFOPT_X_001_F001 | `projects.document_revision`, bigint, non-null, default 0 |
| VFOPT_X_001_F005 | Owner/activity and public/creation project indexes, including descending order |
| VF_CHAT_001 | Conversation/message columns, ownership relations, sequence and retry uniqueness, indexes |
| VF_MEMORY_001 | Approved preference storage, owner relation, uniqueness and value constraint |
| VF_DOCUMENT_001 | Document/version/chunk columns, ownership relations, version/checksum uniqueness and indexes |
| VF_EMBEDDING_001 | Model-bound derived vectors, unique chunk/model pairs, source deletion cascade |

The document migration was applied during AIR-04; the derived-vector migration was applied during AIR-05, both with schema snapshots. Chat, memory and project changes were already present. No current script is pending application or commit. A fresh connection confirmed table visibility. MySQL DDL commits implicitly; transactional verification records were intentionally rolled back.

Aggregate rows at verification: 7 users, 5 projects, 5 code files, 71 components, 2 conversations, 30 messages, 3 approved memories, 0 project shares. Document/version/chunk tables each contain 0 rows: test documents were synthetic verification data and deliberately cleaned up. Users must upload their own reference documents; no demo corpus is seeded.

All 14 inspected foreign-key relationships have zero orphan rows. Chat insert/update/ordered-read/cascade tests passed; memory insert/read/update/invalid-value rejection/cascade tests passed; document and embedding insert/read/cascade tests passed. These use temporary verification records and preserve existing user data. Counts establish current availability, not a historical backup comparison or proof that deleted data can be recovered.

`flyway_schema_history` contains successful legacy versions 1–4. Their original migration files are absent from this checkout, so Flyway remains disabled; the additive scripts are verified against the actual schema rather than inserted into legacy Flyway history.

Reproduce the read-only schema/count/orphan audit with `python scripts/audit_schema.py` from Voltforge_BL using the configured development connection. It writes `target/shared-db-schema-audit.json` without account identifiers, credentials or private message contents. Feature verification helpers: `chat_schema.py --verify`, `memory_schema.py`, and `document_schema.py --verify`.

Database readiness does not establish model-generation readiness. AIR-04/05 native-flow checks passed; broader load acceptance and AIR-02/03's missing live search-provider credential remain tracked separately in the backlog.
