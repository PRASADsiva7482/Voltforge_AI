"""Checksum-verified local store and exact electronics knowledge lookup."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

from electronics_corpus.schema import (
    CORPUS_CONTRACT_VERSION,
    CORPUS_ROOT,
    KnowledgeContractError,
    validate_knowledge_record,
)


CATALOG_PATH = CORPUS_ROOT / "catalog.v1.json"
SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_lookup_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


@dataclass(frozen=True)
class LookupResult:
    status: str
    reasonCode: str
    query: str
    records: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reasonCode": self.reasonCode,
            "query": self.query,
            "records": [dict(item) for item in self.records],
        }


class ElectronicsCorpus:
    """Load immutable packs and provide exact, non-ranking local lookup."""

    def __init__(self, catalog_path: str | Path = CATALOG_PATH):
        self.catalog_path = Path(catalog_path).resolve()
        self.root = self.catalog_path.parent.resolve()
        self.catalog = self._load_catalog()
        self.records = self._load_records()
        self.by_id = {item["recordId"]: item for item in self.records}
        self._validate_relations()
        self.alias_index = self._build_alias_index()

    def _read_json(self, path: Path, code: str) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise KnowledgeContractError(code, f"Invalid corpus JSON: {path}") from error
        if not isinstance(value, dict):
            raise KnowledgeContractError(code, f"Corpus JSON must be an object: {path}")
        return value

    def _resolve_pack_path(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as error:
            raise KnowledgeContractError(
                "KNOWLEDGE_PACK_PATH_INVALID", f"Pack escapes corpus root: {relative}"
            ) from error
        return path

    def _load_catalog(self) -> dict[str, Any]:
        catalog = self._read_json(self.catalog_path, "KNOWLEDGE_CATALOG_INVALID")
        required = {
            "schemaVersion",
            "corpusId",
            "version",
            "effectiveFrom",
            "source",
            "recordSchema",
            "builder",
            "recordCount",
            "recordTypeCounts",
            "packs",
        }
        if catalog.get("schemaVersion") != 1 or not required.issubset(catalog):
            raise KnowledgeContractError(
                "KNOWLEDGE_CATALOG_INVALID", "Corpus catalog has an unsupported contract."
            )
        if catalog.get("version") != CORPUS_CONTRACT_VERSION:
            raise KnowledgeContractError(
                "KNOWLEDGE_CATALOG_VERSION_UNSUPPORTED", "Corpus version is not supported."
            )
        for descriptor_name in ("recordSchema", "builder"):
            descriptor = catalog.get(descriptor_name)
            if not isinstance(descriptor, dict):
                raise KnowledgeContractError(
                    "KNOWLEDGE_CATALOG_INVALID", f"Missing {descriptor_name} descriptor."
                )
            path = self._resolve_pack_path(str(descriptor.get("path") or ""))
            expected = descriptor.get("sha256")
            if not path.is_file() or not isinstance(expected, str) or sha256_file(path) != expected:
                raise KnowledgeContractError(
                    "KNOWLEDGE_PIPELINE_REVISION_MISMATCH",
                    f"Corpus {descriptor_name} checksum does not match the catalog.",
                )
        return catalog

    def _load_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        type_counts: dict[str, int] = {}
        for pack in self.catalog.get("packs", []):
            if not isinstance(pack, dict):
                raise KnowledgeContractError("KNOWLEDGE_CATALOG_INVALID", "Invalid pack descriptor.")
            path = self._resolve_pack_path(str(pack.get("path") or ""))
            expected_hash = pack.get("sha256")
            if not path.is_file() or not isinstance(expected_hash, str) or not SHA256_PATTERN.fullmatch(expected_hash):
                raise KnowledgeContractError("KNOWLEDGE_PACK_MISSING", f"Knowledge pack is unavailable: {path}")
            if sha256_file(path) != expected_hash:
                raise KnowledgeContractError(
                    "KNOWLEDGE_PACK_CHECKSUM_MISMATCH", f"Knowledge pack changed: {path}"
                )
            pack_records: list[dict[str, Any]] = []
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise KnowledgeContractError(
                            "KNOWLEDGE_PACK_JSON_INVALID", f"Invalid JSON at {path}:{line_number}"
                        ) from error
                    validated = validate_knowledge_record(raw)
                    if validated["recordType"] != pack.get("recordType"):
                        raise KnowledgeContractError(
                            "KNOWLEDGE_PACK_TYPE_MISMATCH", f"Mixed record types in {path}"
                        )
                    pack_records.append(validated)
            if len(pack_records) != pack.get("recordCount"):
                raise KnowledgeContractError(
                    "KNOWLEDGE_PACK_COUNT_MISMATCH", f"Knowledge pack count changed: {path}"
                )
            records.extend(pack_records)
            type_counts[str(pack.get("recordType"))] = len(pack_records)
        ids = [item["recordId"] for item in records]
        if len(ids) != len(set(ids)):
            raise KnowledgeContractError("KNOWLEDGE_RECORD_ID_DUPLICATE", "Record IDs must be unique.")
        if len(records) != self.catalog.get("recordCount") or type_counts != self.catalog.get("recordTypeCounts"):
            raise KnowledgeContractError(
                "KNOWLEDGE_CATALOG_COUNT_MISMATCH", "Catalog record totals do not match packs."
            )
        return records

    def _validate_relations(self) -> None:
        ids = set(self.by_id)
        for record in self.records:
            missing = sorted(
                relation["targetRecordId"]
                for relation in record.get("relations", [])
                if relation["targetRecordId"] not in ids
            )
            if missing:
                raise KnowledgeContractError(
                    "KNOWLEDGE_RELATION_UNRESOLVED",
                    f"{record['recordId']} references missing records: {missing}",
                )

    def _build_alias_index(self) -> dict[tuple[str, str], tuple[str, ...]]:
        mutable: dict[tuple[str, str], set[str]] = {}
        for record in self.records:
            subject = record["subject"]
            names: Iterable[str] = (
                record["recordId"],
                subject["subjectId"],
                subject["familyId"],
                subject["name"],
                *subject.get("aliases", []),
            )
            for name in names:
                key = (record["recordType"], normalize_lookup_key(name))
                mutable.setdefault(key, set()).add(record["recordId"])
        return {key: tuple(sorted(value)) for key, value in mutable.items()}

    def lookup(
        self, record_type: str, query: str, *, variant: str | None = None
    ) -> LookupResult:
        ids = self.alias_index.get((record_type, normalize_lookup_key(query)), ())
        candidates = [self.by_id[item] for item in ids]
        if variant is not None:
            variant_key = normalize_lookup_key(variant)
            candidates = [
                item
                for item in candidates
                if normalize_lookup_key(item["subject"]["variant"]) == variant_key
                or variant_key
                in {normalize_lookup_key(alias) for alias in item["subject"].get("aliases", [])}
            ]
        if not candidates:
            return LookupResult("unknown", "unsupported-or-missing-evidence", query)
        if len(candidates) > 1:
            return LookupResult("ambiguous", "variant-required", query, tuple(candidates))
        record = candidates[0]
        if record["supportStatus"] != "supported":
            return LookupResult("unknown", record["supportStatus"], query, (record,))
        return LookupResult("found", "exact-evidenced-match", query, (record,))

    def record(self, record_id: str) -> LookupResult:
        item = self.by_id.get(record_id)
        if item is None:
            return LookupResult("unknown", "record-not-found", record_id)
        if item["supportStatus"] != "supported":
            return LookupResult("unknown", item["supportStatus"], record_id, (item,))
        return LookupResult("found", "exact-record-id", record_id, (item,))

    def claims(self, record: dict[str, Any]) -> dict[str, Any]:
        return {item["property"]: item.get("value") for item in record["claims"]}

    def lookup_wiring_recipe(self, board: str, component: str) -> LookupResult:
        board_result = self.lookup("board", board)
        if board_result.status != "found":
            return LookupResult(board_result.status, f"board-{board_result.reasonCode}", board, board_result.records)
        component_result = self.lookup("component", component)
        if component_result.status != "found":
            return LookupResult(
                component_result.status,
                f"component-{component_result.reasonCode}",
                component,
                component_result.records,
            )
        board_id = board_result.records[0]["recordId"]
        component_id = component_result.records[0]["recordId"]
        matches = []
        for record in self.records:
            if record["recordType"] != "wiring-recipe":
                continue
            claims = self.claims(record)
            if claims.get("board-record-id") == board_id and claims.get("component-record-id") == component_id:
                matches.append(record)
        if not matches:
            return LookupResult("unknown", "wiring-recipe-not-curated", f"{board}+{component}")
        if len(matches) > 1:
            return LookupResult("ambiguous", "wiring-variant-required", f"{board}+{component}", tuple(matches))
        return LookupResult("found", "exact-evidenced-match", f"{board}+{component}", (matches[0],))


@lru_cache(maxsize=1)
def get_electronics_corpus() -> ElectronicsCorpus:
    return ElectronicsCorpus()
