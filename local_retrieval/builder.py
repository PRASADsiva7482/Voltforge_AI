"""Reproducibly build the project-owned lexical index from approved facts."""

from __future__ import annotations

from collections import Counter
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from data_governance import load_source_registry, require_approved_source
from electronics_corpus import ElectronicsCorpus
from electronics_corpus.store import CATALOG_PATH, sha256_file
from local_retrieval.schema import (
    RETRIEVAL_INDEX_ID,
    RETRIEVAL_INDEX_VERSION,
    RETRIEVAL_POLICY_ID,
    IndexedFact,
    RetrievalContractError,
    RetrievalIndex,
    RetrievalIndexChunk,
    canonical_json,
    load_retrieval_policy,
    sha256_json,
)


BUILDER_PATH = Path(__file__).resolve()
_TOKEN = re.compile(r"[a-z0-9]+")
_ALPHA_NUMERIC = re.compile(r"[a-z]+|[0-9]+")


def normalize_filter_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())[:200]


def tokenize(value: object, stop_words: Iterable[str] = ()) -> tuple[str, ...]:
    blocked = {item.casefold() for item in stop_words}
    tokens: list[str] = []
    for raw in _TOKEN.findall(str(value or "").casefold()):
        if raw not in blocked:
            tokens.append(raw[:100])
        segments = _ALPHA_NUMERIC.findall(raw)
        if len(segments) > 1:
            tokens.extend(
                item[:100]
                for item in segments
                if item not in blocked and (len(item) >= 2 or item.isdigit())
            )
    return tuple(tokens)


def _json_pointer(value: object) -> str:
    return str(value).replace("~", "~0").replace("/", "~1")


def _strict_json_size(value: Any) -> int:
    try:
        return len(canonical_json(value))
    except (TypeError, ValueError) as error:
        raise RetrievalContractError(
            "RETRIEVAL_SOURCE_FACT_INVALID",
            "A curated fact cannot be represented as strict JSON.",
        ) from error


def _flatten_value(value: Any, pointer: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, Mapping):
        if not value:
            return [(pointer, {})]
        flattened: list[tuple[str, Any]] = []
        for key in sorted(value, key=str):
            child_pointer = f"{pointer}/{_json_pointer(key)}"
            flattened.extend(_flatten_value(value[key], child_pointer))
        return flattened
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = list(value)
        if not values or (
            all(item is None or isinstance(item, (str, int, float, bool)) for item in values)
            and _strict_json_size(values) <= 800
        ):
            return [(pointer, values)]
        flattened = []
        for index, item in enumerate(values):
            flattened.extend(_flatten_value(item, f"{pointer}/{index}"))
        return flattened
    return [(pointer, value)]


def _facts(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for claim in record["claims"]:
        for pointer, value in _flatten_value(claim.get("value")):
            identity = {
                "recordId": record["recordId"],
                "claimId": claim["claimId"],
                "pointer": pointer,
                "value": value,
            }
            rows.append(
                IndexedFact(
                    factId=f"fact:{hashlib.sha256(canonical_json(identity)).hexdigest()[:24]}",
                    claimId=claim["claimId"],
                    property=claim["property"],
                    pointer=pointer,
                    value=value,
                    unit=claim.get("unit"),
                    status=claim["status"],
                    conditions=list(claim.get("conditions") or []),
                    evidenceRefs=sorted(set(claim["evidenceRefs"])),
                ).model_dump(mode="json")
            )
    rows.sort(
        key=lambda item: (
            item["property"],
            item["claimId"],
            item["pointer"],
            sha256_json(item.get("value")),
        )
    )
    return rows


def _subject_keys(record: Mapping[str, Any]) -> set[str]:
    subject = record["subject"]
    values = {
        record["recordId"],
        subject["subjectId"],
        subject["familyId"],
        subject["name"],
        subject["variant"],
        *subject.get("aliases", []),
    }
    return {key for value in values if (key := normalize_filter_key(value))}


def _record_references(record: Mapping[str, Any], known_ids: set[str]) -> set[str]:
    references = {
        item["targetRecordId"]
        for item in record.get("relations", [])
        if item.get("targetRecordId") in known_ids
    }

    def visit(value: Any) -> None:
        if isinstance(value, str) and value in known_ids:
            references.add(value)
        elif isinstance(value, Mapping):
            for child in value.values():
                visit(child)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for child in value:
                visit(child)

    for claim in record["claims"]:
        visit(claim.get("value"))
    return references


def _filter_keys(
    record: Mapping[str, Any], records_by_id: Mapping[str, Mapping[str, Any]]
) -> tuple[list[str], list[str]]:
    board_keys: set[str] = set()
    component_keys: set[str] = set()
    if record["recordType"] == "board":
        board_keys.update(_subject_keys(record))
    if record["recordType"] == "component":
        component_keys.update(_subject_keys(record))
    for record_id in _record_references(record, set(records_by_id)):
        referenced = records_by_id[record_id]
        if referenced["recordType"] == "board":
            board_keys.update(_subject_keys(referenced))
        elif referenced["recordType"] == "component":
            component_keys.update(_subject_keys(referenced))
    return sorted(board_keys), sorted(component_keys)


def _weighted_terms(
    record: Mapping[str, Any],
    facts: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> Counter[str]:
    ranking = policy["ranking"]
    stop_words = ranking["stopWords"]
    subject = record["subject"]
    weighted: Counter[str] = Counter()

    def add(value: object, weight_name: str) -> None:
        weight = int(ranking[weight_name])
        for token in tokenize(value, stop_words):
            weighted[token] += weight

    add(record["recordType"], "recordTypeWeight")
    add(subject["name"], "subjectNameWeight")
    add(subject["variant"], "subjectVariantWeight")
    for alias in subject.get("aliases", []):
        add(alias, "aliasWeight")
    for tag in record.get("tags", []):
        add(tag, "tagWeight")
    for fact in facts:
        add(fact["property"], "claimPropertyWeight")
        add(fact["pointer"], "factPointerWeight")
        add(json.dumps(fact.get("value"), ensure_ascii=False, sort_keys=True), "factValueWeight")
        for condition in fact.get("conditions", []):
            add(condition, "conditionWeight")
    for item in evidence:
        add(item["title"], "evidenceTitleWeight")
        add(item["locator"], "evidenceTitleWeight")
    return weighted


def _chunk_content(
    record: Mapping[str, Any],
    facts: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    revision = record["effectiveRevision"]
    return {
        "recordId": record["recordId"],
        "recordType": record["recordType"],
        "supportStatus": record["supportStatus"],
        "subject": {
            key: record["subject"][key]
            for key in ("subjectId", "familyId", "name", "variant")
        },
        "source": {
            "sourceId": record["provenance"]["sourceId"],
            "sourceRevision": record["provenance"]["sourceRevision"],
            "recordRevision": revision["revision"],
            "validFrom": revision["validFrom"],
            "recordId": record["recordId"],
        },
        "facts": list(facts),
        "evidence": list(evidence),
    }


def _make_chunk(
    record: Mapping[str, Any],
    facts: Sequence[Mapping[str, Any]],
    evidence_by_id: Mapping[str, Mapping[str, Any]],
    board_keys: list[str],
    component_keys: list[str],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    evidence_ids = sorted(
        {reference for fact in facts for reference in fact.get("evidenceRefs", [])}
    )
    evidence = [dict(evidence_by_id[item]) for item in evidence_ids]
    content = _chunk_content(record, facts, evidence)
    maximum_bytes = int(policy["limits"]["maximumChunkContentBytes"])
    if len(canonical_json(content)) > maximum_bytes:
        raise RetrievalContractError(
            "RETRIEVAL_CHUNK_TOO_LARGE",
            "A curated fact cannot fit the bounded retrieval chunk contract.",
        )
    identity = {
        "recordId": record["recordId"],
        "factIds": [item["factId"] for item in facts],
    }
    terms = _weighted_terms(record, facts, evidence, policy)
    value = {
        "chunkId": f"vf-retrieval-chunk-v1-{sha256_json(identity)[:24]}",
        **content,
        "boardKeys": board_keys,
        "componentKeys": component_keys,
        "termFrequencies": dict(sorted(terms.items())),
        "documentLength": sum(terms.values()),
        "contentSha256": sha256_json(content),
    }
    return RetrievalIndexChunk.model_validate(value).model_dump(mode="json")


def build_chunks(corpus: ElectronicsCorpus, policy: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = sorted(corpus.records, key=lambda item: item["recordId"])
    records_by_id = {item["recordId"]: item for item in records}
    maximum_facts = int(policy["limits"]["maximumFactsPerChunk"])
    chunks: list[dict[str, Any]] = []
    for record in records:
        evidence_by_id = {
            item["evidenceId"]: {
                key: item[key]
                for key in (
                    "evidenceId",
                    "evidenceKind",
                    "publisher",
                    "title",
                    "documentRevision",
                    "locator",
                    "verifiedAt",
                )
            }
            for item in record["provenance"]["evidence"]
        }
        board_keys, component_keys = _filter_keys(record, records_by_id)
        facts = _facts(record)
        for start in range(0, len(facts), maximum_facts):
            group = facts[start : start + maximum_facts]
            chunks.append(
                _make_chunk(
                    record,
                    group,
                    evidence_by_id,
                    board_keys,
                    component_keys,
                    policy,
                )
            )
    chunks.sort(key=lambda item: (item["recordId"], item["chunkId"]))
    return chunks


def _source_entry(source_id: str) -> dict[str, Any]:
    registry = load_source_registry()
    for source in registry["sources"]:
        if source.get("sourceId") == source_id:
            return dict(source)
    raise RetrievalContractError(
        "RETRIEVAL_SOURCE_NOT_APPROVED", "The retrieval source is not registered."
    )


def build_index(catalog_path: str | Path = CATALOG_PATH) -> RetrievalIndex:
    policy = load_retrieval_policy()
    source_policy = policy["source"]
    try:
        approved = require_approved_source(
            source_policy["sourceId"], source_policy["requiredUse"]
        )
    except Exception as error:
        raise RetrievalContractError(
            "RETRIEVAL_SOURCE_NOT_APPROVED",
            "The curated source is not approved for runtime retrieval.",
        ) from error
    if approved.get("revision") != source_policy["sourceRevision"]:
        raise RetrievalContractError(
            "RETRIEVAL_SOURCE_REVISION_MISMATCH",
            "The approved retrieval source revision does not match policy.",
        )
    corpus = ElectronicsCorpus(catalog_path)
    if (
        corpus.catalog.get("corpusId") != source_policy["corpusId"]
        or corpus.catalog.get("version") != source_policy["corpusVersion"]
    ):
        raise RetrievalContractError(
            "RETRIEVAL_SOURCE_REVISION_MISMATCH",
            "The curated corpus revision does not match retrieval policy.",
        )
    chunks = build_chunks(corpus, policy)
    frequencies: Counter[str] = Counter()
    for chunk in chunks:
        frequencies.update(chunk["termFrequencies"].keys())
    with localcontext() as context:
        context.prec = int(policy["algorithm"]["decimalPrecision"])
        average = Decimal(sum(item["documentLength"] for item in chunks)) / Decimal(
            len(chunks)
        )
        average_text = format(average.quantize(Decimal("0.000001")), "f")
    source_entry = _source_entry(source_policy["sourceId"])
    raw = {
        "schemaVersion": 1,
        "contractVersion": "1.0.0",
        "indexId": RETRIEVAL_INDEX_ID,
        "indexVersion": RETRIEVAL_INDEX_VERSION,
        "policyId": RETRIEVAL_POLICY_ID,
        "policySha256": policy["policySha256"],
        "algorithm": policy["algorithm"]["name"],
        "sourceId": source_policy["sourceId"],
        "sourceRevision": source_policy["sourceRevision"],
        "sourceEntrySha256": sha256_json(source_entry),
        "sourceCatalogSha256": sha256_file(Path(catalog_path)),
        "corpusContentSha256": sha256_json(
            sorted(corpus.records, key=lambda item: item["recordId"])
        ),
        "builderSha256": sha256_file(BUILDER_PATH),
        "recordCount": len(corpus.records),
        "chunkCount": len(chunks),
        "averageDocumentLength": average_text,
        "documentFrequencies": dict(sorted(frequencies.items())),
        "chunks": chunks,
        "rawSourceDocumentsStored": False,
        "embeddingsPresent": False,
        "indexSha256": "0" * 64,
    }
    unsigned = dict(raw)
    unsigned.pop("indexSha256")
    raw["indexSha256"] = sha256_json(unsigned)
    return RetrievalIndex.model_validate(raw)
