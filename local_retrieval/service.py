"""Offline ranked retrieval over the checksum-bound curated fact index."""

from __future__ import annotations

from collections import Counter
from decimal import Decimal, localcontext
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

from data_governance import load_source_registry, require_approved_source
from electronics_corpus import ElectronicsCorpus
from electronics_corpus.store import CATALOG_PATH, sha256_file
from local_retrieval.builder import (
    BUILDER_PATH,
    build_index,
    normalize_filter_key,
    tokenize,
)
from local_retrieval.schema import (
    RETRIEVAL_INDEX_VERSION,
    RETRIEVAL_INDEX_ID,
    RETRIEVAL_INDEX_PATH,
    RETRIEVAL_POLICY_ID,
    RetrievalContractError,
    RetrievalIndex,
    RetrievalQuery,
    RetrievalResponse,
    load_retrieval_policy,
    sha256_json,
    validate_retrieval_index,
    validate_retrieval_response,
)
from observability import observability


def _source_entry(source_id: str) -> dict[str, Any]:
    for item in load_source_registry()["sources"]:
        if item.get("sourceId") == source_id:
            return dict(item)
    raise RetrievalContractError(
        "RETRIEVAL_SOURCE_NOT_APPROVED", "The retrieval source is not registered."
    )


def _chunk_content(chunk: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: chunk[key]
        for key in (
            "recordId",
            "recordType",
            "supportStatus",
            "subject",
            "source",
            "facts",
            "evidence",
        )
    }


class LocalRetrievalService:
    """Load one immutable index and return bounded exact-source chunks."""

    def __init__(
        self,
        index_path: str | Path = RETRIEVAL_INDEX_PATH,
        catalog_path: str | Path = CATALOG_PATH,
    ):
        self.index_path = Path(index_path).resolve()
        self.catalog_path = Path(catalog_path).resolve()
        self.policy = load_retrieval_policy()
        self.corpus = ElectronicsCorpus(self.catalog_path)
        self.index = self._load_and_verify_index()

    def _read_index(self) -> RetrievalIndex:
        try:
            raw = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise RetrievalContractError(
                "RETRIEVAL_INDEX_UNAVAILABLE", "The curated local index is unreadable."
            ) from error
        if not isinstance(raw, dict):
            raise RetrievalContractError(
                "RETRIEVAL_INDEX_INVALID", "The curated local index must be an object."
            )
        return validate_retrieval_index(raw)

    def _load_and_verify_index(self) -> RetrievalIndex:
        index = self._read_index()
        raw = index.model_dump(mode="json")
        unsigned = dict(raw)
        declared_digest = unsigned.pop("indexSha256")
        if declared_digest != sha256_json(unsigned):
            raise RetrievalContractError(
                "RETRIEVAL_INDEX_CHECKSUM_MISMATCH",
                "The curated local index checksum is invalid.",
            )
        if (
            index.policyId != self.policy["policyId"]
            or index.policySha256 != self.policy["policySha256"]
            or index.indexId != self.policy["indexId"]
        ):
            raise RetrievalContractError(
                "RETRIEVAL_INDEX_POLICY_MISMATCH",
                "The curated local index was built for a different policy.",
            )
        source_policy = self.policy["source"]
        try:
            approved = require_approved_source(
                source_policy["sourceId"], source_policy["requiredUse"]
            )
        except Exception as error:
            raise RetrievalContractError(
                "RETRIEVAL_SOURCE_NOT_APPROVED",
                "The indexed source is no longer approved for runtime retrieval.",
            ) from error
        if (
            approved.get("revision") != index.sourceRevision
            or sha256_json(_source_entry(index.sourceId)) != index.sourceEntrySha256
        ):
            raise RetrievalContractError(
                "RETRIEVAL_INDEX_SOURCE_MISMATCH",
                "The approved source revision changed after this index was built.",
            )
        if (
            sha256_file(self.catalog_path) != index.sourceCatalogSha256
            or sha256_json(sorted(self.corpus.records, key=lambda item: item["recordId"]))
            != index.corpusContentSha256
            or sha256_file(BUILDER_PATH) != index.builderSha256
        ):
            raise RetrievalContractError(
                "RETRIEVAL_INDEX_SOURCE_MISMATCH",
                "The corpus catalog, content, or index builder changed after indexing.",
            )
        for chunk in index.chunks:
            if sha256_json(_chunk_content(chunk.model_dump(mode="json"))) != chunk.contentSha256:
                raise RetrievalContractError(
                    "RETRIEVAL_INDEX_CONTENT_MISMATCH",
                    "An indexed retrieval chunk no longer matches its content hash.",
                )
        regenerated = build_index(self.catalog_path).model_dump(mode="json")
        if regenerated != raw:
            raise RetrievalContractError(
                "RETRIEVAL_INDEX_STALE",
                "The checked local index does not match a reproducible rebuild.",
            )
        return index

    def _resolved_filter_keys(self, record_type: str, values: Iterable[str]) -> set[str]:
        keys = {
            normalized
            for item in values
            if (normalized := normalize_filter_key(item))
        }
        for item in values:
            result = self.corpus.lookup(record_type, item)
            records = list(result.records)
            if not records:
                query_terms = {
                    term
                    for term in tokenize(item)
                    if len(term) >= 3 or (term.isdigit() and len(term) >= 2)
                }
                ranked = []
                for record in self.corpus.records:
                    if record["recordType"] != record_type:
                        continue
                    subject = record["subject"]
                    subject_terms = set(
                        tokenize(
                            " ".join(
                                [
                                    subject["name"],
                                    subject["variant"],
                                    *subject.get("aliases", []),
                                ]
                            )
                        )
                    )
                    overlap = len(query_terms.intersection(subject_terms))
                    if overlap >= max(1, (len(query_terms) + 1) // 2):
                        ranked.append((overlap, record["recordId"], record))
                if ranked:
                    best = max(item[0] for item in ranked)
                    records = [item[2] for item in ranked if item[0] == best]
            for record in records:
                subject = record["subject"]
                for value in (
                    record["recordId"],
                    subject["subjectId"],
                    subject["name"],
                    subject["variant"],
                    *subject.get("aliases", []),
                ):
                    if normalized := normalize_filter_key(value):
                        keys.add(normalized)
        return keys

    def search(self, query: RetrievalQuery | Mapping[str, Any]) -> RetrievalResponse:
        """Run curated retrieval while recording content-free search facts."""
        started = time.perf_counter()
        try:
            response = self._search(query)
        except Exception:
            observability.record_retrieval(
                status="failed",
                failure_code="LOCAL_RETRIEVAL_FAILED",
                duration_ms=(time.perf_counter() - started) * 1000,
            )
            raise
        observability.record_retrieval(
            status=response.status,
            failure_code=response.reasonCode,
            duration_ms=(time.perf_counter() - started) * 1000,
        )
        return response

    def _search(self, query: RetrievalQuery | Mapping[str, Any]) -> RetrievalResponse:
        request = query if isinstance(query, RetrievalQuery) else RetrievalQuery.model_validate(query)
        stop_words = self.policy["ranking"]["stopWords"]
        maximum_query_tokens = int(self.policy["limits"]["maximumQueryTokens"])
        query_tokens = tokenize(request.text, stop_words)[:maximum_query_tokens]
        query_hash = hashlib.sha256(request.text.encode("utf-8")).hexdigest()
        filters = {
            "boardFilters": sorted(request.boardFilters, key=str.casefold),
            "componentFilters": sorted(request.componentFilters, key=str.casefold),
            "recordTypes": sorted(request.recordTypes),
        }
        if not query_tokens:
            return self._response(
                status="no-results",
                reason_code="RETRIEVAL_QUERY_HAS_NO_INDEX_TERMS",
                query_hash=query_hash,
                filters=filters,
            )
        board_keys = self._resolved_filter_keys("board", request.boardFilters)
        component_keys = self._resolved_filter_keys("component", request.componentFilters)
        query_counts = Counter(query_tokens)
        candidates: list[tuple[Decimal, str, dict[str, Any], list[str]]] = []
        with localcontext() as context:
            context.prec = int(self.policy["algorithm"]["decimalPrecision"])
            total_documents = Decimal(self.index.chunkCount)
            average_length = Decimal(self.index.averageDocumentLength)
            k1 = Decimal(self.policy["algorithm"]["k1"])
            b = Decimal(self.policy["algorithm"]["b"])
            filter_boost = Decimal(self.policy["ranking"]["filterMatchBoost"])
            exact_subject_boost = Decimal(self.policy["ranking"]["exactSubjectBoost"])
            normalized_query = normalize_filter_key(request.text)
            for chunk_model in self.index.chunks:
                chunk = chunk_model.model_dump(mode="json")
                if request.recordTypes and chunk["recordType"] not in request.recordTypes:
                    continue
                chunk_board_keys = set(chunk["boardKeys"])
                chunk_component_keys = set(chunk["componentKeys"])
                if board_keys and chunk_board_keys and not board_keys.intersection(chunk_board_keys):
                    continue
                if (
                    component_keys
                    and chunk_component_keys
                    and not component_keys.intersection(chunk_component_keys)
                ):
                    continue
                matched = sorted(set(query_counts).intersection(chunk["termFrequencies"]))
                if not matched:
                    continue
                document_length = Decimal(chunk["documentLength"])
                score = Decimal(0)
                for term in matched:
                    document_frequency = Decimal(self.index.documentFrequencies[term])
                    inverse_frequency = (
                        Decimal(1)
                        + (total_documents - document_frequency + Decimal("0.5"))
                        / (document_frequency + Decimal("0.5"))
                    ).ln()
                    term_frequency = Decimal(chunk["termFrequencies"][term])
                    denominator = term_frequency + k1 * (
                        Decimal(1) - b + b * document_length / average_length
                    )
                    score += (
                        inverse_frequency
                        * term_frequency
                        * (k1 + Decimal(1))
                        / denominator
                        * Decimal(query_counts[term])
                    )
                if board_keys.intersection(chunk_board_keys):
                    score += filter_boost
                if component_keys.intersection(chunk_component_keys):
                    score += filter_boost
                subject = chunk["subject"]
                subject_keys = {
                    normalize_filter_key(subject["name"]),
                    normalize_filter_key(subject["variant"]),
                }
                if any(key and key in normalized_query for key in subject_keys):
                    score += exact_subject_boost
                candidates.append((score, chunk["chunkId"], chunk, matched))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        maximum_per_record = int(
            self.policy["limits"]["maximumChunksPerRecordInResponse"]
        )
        maximum_results = min(
            request.maximumResults, int(self.policy["limits"]["maximumResults"])
        )
        selected: list[tuple[Decimal, dict[str, Any], list[str]]] = []
        record_counts: Counter[str] = Counter()
        for score, _chunk_id, chunk, matched in candidates:
            if record_counts[chunk["recordId"]] >= maximum_per_record:
                continue
            record_counts[chunk["recordId"]] += 1
            selected.append((score, chunk, matched))
            if len(selected) >= maximum_results:
                break
        if not selected:
            return self._response(
                status="no-results",
                reason_code="RETRIEVAL_NO_MATCH",
                query_hash=query_hash,
                filters=filters,
                candidate_count=len(candidates),
            )
        results = [
            self._result(rank, score, chunk, matched)
            for rank, (score, chunk, matched) in enumerate(selected, 1)
        ]
        return self._response(
            status="complete",
            reason_code="RETRIEVAL_COMPLETE",
            query_hash=query_hash,
            filters=filters,
            candidate_count=len(candidates),
            results=results,
        )

    def _result(
        self,
        rank: int,
        score: Decimal,
        chunk: Mapping[str, Any],
        matched: Sequence[str],
    ) -> dict[str, Any]:
        evidence = list(chunk["evidence"])[
            : int(self.policy["limits"]["maximumEvidencePerResult"])
        ]
        citation_identity = {
            "chunkId": chunk["chunkId"],
            "contentSha256": chunk["contentSha256"],
        }
        claim_ids = sorted({item["claimId"] for item in chunk["facts"]})
        citation = {
            "citationId": f"citation:local:{sha256_json(citation_identity)[:24]}",
            "sourceId": chunk["source"]["sourceId"],
            "sourceRevision": chunk["source"]["sourceRevision"],
            "recordId": chunk["recordId"],
            "title": f"{chunk['subject']['name']} — {chunk['subject']['variant']}",
            "locator": f"{chunk['recordId']}#claims={','.join(claim_ids)}",
            "contentSha256": chunk["contentSha256"],
        }
        result_identity = {
            "rank": rank,
            "chunkId": chunk["chunkId"],
            "citationId": citation["citationId"],
        }
        return {
            "resultId": f"retrieval:local:{sha256_json(result_identity)[:24]}",
            "rank": rank,
            "score": format(score.quantize(Decimal("0.000001")), "f"),
            "matchedTerms": list(matched)[
                : int(self.policy["limits"]["maximumMatchedTermsPerResult"])
            ],
            "chunkId": chunk["chunkId"],
            "recordId": chunk["recordId"],
            "recordType": chunk["recordType"],
            "supportStatus": chunk["supportStatus"],
            "subject": chunk["subject"],
            "source": chunk["source"],
            "facts": chunk["facts"],
            "evidence": evidence,
            "contentSha256": chunk["contentSha256"],
            "citation": citation,
        }

    def _response(
        self,
        *,
        status: str,
        reason_code: str,
        query_hash: str,
        filters: Mapping[str, list[str]],
        candidate_count: int = 0,
        results: Sequence[Mapping[str, Any]] = (),
    ) -> RetrievalResponse:
        identity = {
            "querySha256": query_hash,
            "indexSha256": self.index.indexSha256,
            "filters": filters,
            "results": [item["resultId"] for item in results],
        }
        response = RetrievalResponse.model_validate(
            {
                "schemaVersion": 1,
                "contractVersion": "1.0.0",
                "policyId": RETRIEVAL_POLICY_ID,
                "policySha256": self.policy["policySha256"],
                "responseId": f"vf-retrieval-response-v1-{sha256_json(identity)[:24]}",
                "status": status,
                "reasonCode": reason_code,
                "querySha256": query_hash,
                "indexId": RETRIEVAL_INDEX_ID,
                "indexVersion": self.index.indexVersion,
                "indexSha256": self.index.indexSha256,
                "sourceId": self.index.sourceId,
                "sourceRevision": self.index.sourceRevision,
                "results": list(results),
                "candidateCount": candidate_count,
                "returnedCount": len(results),
                "filtersApplied": dict(filters),
                "degraded": False,
                "embeddingsUsed": False,
                "networkAccessed": False,
                "rawQueryStored": False,
                "rawProjectContextStored": False,
            }
        )
        return validate_retrieval_response(response)


def unavailable_response(query: RetrievalQuery, error: Exception) -> RetrievalResponse:
    policy = load_retrieval_policy()
    query_hash = hashlib.sha256(query.text.encode("utf-8")).hexdigest()
    code = getattr(error, "code", "RETRIEVAL_INDEX_UNAVAILABLE")
    identity = {"querySha256": query_hash, "code": code, "policy": policy["policySha256"]}
    return validate_retrieval_response(
        RetrievalResponse.model_validate(
            {
                "schemaVersion": 1,
                "contractVersion": "1.0.0",
                "policyId": RETRIEVAL_POLICY_ID,
                "policySha256": policy["policySha256"],
                "responseId": f"vf-retrieval-response-v1-{sha256_json(identity)[:24]}",
                "status": "unavailable",
                "reasonCode": str(code)[:100],
                "querySha256": query_hash,
                "indexId": RETRIEVAL_INDEX_ID,
            "indexVersion": RETRIEVAL_INDEX_VERSION,
                "indexSha256": None,
                "sourceId": policy["source"]["sourceId"],
                "sourceRevision": policy["source"]["sourceRevision"],
                "results": [],
                "candidateCount": 0,
                "returnedCount": 0,
                "filtersApplied": {
                    "boardFilters": sorted(query.boardFilters, key=str.casefold),
                    "componentFilters": sorted(query.componentFilters, key=str.casefold),
                    "recordTypes": sorted(query.recordTypes),
                },
                "degraded": True,
                "embeddingsUsed": False,
                "networkAccessed": False,
                "rawQueryStored": False,
                "rawProjectContextStored": False,
            }
        )
    )


@lru_cache(maxsize=1)
def get_local_retrieval_service() -> LocalRetrievalService:
    return LocalRetrievalService()


def search_curated_local(query: RetrievalQuery | Mapping[str, Any]) -> RetrievalResponse:
    request = query if isinstance(query, RetrievalQuery) else RetrievalQuery.model_validate(query)
    try:
        return get_local_retrieval_service().search(request)
    except RetrievalContractError as error:
        return unavailable_response(request, error)


def search_for_request(request: Any) -> RetrievalResponse:
    components = []
    for item in list(getattr(request, "components", []) or [])[:20]:
        if not isinstance(item, Mapping):
            continue
        value = item.get("type") or item.get("name")
        if value and str(value).casefold() not in {existing.casefold() for existing in components}:
            components.append(str(value)[:200])
    board = str(getattr(request, "boardType", "") or "").strip()
    return search_curated_local(
        RetrievalQuery(
            text=str(getattr(request, "message", ""))[:2_000],
            maximumResults=5,
            boardFilters=[board] if board else [],
            componentFilters=components,
        )
    )


def retrieval_tool_event(response: RetrievalResponse) -> dict[str, Any]:
    compact_results = []
    for result in response.results[:2]:
        compact = {
            "recordId": result.recordId,
            "recordType": result.recordType,
            "subject": result.subject.name,
            "sourceId": result.source.sourceId,
            "sourceRevision": result.source.sourceRevision,
            "recordRevision": result.source.recordRevision,
            "contentSha256": result.contentSha256,
            "citationId": result.citation.citationId,
        }
        for fact in result.facts:
            bounded_fact = {
                key: value
                for key, value in fact.model_dump(mode="json").items()
                if key in {"claimId", "property", "pointer", "value", "status", "evidenceRefs"}
            }
            rendered = json.dumps(
                bounded_fact,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            if len(rendered) <= 600:
                compact["fact"] = rendered
                break
        compact_results.append(compact)
    return {
        "name": "curated-local-retrieval",
        "version": response.contractVersion,
        "status": "complete"
        if response.status in {"complete", "no-results"}
        else "unavailable",
        "authority": "retrieved",
        "summary": (
            f"Retrieved {response.returnedCount} checksum-bound local evidence chunk(s)."
            if response.status == "complete"
            else "The curated local index had no matching evidence."
            if response.status == "no-results"
            else "The curated local index is unavailable; no stale evidence was returned."
        ),
        "evidence": {
            "policyId": response.policyId,
            "policySha256": response.policySha256,
            "indexId": response.indexId,
            "indexVersion": response.indexVersion,
            "indexSha256": response.indexSha256,
            "sourceId": response.sourceId,
            "sourceRevision": response.sourceRevision,
            "retrievalStatus": response.status,
            "reasonCode": response.reasonCode,
            "results": compact_results,
            "candidateCount": response.candidateCount,
            "returnedCount": response.returnedCount,
            "embeddingsUsed": False,
            "networkAccessed": False,
            "rawQueryStored": False,
            "rawProjectContextStored": False,
        },
    }


def response_metadata(response: RetrievalResponse) -> dict[str, Any]:
    return {
        "policyId": response.policyId,
        "indexId": response.indexId,
        "indexVersion": response.indexVersion,
        "indexSha256": response.indexSha256,
        "sourceId": response.sourceId,
        "sourceRevision": response.sourceRevision,
        "status": response.status,
        "reasonCode": response.reasonCode,
        "returnedCount": response.returnedCount,
        "candidateCount": response.candidateCount,
        "degraded": response.degraded,
        "embeddingsUsed": False,
        "networkAccessed": False,
        "rawQueryStored": False,
    }


def citations_from_response(response: RetrievalResponse) -> list[dict[str, str]]:
    return [
        {
            key: str(value)
            for key, value in result.citation.model_dump(mode="json").items()
        }
        for result in response.results
    ]


def render_retrieval_summary(response: RetrievalResponse, maximum_results: int = 3) -> str:
    if response.status != "complete":
        return ""
    lines = ["Local curated evidence:"]
    for result in response.results[:maximum_results]:
        facts = []
        for fact in result.facts[:3]:
            value = json.dumps(fact.value, ensure_ascii=False, sort_keys=True)
            label = fact.property + (fact.pointer or "")
            facts.append(f"{label}={value}")
        lines.append(
            f"- [{result.citation.citationId}] {result.subject.name} "
            f"({result.subject.variant}): {'; '.join(facts)}"
        )
    return "\n".join(lines)


def local_retrieval_health() -> dict[str, Any]:
    try:
        policy = load_retrieval_policy()
    except RetrievalContractError as error:
        return {
            "ready": False,
            "code": error.code,
            "policyId": RETRIEVAL_POLICY_ID,
            "indexId": RETRIEVAL_INDEX_ID,
            "embeddingsEnabled": False,
            "networkRequired": False,
            "staleResultsAllowed": False,
            "rawQueryStored": False,
        }
    try:
        service = get_local_retrieval_service()
        return {
            "ready": True,
            "code": "RETRIEVAL_READY",
            "policyId": policy["policyId"],
            "policySha256": policy["policySha256"],
            "indexId": service.index.indexId,
            "indexVersion": service.index.indexVersion,
            "indexSha256": service.index.indexSha256,
            "sourceId": service.index.sourceId,
            "sourceRevision": service.index.sourceRevision,
            "recordCount": service.index.recordCount,
            "chunkCount": service.index.chunkCount,
            "algorithm": service.index.algorithm,
            "embeddingsEnabled": False,
            "networkRequired": False,
            "staleResultsAllowed": False,
            "rawQueryStored": False,
        }
    except RetrievalContractError as error:
        return {
            "ready": False,
            "code": error.code,
            "policyId": policy["policyId"],
            "indexId": policy["indexId"],
            "sourceId": policy["source"]["sourceId"],
            "sourceRevision": policy["source"]["sourceRevision"],
            "embeddingsEnabled": False,
            "networkRequired": False,
            "staleResultsAllowed": False,
            "rawQueryStored": False,
        }
