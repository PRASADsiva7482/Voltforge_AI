"""Generate or verify the content-free VFAI-022 local-retrieval receipt."""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import socket
import sys
import tempfile
from typing import Any, Mapping, Sequence

import requests


AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from api.copilot import prepare_grounded_context
from api.routes import local_retrieval_search
from api.schemas import ChatRequest
from data_governance import require_approved_source
from electronics_corpus import get_electronics_corpus
from local_retrieval import (
    LocalRetrievalService,
    RetrievalContractError,
    RetrievalQuery,
    local_retrieval_health,
    search_curated_local,
    validate_retrieval_response,
)
from local_retrieval.builder import BUILDER_PATH, build_index
from local_retrieval.service import unavailable_response
from local_retrieval.schema import (
    RETRIEVAL_INDEX_PATH,
    RETRIEVAL_INDEX_SCHEMA_PATH,
    RETRIEVAL_RESPONSE_SCHEMA_PATH,
    build_index_json_schema,
    build_response_json_schema,
    canonical_json,
    checked_index_schema,
    checked_response_schema,
    load_retrieval_policy,
    sha256_json,
    validate_retrieval_index,
)


FIXTURE_PATH = AI_ROOT / "local_retrieval" / "evaluation" / "heldout-v1.json"
DEFAULT_REPORT_PATH = AI_ROOT / "evaluation" / "reports" / "curated-local-retrieval-v1.json"
PRIVATE_SENTINEL = "PRIVATE_LOCAL_RETRIEVAL_QUERY_22"


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_fixture() -> dict[str, Any]:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    if (
        not isinstance(fixture, dict)
        or fixture.get("suiteId") != "vfai022-local-retrieval-heldout-v1"
        or not isinstance(fixture.get("cases"), list)
        or len(fixture["cases"]) < 20
    ):
        raise RuntimeError("VFAI-022 held-out fixture contract is invalid")
    ids = [item.get("id") for item in fixture["cases"]]
    if len(ids) != len(set(ids)) or any(not item for item in ids):
        raise RuntimeError("VFAI-022 held-out fixture IDs are invalid")
    return fixture


def _resolve_pointer(value: Any, pointer: str) -> Any:
    current = value
    if not pointer:
        return current
    for raw in pointer.removeprefix("/").split("/"):
        part = raw.replace("~1", "/").replace("~0", "~")
        current = current[int(part)] if isinstance(current, list) else current[part]
    return current


def _result_is_entailed(result: Any, records_by_id: Mapping[str, Mapping[str, Any]]) -> bool:
    record = records_by_id.get(result.recordId)
    if record is None:
        return False
    revision = record["effectiveRevision"]
    if (
        result.source.sourceId != record["provenance"]["sourceId"]
        or result.source.sourceRevision != record["provenance"]["sourceRevision"]
        or result.source.recordRevision != revision["revision"]
        or result.source.validFrom != revision["validFrom"]
        or result.citation.recordId != result.recordId
        or result.citation.contentSha256 != result.contentSha256
    ):
        return False
    claims = {item["claimId"]: item for item in record["claims"]}
    evidence = {item["evidenceId"]: item for item in record["provenance"]["evidence"]}
    for fact in result.facts:
        claim = claims.get(fact.claimId)
        if claim is None:
            return False
        if (
            fact.property != claim["property"]
            or fact.status != claim["status"]
            or fact.unit != claim.get("unit")
            or fact.conditions != claim.get("conditions", [])
            or fact.evidenceRefs != sorted(set(claim["evidenceRefs"]))
            or fact.value != _resolve_pointer(claim.get("value"), fact.pointer)
        ):
            return False
    for item in result.evidence:
        source = evidence.get(item.evidenceId)
        if source is None:
            return False
        for key, value in item.model_dump(mode="json").items():
            if source.get(key) != value:
                return False
    return True


def _tampered_index_degrades(query: RetrievalQuery) -> tuple[bool, str]:
    raw = json.loads(RETRIEVAL_INDEX_PATH.read_text(encoding="utf-8"))
    raw["sourceCatalogSha256"] = "0" * 64
    unsigned = dict(raw)
    unsigned.pop("indexSha256")
    raw["indexSha256"] = sha256_json(unsigned)
    with tempfile.TemporaryDirectory(prefix="vfai022-") as directory:
        path = Path(directory) / "index.json"
        path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
        try:
            LocalRetrievalService(path)
        except RetrievalContractError as error:
            response = unavailable_response(query, error)
            return (
                error.code == "RETRIEVAL_INDEX_SOURCE_MISMATCH"
                and response.status == "unavailable"
                and response.degraded
                and response.returnedCount == 0,
                error.code,
            )
    return False, "NO_ERROR"


def build_report() -> dict[str, Any]:
    fixture = _load_fixture()
    policy = load_retrieval_policy()
    index = validate_retrieval_index(
        json.loads(RETRIEVAL_INDEX_PATH.read_text(encoding="utf-8"))
    )
    corpus = get_electronics_corpus()
    records_by_id = {item["recordId"]: item for item in corpus.records}
    original_request = requests.sessions.Session.request
    original_connection = socket.create_connection

    def denied(*_args, **_kwargs):
        raise AssertionError("VFAI-022 attempted network access")

    requests.sessions.Session.request = denied
    socket.create_connection = denied
    network_denied = True
    responses = []
    positions = []
    try:
        for case in fixture["cases"]:
            query = RetrievalQuery(
                text=case["query"],
                maximumResults=int(fixture["maximumRank"]),
                boardFilters=case.get("boardFilters", []),
                componentFilters=case.get("componentFilters", []),
                recordTypes=case.get("recordTypes", []),
            )
            response = search_curated_local(query)
            validate_retrieval_response(response)
            responses.append(response)
            positions.append(
                next(
                    (
                        result.rank
                        for result in response.results
                        if result.recordId == case["expectedRecordId"]
                    ),
                    0,
                )
            )
        private = search_curated_local(RetrievalQuery(text=PRIVATE_SENTINEL))
    except AssertionError:
        network_denied = False
        raise
    finally:
        requests.sessions.Session.request = original_request
        socket.create_connection = original_connection

    recall = Decimal(sum(position > 0 for position in positions)) / Decimal(len(positions))
    mrr = sum(
        (Decimal(1) / Decimal(position) if position else Decimal(0))
        for position in positions
    ) / Decimal(len(positions))
    all_results = [result for response in responses for result in response.results]
    entailed = sum(_result_is_entailed(result, records_by_id) for result in all_results)
    entailment = Decimal(entailed) / Decimal(len(all_results))
    unsupported_rate = Decimal(len(all_results) - entailed) / Decimal(len(all_results))

    deterministic_rebuild = build_index().model_dump(mode="json") == index.model_dump(
        mode="json"
    )
    reordered_cases = list(reversed(fixture["cases"]))
    reordered_positions = {}
    for case in reordered_cases:
        response = search_curated_local(
            RetrievalQuery(
                text=case["query"],
                maximumResults=int(fixture["maximumRank"]),
                boardFilters=case.get("boardFilters", []),
                componentFilters=case.get("componentFilters", []),
                recordTypes=case.get("recordTypes", []),
            )
        )
        reordered_positions[case["id"]] = [item.resultId for item in response.results]
    original_positions = {
        case["id"]: [item.resultId for item in response.results]
        for case, response in zip(fixture["cases"], responses, strict=True)
    }

    stale_ok, stale_code = _tampered_index_degrades(
        RetrievalQuery(text="UNO R3 logic voltage")
    )
    no_match = search_curated_local(
        RetrievalQuery(text="xyzzynonexistenttokenvalue", maximumResults=5)
    )
    chat_request = ChatRequest(
        message="How do I wire the SSD1306 over I2C?",
        projectId="vfai022-evaluation-project",
        projectRevision="vfai022-evaluation-revision",
        boardType="ARDUINO_UNO",
        components=[{"id": "oled", "type": "SSD1306_STEMMA_QT"}],
    )
    grounded = prepare_grounded_context(chat_request)
    route_response = local_retrieval_search(
        RetrievalQuery(text="RP2350 Pico 2 architecture", maximumResults=3)
    )
    public_values = {
        "privateResponse": private.model_dump(mode="json"),
        "health": local_retrieval_health(),
        "route": route_response,
    }

    recall_threshold = Decimal(policy["quality"]["heldOutRecallAt5Minimum"])
    mrr_threshold = Decimal(policy["quality"]["heldOutMeanReciprocalRankMinimum"])
    entailment_threshold = Decimal(policy["quality"]["citationEntailmentMinimum"])
    unsupported_threshold = Decimal(
        policy["quality"]["unsupportedCitationRateMaximum"]
    )
    checks = {
        "policyChecksumVerified": policy["policySha256"]
        == _policy_digest(policy),
        "checkedIndexSchemaMatchesExecutableContract": checked_index_schema()
        == build_index_json_schema(),
        "checkedResponseSchemaMatchesExecutableContract": checked_response_schema()
        == build_response_json_schema(),
        "sourceApprovedForRuntimeRetrieval": require_approved_source(
            policy["source"]["sourceId"], policy["source"]["requiredUse"]
        )["revision"]
        == policy["source"]["sourceRevision"],
        "indexRebuildIsByteStable": deterministic_rebuild,
        "networkDeniedDuringBuildAndSearch": network_denied,
        "lexicalOnlyNoEmbeddings": index.embeddingsPresent is False
        and policy["algorithm"]["embeddingsEnabled"] is False,
        "allEightRecordTypesIndexed": {item.recordType for item in index.chunks}
        == {
            "board",
            "pin-map",
            "component",
            "wiring-recipe",
            "firmware-api",
            "compiler-diagnostic",
            "simulation-behavior",
            "safety-constraint",
        },
        "heldOutRecallAt5MeetsThreshold": recall >= recall_threshold,
        "heldOutMeanReciprocalRankMeetsThreshold": mrr >= mrr_threshold,
        "citationEntailmentMeetsThreshold": entailment >= entailment_threshold,
        "unsupportedCitationRateWithinThreshold": unsupported_rate
        <= unsupported_threshold,
        "queryOrderDoesNotChangeResults": reordered_positions == original_positions,
        "boardAndComponentFiltersRetrieveExactWiring": all(
            position == 1
            for case, position in zip(fixture["cases"], positions, strict=True)
            if case["id"].startswith("ret-wiring")
        ),
        "resultsAreStrictlyBounded": all(
            response.returnedCount <= int(policy["limits"]["maximumResults"])
            and all(
                len(result.facts) <= int(policy["limits"]["maximumFactsPerChunk"])
                for result in response.results
            )
            for response in responses
        ),
        "staleIndexFailsClosed": stale_ok
        and stale_code == "RETRIEVAL_INDEX_SOURCE_MISMATCH",
        "noMatchReturnsNoEvidence": no_match.status == "no-results"
        and no_match.returnedCount == 0,
        "chatContextSelectsCuratedRetrieval": any(
            event.get("name") == "curated-local-retrieval"
            for event in grounded.tool_events
        )
        and grounded.response_metadata["localRetrieval"]["selectedForModelContext"]
        is True,
        "engineeringAuthorityRemainsPresent": any(
            event.get("name") == "engineering-authority-index"
            for event in grounded.tool_events
        ),
        "apiRouteReturnsTypedLocalEvidence": route_response["status"] == "complete"
        and route_response["sourceId"] == policy["source"]["sourceId"],
        "healthPinsIndexAndRejectsStaleResults": public_values["health"]["ready"]
        is True
        and public_values["health"]["staleResultsAllowed"] is False,
        "rawPrivateQueryNotStored": PRIVATE_SENTINEL
        not in json.dumps(public_values, sort_keys=True),
    }
    report = {
        "schemaVersion": 1,
        "reportId": "vfai022-curated-local-retrieval-v1",
        "generatedOn": "2026-08-30",
        "policyId": policy["policyId"],
        "policySha256": policy["policySha256"],
        "indexId": index.indexId,
        "indexVersion": index.indexVersion,
        "indexSha256": index.indexSha256,
        "indexBuilderSha256": _sha_file(BUILDER_PATH),
        "indexSchemaSha256": _sha_file(RETRIEVAL_INDEX_SCHEMA_PATH),
        "responseSchemaSha256": _sha_file(RETRIEVAL_RESPONSE_SCHEMA_PATH),
        "fixtureId": fixture["suiteId"],
        "fixtureSha256": _sha_file(FIXTURE_PATH),
        "evaluatorSha256": _sha_file(Path(__file__).resolve()),
        "recordCount": index.recordCount,
        "chunkCount": index.chunkCount,
        "termCount": len(index.documentFrequencies),
        "heldOutCaseCount": len(fixture["cases"]),
        "heldOutRecallAt5": format(recall.quantize(Decimal("0.000001")), "f"),
        "heldOutMeanReciprocalRank": format(mrr.quantize(Decimal("0.000001")), "f"),
        "citationEntailment": format(entailment.quantize(Decimal("0.000001")), "f"),
        "unsupportedCitationRate": format(
            unsupported_rate.quantize(Decimal("0.000001")), "f"
        ),
        "staleIndexReasonCode": stale_code,
        "checks": checks,
        "embeddingsApproved": False,
        "networkRequired": False,
        "rawQueriesStored": False,
        "reportSha256": "",
    }
    report["reportSha256"] = _receipt_digest(report)
    return report


def _policy_digest(policy: Mapping[str, Any]) -> str:
    unsigned = dict(policy)
    unsigned.pop("policySha256", None)
    return sha256_json(unsigned)


def _receipt_digest(report: Mapping[str, Any]) -> str:
    unsigned = dict(report)
    unsigned.pop("reportSha256", None)
    return hashlib.sha256(canonical_json(unsigned)).hexdigest()


def _write_report(report: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def evaluate(path: Path) -> dict[str, Any]:
    report = build_report()
    if not all(report["checks"].values()):
        failed = [key for key, value in report["checks"].items() if not value]
        raise RuntimeError(f"VFAI-022 evaluation failed: {failed}")
    _write_report(report, path)
    return report


def verify(path: Path) -> dict[str, Any]:
    checked = json.loads(path.read_text(encoding="utf-8"))
    generated = build_report()
    if checked != generated or checked.get("reportSha256") != _receipt_digest(checked):
        raise RuntimeError("VFAI-022 evaluation report is stale or invalid")
    if not all(checked["checks"].values()):
        raise RuntimeError("VFAI-022 evaluation report contains a failed check")
    return checked


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("evaluate", "verify"))
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    report = (
        evaluate(arguments.output.resolve())
        if arguments.command == "evaluate"
        else verify(arguments.output.resolve())
    )
    print(
        json.dumps(
            {
                "ok": True,
                "command": arguments.command,
                "reportId": report["reportId"],
                "reportSha256": report["reportSha256"],
                "heldOutRecallAt5": report["heldOutRecallAt5"],
                "heldOutMeanReciprocalRank": report[
                    "heldOutMeanReciprocalRank"
                ],
                "citationEntailment": report["citationEntailment"],
                "checks": report["checks"],
                "embeddingsApproved": report["embeddingsApproved"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
