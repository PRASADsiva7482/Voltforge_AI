from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket

import pytest
import requests

from api.copilot import prepare_grounded_context
from api.routes import chat as chat_route, local_retrieval_search
from api.schemas import ChatRequest
from local_retrieval import (
    LocalRetrievalService,
    RetrievalContractError,
    RetrievalQuery,
    local_retrieval_health,
    search_curated_local,
    validate_retrieval_index,
    validate_retrieval_response,
)
from local_retrieval.builder import build_index
from local_retrieval.schema import (
    RETRIEVAL_INDEX_PATH,
    build_index_json_schema,
    build_response_json_schema,
    checked_index_schema,
    checked_response_schema,
    load_retrieval_policy,
    sha256_json,
)
from tools.evaluate_local_retrieval import evaluate, verify


def _write_tampered_index(tmp_path: Path, mutate) -> Path:
    raw = json.loads(RETRIEVAL_INDEX_PATH.read_text(encoding="utf-8"))
    mutate(raw)
    unsigned = dict(raw)
    unsigned.pop("indexSha256")
    raw["indexSha256"] = sha256_json(unsigned)
    path = tmp_path / "index.json"
    path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    return path


def test_policy_checked_schemas_and_index_are_reproducible() -> None:
    policy = load_retrieval_policy()
    checked = validate_retrieval_index(
        json.loads(RETRIEVAL_INDEX_PATH.read_text(encoding="utf-8"))
    )
    rebuilt = build_index()

    assert policy["algorithm"]["embeddingsEnabled"] is False
    assert checked_index_schema() == build_index_json_schema()
    assert checked_response_schema() == build_response_json_schema()
    assert rebuilt.model_dump(mode="json") == checked.model_dump(mode="json")
    assert checked.recordCount == 61
    assert checked.chunkCount == 104
    assert {item.recordType for item in checked.chunks} == {
        "board",
        "pin-map",
        "component",
        "wiring-recipe",
        "firmware-api",
        "compiler-diagnostic",
        "simulation-behavior",
        "safety-constraint",
    }


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("RA4M1 UNO R4 WiFi architecture", "vf-knowledge-v1-board.arduino.uno-r4-wifi"),
        (
            "fatal error no such file missing header remediation",
            "vf-knowledge-v1-compiler.missing-header",
        ),
        (
            "flyback diode inductive load required action",
            "vf-knowledge-v1-safety.inductive-flyback",
        ),
        (
            "open drain I2C pull ups digital bus simulation",
            "vf-knowledge-v1-simulation.i2c-digital",
        ),
    ],
)
def test_ranked_lexical_search_returns_expected_exact_record(
    query: str, expected: str
) -> None:
    response = search_curated_local(RetrievalQuery(text=query, maximumResults=5))

    assert response.status == "complete"
    assert response.results[0].recordId == expected
    assert response.results[0].facts
    assert response.results[0].citation.contentSha256 == response.results[0].contentSha256
    assert response.networkAccessed is False
    assert response.embeddingsUsed is False


def test_board_and_component_filters_select_exact_wiring_variant() -> None:
    response = search_curated_local(
        RetrievalQuery(
            text="SSD1306 I2C connections preconditions",
            boardFilters=["ARDUINO_UNO"],
            componentFilters=["SSD1306_STEMMA_QT"],
        )
    )

    assert response.results[0].recordId == (
        "vf-knowledge-v1-wiring.arduino.uno-r3.ssd1306-stemma-i2c"
    )
    assert response.results[0].source.sourceRevision == "1.1.0"
    assert response.results[0].source.recordRevision


def test_unknown_query_returns_no_evidence() -> None:
    response = search_curated_local(RetrievalQuery(text="xyzzynonexistenttokenvalue"))

    assert response.status == "no-results"
    assert response.reasonCode == "RETRIEVAL_NO_MATCH"
    assert response.results == []
    assert response.returnedCount == 0
    validate_retrieval_response(response)


def test_source_catalog_mismatch_rejects_all_stale_results(tmp_path: Path) -> None:
    path = _write_tampered_index(
        tmp_path, lambda raw: raw.__setitem__("sourceCatalogSha256", "0" * 64)
    )

    with pytest.raises(RetrievalContractError) as captured:
        LocalRetrievalService(path)

    assert captured.value.code == "RETRIEVAL_INDEX_SOURCE_MISMATCH"


def test_index_version_mismatch_is_explicit(tmp_path: Path) -> None:
    path = _write_tampered_index(
        tmp_path, lambda raw: raw.__setitem__("indexVersion", "9.0.0")
    )

    with pytest.raises(RetrievalContractError) as captured:
        LocalRetrievalService(path)

    assert captured.value.code == "RETRIEVAL_INDEX_SCHEMA_INVALID"


def test_chunk_tampering_fails_checksum_before_search(tmp_path: Path) -> None:
    raw = json.loads(RETRIEVAL_INDEX_PATH.read_text(encoding="utf-8"))
    tampered = deepcopy(raw)
    tampered["chunks"][0]["facts"][0]["value"] = "invented-value"
    path = tmp_path / "index.json"
    path.write_text(json.dumps(tampered, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(RetrievalContractError) as captured:
        LocalRetrievalService(path)

    assert captured.value.code == "RETRIEVAL_INDEX_CHECKSUM_MISMATCH"


def test_build_and_search_have_no_network_dependency(monkeypatch) -> None:
    def denied(*_args, **_kwargs):
        raise AssertionError("local retrieval attempted network access")

    monkeypatch.setattr(requests.sessions.Session, "request", denied)
    monkeypatch.setattr(socket, "create_connection", denied)

    service = LocalRetrievalService()
    response = service.search(RetrievalQuery(text="Pico 2 RP2350 architecture"))

    assert response.status == "complete"
    assert build_index().embeddingsPresent is False


def test_chat_context_selects_compact_retrieval_and_preserves_authority() -> None:
    request = ChatRequest(
        message="How do I wire the SSD1306 over I2C?",
        projectId="project-retrieval-22",
        projectRevision="revision-retrieval-22",
        boardType="ARDUINO_UNO",
        components=[{"id": "oled", "type": "SSD1306_STEMMA_QT"}],
    )
    grounded = prepare_grounded_context(request)
    names = [item["name"] for item in grounded.tool_events]

    assert names[0] == "engineering-authority-index"
    assert "curated-local-retrieval" in names
    assert len(names) <= 8
    assert grounded.response_metadata["localRetrieval"]["selectedForModelContext"] is True
    typed = next(
        item
        for item in grounded.task_record["input"]["toolEvidence"]
        if item["toolName"] == "tool:curated-local-retrieval"
    )
    assert typed["authority"] == "retrieved"
    assert typed["payload"]["networkAccessed"] is False
    assert len(typed["payload"]["results"]) <= 2


def test_route_health_and_privacy_metadata_are_typed() -> None:
    response = local_retrieval_search(
        RetrievalQuery(text="Arduino Wire I2C API", maximumResults=3)
    )
    health = local_retrieval_health()

    assert response["status"] == "complete"
    assert response["returnedCount"] <= 3
    assert response["rawQueryStored"] is False
    assert health["ready"] is True
    assert health["algorithm"] == "bm25-lexical-v1"
    assert health["staleResultsAllowed"] is False
    assert health["embeddingsEnabled"] is False


def test_synchronous_chat_exposes_and_renders_local_evidence() -> None:
    response = chat_route(
        ChatRequest(
            message="What MCU architecture and logic voltage does Arduino UNO R3 use?",
            boardType="ARDUINO_UNO",
            projectRevision="retrieval-chat-revision",
        )
    )

    assert response.localRetrieval is not None
    assert response.localRetrieval["status"] == "complete"
    assert response.citations
    assert "Local curated evidence:" in response.reply


def test_private_query_is_represented_only_by_hash() -> None:
    sentinel = "PRIVATE_RETRIEVAL_TEST_VALUE_22"
    response = search_curated_local(RetrievalQuery(text=sentinel))
    rendered = json.dumps(response.model_dump(mode="json"), sort_keys=True)

    assert sentinel not in rendered
    assert response.querySha256 == hashlib.sha256(sentinel.encode()).hexdigest()
    assert response.rawQueryStored is False


def test_evaluation_receipt_is_reproducible(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"

    generated = evaluate(path)
    checked = verify(path)

    assert generated == checked
    assert generated["heldOutRecallAt5"] == "1.000000"
    assert generated["citationEntailment"] == "1.000000"
    assert all(generated["checks"].values())
