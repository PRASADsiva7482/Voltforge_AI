import json
from pathlib import Path
import socket

import pytest

from benchmarking.runtime_baseline import (
    BenchmarkSettings,
    build_report,
    derive_cpu_budgets,
    offline_runtime,
    percentile,
    process_memory_bytes,
    validate_report,
    validate_schema_contract,
    write_report,
)


def test_percentile_interpolates_small_samples() -> None:
    assert percentile([10.0, 20.0], 0.5) == 15.0
    assert percentile([30.0, 10.0, 20.0], 0.95) == pytest.approx(29.0)


def test_process_memory_probe_returns_current_and_peak_working_set() -> None:
    current, peak = process_memory_bytes()
    assert current is not None and current > 0
    assert peak is not None and peak >= current


def test_offline_runtime_disables_features_and_blocks_sockets(monkeypatch) -> None:
    monkeypatch.setenv("VOLTFORGE_AI_INTERNET_RETRIEVAL_ENABLED", "true")
    original_socket = socket.socket
    with offline_runtime():
        assert socket.socket is not original_socket
        assert __import__("os").environ["VOLTFORGE_AI_INTERNET_RETRIEVAL_ENABLED"] == "false"
        blocked = socket.socket()
        try:
            with pytest.raises(OSError, match="blocks outbound network"):
                blocked.connect(("192.0.2.1", 9))
        finally:
            blocked.close()
    assert socket.socket is original_socket


def test_budgets_are_derived_from_measurements() -> None:
    cold_start = {
        "processLaunchToReady": {"max": 100.0},
        "peakWorkingSet": {"max": 200.0},
    }
    generation = {
        "firstDeltaLatency": {"p95": 20.0},
        "generationThroughput": {"median": 1_000.0},
        "memory": {"peakWorkingSetMiB": 220.0},
        "concurrency": [
            {
                "concurrency": 4,
                "throughputRequestsPerSecond": 40.0,
                "requestLatency": {"p95": 50.0},
                "memory": {"peakWorkingSetMiB": 230.0},
            }
        ],
    }

    budgets = derive_cpu_budgets("benchmark-1", cold_start, generation)

    assert budgets["sourceBenchmarkId"] == "benchmark-1"
    assert budgets["limits"] == {
        "maximumColdStartMs": 130,
        "maximumFirstDeltaMs": 32,
        "minimumCharactersPerSecond": 700,
        "maximumWorkingSetMiB": 292,
        "concurrency": 4,
        "maximumConcurrentP95Ms": 77,
        "minimumConcurrentRequestsPerSecond": 28.0,
        "maximumFailedRequests": 0,
    }


def test_quick_report_is_offline_truthful_and_json_serializable(tmp_path: Path) -> None:
    report = build_report(
        BenchmarkSettings(
            cold_start_runs=1,
            warmup_requests=0,
            measured_requests=1,
            concurrency_levels=(1,),
        ),
        probe_acceleration=False,
    )
    validate_report(report)
    validate_schema_contract(report)

    assert report["networkAccessAllowed"] is False
    assert report["hardware"]["referenceTier"] == "cpu-development-reference"
    assert report["software"]["python"]["version"]
    assert report["benchmarks"]["localDeterministicGeneration"]["completedRequests"] == 1
    assert report["benchmarks"]["localDeterministicGeneration"]["tokenMetricsApplicable"] is False
    assert report["model"]["neuralGeneration"]["firstTokenLatencyMs"] is None
    assert report["model"]["neuralGeneration"]["tokensPerSecond"] is None

    output = tmp_path / "runtime.json"
    write_report(report, output)
    restored = json.loads(output.read_text(encoding="utf-8"))
    assert restored["derivedBudgets"]["sourceBenchmarkId"] == restored["benchmarkId"]


def test_runtime_schema_declares_offline_and_truthful_metric_contracts() -> None:
    root = Path(__file__).resolve().parents[1]
    schema = json.loads(
        (root / "benchmarks" / "runtime-baseline.schema.json").read_text(encoding="utf-8")
    )
    assert schema["properties"]["networkAccessAllowed"] == {"const": False}
    generation = schema["properties"]["benchmarks"]["properties"][
        "localDeterministicGeneration"
    ]["properties"]
    assert generation["outputUnit"] == {"const": "characters"}
    assert generation["tokenMetricsApplicable"] == {"const": False}
