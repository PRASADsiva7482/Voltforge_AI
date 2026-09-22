"""Repeatable, offline hardware and runtime baselines for VoltForge AI.

The current production path is a deterministic local engine.  Neural metrics
remain explicitly unavailable until an approved artifact and inference runtime
exist; this module never substitutes character rates for neural token rates.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import ipaddress
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Iterator, Sequence


AI_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ENTRYPOINT = AI_ROOT / "tools" / "benchmark_runtime.py"
DEFAULT_OUTPUT = AI_ROOT / "benchmarks" / "reports" / "cpu-reference.json"
DEFAULT_SCHEMA = AI_ROOT / "benchmarks" / "runtime-baseline.schema.json"
SCHEMA_VERSION = 1
MEBIBYTE = 1024 * 1024

if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))


@dataclass(frozen=True)
class BenchmarkSettings:
    cold_start_runs: int = 3
    warmup_requests: int = 1
    measured_requests: int = 5
    concurrency_levels: tuple[int, ...] = (1, 2, 4)

    def validate(self) -> None:
        if not 1 <= self.cold_start_runs <= 20:
            raise ValueError("cold_start_runs must be between 1 and 20")
        if not 0 <= self.warmup_requests <= 20:
            raise ValueError("warmup_requests must be between 0 and 20")
        if not 1 <= self.measured_requests <= 100:
            raise ValueError("measured_requests must be between 1 and 100")
        if not self.concurrency_levels:
            raise ValueError("at least one concurrency level is required")
        if any(level < 1 or level > 32 for level in self.concurrency_levels):
            raise ValueError("concurrency levels must be between 1 and 32")
        if tuple(sorted(set(self.concurrency_levels))) != self.concurrency_levels:
            raise ValueError("concurrency levels must be unique and increasing")


def _round(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def percentile(values: Sequence[float], fraction: float) -> float:
    """Return a linearly interpolated percentile for small benchmark samples."""
    if not values:
        raise ValueError("cannot calculate a percentile from an empty sample")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("percentile fraction must be between 0 and 1")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: Sequence[float], unit: str) -> dict[str, Any]:
    if not values:
        return {"unit": unit, "samples": 0, "min": None, "median": None, "p95": None, "max": None}
    return {
        "unit": unit,
        "samples": len(values),
        "min": _round(min(values)),
        "median": _round(statistics.median(values)),
        "p95": _round(percentile(values, 0.95)),
        "max": _round(max(values)),
    }


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _total_memory_bytes() -> int | None:
    if sys.platform == "win32":
        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.total_physical)
        return None
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        return int(page_size * page_count)
    except (AttributeError, OSError, ValueError):
        return None


def _windows_process_memory() -> tuple[int | None, int | None]:
    class ProcessMemoryCountersEx(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("page_fault_count", wintypes.DWORD),
            ("peak_working_set_size", ctypes.c_size_t),
            ("working_set_size", ctypes.c_size_t),
            ("quota_peak_paged_pool_usage", ctypes.c_size_t),
            ("quota_paged_pool_usage", ctypes.c_size_t),
            ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
            ("quota_non_paged_pool_usage", ctypes.c_size_t),
            ("pagefile_usage", ctypes.c_size_t),
            ("peak_pagefile_usage", ctypes.c_size_t),
            ("private_usage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCountersEx),
        wintypes.DWORD,
    )
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = ProcessMemoryCountersEx()
    counters.cb = ctypes.sizeof(counters)
    process = kernel32.GetCurrentProcess()
    success = psapi.GetProcessMemoryInfo(
        process, ctypes.byref(counters), counters.cb
    )
    if not success:
        return None, None
    return int(counters.working_set_size), int(counters.peak_working_set_size)


def process_memory_bytes() -> tuple[int | None, int | None]:
    """Return current RSS and OS-observed peak RSS where supported."""
    if sys.platform == "win32":
        return _windows_process_memory()
    current: int | None = None
    if sys.platform.startswith("linux"):
        try:
            pages = int(Path("/proc/self/statm").read_text(encoding="ascii").split()[1])
            current = pages * int(os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError, IndexError):
            current = None
    try:
        import resource

        reported = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        peak = reported if sys.platform == "darwin" else reported * 1024
    except (ImportError, OSError, ValueError):
        peak = None
    return current, peak


class PeakMemorySampler:
    """Poll process RSS while a short benchmark section executes."""

    def __init__(self, interval_seconds: float = 0.005):
        self.interval_seconds = interval_seconds
        self.start_bytes: int | None = None
        self.peak_bytes: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "PeakMemorySampler":
        self.start_bytes = process_memory_bytes()[0]
        self.peak_bytes = self.start_bytes
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            current = process_memory_bytes()[0]
            if current is not None and (self.peak_bytes is None or current > self.peak_bytes):
                self.peak_bytes = current

    def __exit__(self, *_args: object) -> None:
        current, observed_peak = process_memory_bytes()
        for candidate in (current, observed_peak):
            if candidate is not None and (self.peak_bytes is None or candidate > self.peak_bytes):
                self.peak_bytes = candidate
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def report(self) -> dict[str, float | None]:
        start = self.start_bytes
        peak = self.peak_bytes
        delta = None if start is None or peak is None else max(0, peak - start)
        return {
            "startWorkingSetMiB": None if start is None else _round(start / MEBIBYTE),
            "peakWorkingSetMiB": None if peak is None else _round(peak / MEBIBYTE),
            "peakIncreaseMiB": None if delta is None else _round(delta / MEBIBYTE),
        }


@contextmanager
def offline_runtime() -> Iterator[None]:
    """Disable retrieval and reject non-loopback socket connections."""
    environment_names = (
        "VOLTFORGE_AI_INTERNET_RETRIEVAL_ENABLED",
        "VOLTFORGE_AI_STORE_CONVERSATIONS",
        "VOLTFORGE_AI_DB_ENABLED",
    )
    previous_environment = {name: os.environ.get(name) for name in environment_names}
    original_socket = socket.socket
    original_create_connection = socket.create_connection

    def is_loopback(address: object) -> bool:
        if not isinstance(address, tuple) or not address:
            return False
        host = address[0]
        if not isinstance(host, str):
            return False
        if host.lower() == "localhost":
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False

    class OfflineSocket(original_socket):
        def connect(self, address: object) -> None:
            if is_loopback(address):
                return super().connect(address)
            raise OSError("VoltForge runtime benchmark blocks outbound network access")

        def connect_ex(self, address: object) -> int:
            if is_loopback(address):
                return super().connect_ex(address)
            raise OSError("VoltForge runtime benchmark blocks outbound network access")

    def blocked_connection(address: object, *args: object, **kwargs: object):
        if is_loopback(address):
            return original_create_connection(address, *args, **kwargs)
        raise OSError("VoltForge runtime benchmark blocks outbound network access")

    for name in environment_names:
        os.environ[name] = "false"
    socket.socket = OfflineSocket
    socket.create_connection = blocked_connection
    try:
        yield
    finally:
        socket.socket = original_socket
        socket.create_connection = original_create_connection
        for name, value in previous_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def hardware_inventory(accelerator: dict[str, Any]) -> dict[str, Any]:
    total_memory = _total_memory_bytes()
    storage = shutil.disk_usage(AI_ROOT)
    cpu_name = platform.processor().strip() or os.environ.get("PROCESSOR_IDENTIFIER", "").strip()
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as processor_key:
                registry_name = str(
                    winreg.QueryValueEx(processor_key, "ProcessorNameString")[0]
                ).strip()
                if registry_name:
                    cpu_name = registry_name
        except (OSError, ValueError):
            pass
    return {
        "referenceTier": "cpu-development-reference",
        "operatingSystem": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "architecture": platform.machine(),
        },
        "cpu": {
            "model": cpu_name or "unknown",
            "logicalCores": os.cpu_count(),
            "physicalCores": None,
            "physicalCoreDetection": "not-available-with-stdlib",
        },
        "memory": {
            "totalBytes": total_memory,
            "totalGiB": None if total_memory is None else _round(total_memory / (1024**3)),
        },
        "storage": {
            "benchmarkPath": str(AI_ROOT),
            "totalBytes": storage.total,
            "freeBytes": storage.free,
            "totalGiB": _round(storage.total / (1024**3)),
            "freeGiB": _round(storage.free / (1024**3)),
        },
        "accelerator": accelerator,
    }


def software_inventory() -> dict[str, Any]:
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executableArchitecture": platform.architecture()[0],
        },
        "packages": {
            "numpy": _package_version("numpy"),
            "torch": _package_version("torch"),
            "fastapi": _package_version("fastapi"),
            "pydantic": _package_version("pydantic"),
            "uvicorn": _package_version("uvicorn"),
        },
    }


def _offline_subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["VOLTFORGE_AI_INTERNET_RETRIEVAL_ENABLED"] = "false"
    environment["VOLTFORGE_AI_STORE_CONVERSATIONS"] = "false"
    environment["VOLTFORGE_AI_DB_ENABLED"] = "false"
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def _run_worker(worker: str, timeout_seconds: int = 60) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, str(BENCHMARK_ENTRYPOINT), "--worker", worker],
        cwd=AI_ROOT,
        env=_offline_subprocess_environment(),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    process_wall_ms = (time.perf_counter() - started) * 1_000
    if completed.returncode != 0:
        error = completed.stderr.strip() or completed.stdout.strip() or "worker failed"
        raise RuntimeError(f"benchmark {worker} worker failed: {error}")
    try:
        return json.loads(completed.stdout), process_wall_ms
    except json.JSONDecodeError as error:
        raise RuntimeError(f"benchmark {worker} worker emitted invalid JSON") from error


def cold_start_worker() -> dict[str, Any]:
    """Import the service and resolve model health in a fresh process."""
    with offline_runtime(), PeakMemorySampler() as memory:
        import_started = time.perf_counter()
        from main import app  # noqa: F401

        import_ms = (time.perf_counter() - import_started) * 1_000
        health_started = time.perf_counter()
        from model.artifact_registry import get_artifact_health

        model_health = get_artifact_health()
        health_ms = (time.perf_counter() - health_started) * 1_000
    return {
        "serviceImportMs": _round(import_ms),
        "artifactHealthResolutionMs": _round(health_ms),
        "workerMeasuredStartupMs": _round(import_ms + health_ms),
        "memory": memory.report(),
        "modelState": {
            "ready": bool(model_health["ready"]),
            "state": model_health["state"],
            "code": model_health["code"],
            "artifactId": model_health["artifactId"],
            "releaseStatus": model_health["releaseStatus"],
        },
    }


def probe_cold_start(run_count: int) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    for _ in range(run_count):
        sample, process_wall_ms = _run_worker("cold-start")
        sample["processLaunchToReadyMs"] = _round(process_wall_ms)
        samples.append(sample)
    return {
        "scope": "fresh Python process, service import, route registration, and fail-closed artifact health resolution",
        "runs": samples,
        "processLaunchToReady": summarize(
            [sample["processLaunchToReadyMs"] for sample in samples], "ms"
        ),
        "serviceImport": summarize([sample["serviceImportMs"] for sample in samples], "ms"),
        "artifactHealthResolution": summarize(
            [sample["artifactHealthResolutionMs"] for sample in samples], "ms"
        ),
        "peakWorkingSet": summarize(
            [
                sample["memory"]["peakWorkingSetMiB"]
                for sample in samples
                if sample["memory"]["peakWorkingSetMiB"] is not None
            ],
            "MiB",
        ),
    }


def accelerator_worker() -> dict[str, Any]:
    torch_version = _package_version("torch")
    result: dict[str, Any] = {
        "torchInstalled": torch_version is not None,
        "torchVersion": torch_version,
        "torchCudaAvailable": False,
        "torchMpsAvailable": False,
        "devices": [],
        "nvidiaSmi": None,
        "runtimeUsable": False,
    }
    if torch_version is not None:
        try:
            import torch

            result["torchCudaAvailable"] = bool(torch.cuda.is_available())
            mps = getattr(torch.backends, "mps", None)
            result["torchMpsAvailable"] = bool(mps and mps.is_available())
            if result["torchCudaAvailable"]:
                for index in range(torch.cuda.device_count()):
                    properties = torch.cuda.get_device_properties(index)
                    result["devices"].append(
                        {
                            "backend": "cuda",
                            "index": index,
                            "name": properties.name,
                            "totalMemoryBytes": int(properties.total_memory),
                            "totalMemoryGiB": _round(properties.total_memory / (1024**3)),
                        }
                    )
            result["runtimeUsable"] = bool(
                result["torchCudaAvailable"] or result["torchMpsAvailable"]
            )
        except Exception as error:
            result["torchProbeError"] = {
                "type": type(error).__name__,
                "winError": getattr(error, "winerror", None),
                "message": "The installed Torch runtime could not initialize for accelerator probing.",
            }

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if completed.returncode == 0:
            rows = []
            for line in completed.stdout.splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) == 3:
                    rows.append(
                        {
                            "name": parts[0],
                            "memoryMiB": int(parts[1]),
                            "driverVersion": parts[2],
                        }
                    )
            result["nvidiaSmi"] = rows
    except (FileNotFoundError, OSError, subprocess.SubprocessError, ValueError):
        result["nvidiaSmi"] = None
    return result


def probe_accelerator() -> dict[str, Any]:
    result, _ = _run_worker("accelerator", timeout_seconds=90)
    return result


def _event_payload(event: str) -> dict[str, Any]:
    data_line = next(line for line in event.splitlines() if line.startswith("data:"))
    return json.loads(data_line.removeprefix("data:").strip())


async def _measure_one_request(sequence: int) -> dict[str, Any]:
    from api.chat import stream_chat_sse
    from api.schemas import ChatRequest
    from config import get_settings

    payload = ChatRequest.model_validate(
        {
            "message": "Explain whether an Arduino Uno LED on pin 13 needs a series resistor.",
            "boardType": "ARDUINO_UNO",
            "components": [{"id": "led-1", "type": "LED", "name": "Status LED"}],
            "wires": [],
            "sessionId": f"runtime-benchmark-{sequence}",
        }
    )
    settings = get_settings()
    started = time.perf_counter()
    first_delta_ms: float | None = None
    characters = 0
    deltas = 0
    completed = False
    error_code: str | None = None
    async for event in stream_chat_sse(payload, settings=settings):
        elapsed_ms = (time.perf_counter() - started) * 1_000
        if event.startswith("event: delta"):
            if first_delta_ms is None:
                first_delta_ms = elapsed_ms
            characters += len(str(_event_payload(event).get("delta") or ""))
            deltas += 1
        elif event.startswith("event: complete"):
            completed = True
        elif event.startswith("event: error"):
            error_code = str(_event_payload(event).get("code") or "UNKNOWN")
    total_ms = (time.perf_counter() - started) * 1_000
    return {
        "completed": completed,
        "errorCode": error_code,
        "firstDeltaMs": None if first_delta_ms is None else _round(first_delta_ms),
        "totalLatencyMs": _round(total_ms),
        "characters": characters,
        "deltaChunks": deltas,
        "charactersPerSecond": _round(characters / max(total_ms / 1_000, 0.000_001)),
    }


async def _run_request_batch(count: int, concurrency: int, sequence_start: int) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(sequence: int) -> dict[str, Any]:
        async with semaphore:
            return await _measure_one_request(sequence)

    return await asyncio.gather(
        *(bounded(sequence_start + offset) for offset in range(count))
    )


async def _generation_benchmark(settings: BenchmarkSettings) -> dict[str, Any]:
    sequence = 0
    for _ in range(settings.warmup_requests):
        await _measure_one_request(sequence)
        sequence += 1

    with PeakMemorySampler() as sequential_memory:
        sequential_started = time.perf_counter()
        sequential = await _run_request_batch(settings.measured_requests, 1, sequence)
        sequential_wall_ms = (time.perf_counter() - sequential_started) * 1_000
    sequence += settings.measured_requests

    concurrency_results: list[dict[str, Any]] = []
    for concurrency in settings.concurrency_levels:
        request_count = max(settings.measured_requests, concurrency * 2)
        with PeakMemorySampler() as memory:
            started = time.perf_counter()
            requests = await _run_request_batch(request_count, concurrency, sequence)
            wall_ms = (time.perf_counter() - started) * 1_000
        sequence += request_count
        completed = sum(1 for request in requests if request["completed"])
        latencies = [request["totalLatencyMs"] for request in requests]
        concurrency_results.append(
            {
                "concurrency": concurrency,
                "requests": request_count,
                "completed": completed,
                "failed": request_count - completed,
                "wallTimeMs": _round(wall_ms),
                "throughputRequestsPerSecond": _round(
                    completed / max(wall_ms / 1_000, 0.000_001)
                ),
                "requestLatency": summarize(latencies, "ms"),
                "memory": memory.report(),
            }
        )

    successful = [request for request in sequential if request["completed"]]
    first_delta = [request["firstDeltaMs"] for request in successful if request["firstDeltaMs"] is not None]
    total_latency = [request["totalLatencyMs"] for request in successful]
    character_rates = [request["charactersPerSecond"] for request in successful]
    return {
        "mode": "local-deterministic",
        "outputUnit": "characters",
        "tokenMetricsApplicable": False,
        "networkAccessAllowed": False,
        "warmupRequests": settings.warmup_requests,
        "measuredRequests": settings.measured_requests,
        "completedRequests": len(successful),
        "failedRequests": len(sequential) - len(successful),
        "sequentialWallTimeMs": _round(sequential_wall_ms),
        "firstDeltaLatency": summarize(first_delta, "ms"),
        "totalRequestLatency": summarize(total_latency, "ms"),
        "generationThroughput": summarize(character_rates, "characters/second"),
        "memory": sequential_memory.report(),
        "samples": sequential,
        "concurrency": concurrency_results,
    }


def benchmark_generation(settings: BenchmarkSettings) -> dict[str, Any]:
    with offline_runtime():
        return asyncio.run(_generation_benchmark(settings))


def _neural_benchmark_state(model_health: dict[str, Any]) -> dict[str, Any]:
    if not model_health["ready"]:
        return {
            "status": "unavailable",
            "reasonCode": model_health["code"],
            "artifactId": model_health["artifactId"],
            "firstTokenLatencyMs": None,
            "tokensPerSecond": None,
            "peakWorkingSetMiB": None,
            "note": "No approved neural artifact exists; deterministic character metrics are not relabelled as token metrics.",
        }
    return {
        "status": "not-implemented",
        "reasonCode": "NEURAL_INFERENCE_RUNTIME_NOT_IMPLEMENTED",
        "artifactId": model_health["artifactId"],
        "firstTokenLatencyMs": None,
        "tokensPerSecond": None,
        "peakWorkingSetMiB": None,
        "note": "The artifact registry is ready, but VFAI-017 neural generation benchmarking is not implemented.",
    }


def derive_cpu_budgets(
    benchmark_id: str,
    cold_start: dict[str, Any],
    generation: dict[str, Any],
) -> dict[str, Any]:
    """Derive generous regression budgets from this measured reference run."""
    startup_max = float(cold_start["processLaunchToReady"]["max"])
    first_delta_p95 = float(generation["firstDeltaLatency"]["p95"])
    character_median = float(generation["generationThroughput"]["median"])
    generation_peak = generation["memory"]["peakWorkingSetMiB"]
    highest_concurrency = generation["concurrency"][-1]
    concurrency_p95 = float(highest_concurrency["requestLatency"]["p95"])
    concurrency_rate = float(highest_concurrency["throughputRequestsPerSecond"])
    measured_peaks = [
        value
        for value in (
            generation_peak,
            cold_start["peakWorkingSet"]["max"],
            *(item["memory"]["peakWorkingSetMiB"] for item in generation["concurrency"]),
        )
        if value is not None
    ]
    max_memory = max(measured_peaks) if measured_peaks else None
    return {
        "profile": "cpu-development-reference",
        "sourceBenchmarkId": benchmark_id,
        "basis": "measured-runtime-with-explicit-headroom",
        "notBasedOn": "artifact name or parameter-size label",
        "calculation": {
            "maximumColdStartMs": "ceil(measured max process launch-to-ready * 1.25 + 5ms)",
            "maximumFirstDeltaMs": "ceil(measured p95 first-delta * 1.50 + 2ms)",
            "minimumCharactersPerSecond": "floor(measured median * 0.70)",
            "maximumWorkingSetMiB": "ceil(max observed working set * 1.20 + 16MiB)",
            "maximumConcurrentP95Ms": "ceil(highest-level measured p95 * 1.50 + 2ms)",
            "minimumConcurrentRequestsPerSecond": "measured highest-level rate * 0.70",
        },
        "limits": {
            "maximumColdStartMs": math.ceil(startup_max * 1.25 + 5),
            "maximumFirstDeltaMs": math.ceil(first_delta_p95 * 1.5 + 2),
            "minimumCharactersPerSecond": math.floor(character_median * 0.7),
            "maximumWorkingSetMiB": None if max_memory is None else math.ceil(max_memory * 1.2 + 16),
            "concurrency": int(highest_concurrency["concurrency"]),
            "maximumConcurrentP95Ms": math.ceil(concurrency_p95 * 1.5 + 2),
            "minimumConcurrentRequestsPerSecond": _round(concurrency_rate * 0.7),
            "maximumFailedRequests": 0,
        },
    }


def _revision() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=AI_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=AI_ROOT,
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        commit = None
        dirty = None
    harness_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {
        "gitCommit": commit,
        "worktreeDirty": dirty,
        "benchmarkHarnessSha256": harness_hash,
    }


def validate_report(report: dict[str, Any]) -> None:
    required = {
        "schemaVersion",
        "benchmarkId",
        "generatedAt",
        "networkAccessAllowed",
        "configuration",
        "hardware",
        "software",
        "model",
        "benchmarks",
        "derivedBudgets",
    }
    missing = sorted(required - report.keys())
    if missing:
        raise ValueError(f"benchmark report is missing fields: {missing}")
    if report["schemaVersion"] != SCHEMA_VERSION:
        raise ValueError(f"schemaVersion must be {SCHEMA_VERSION}")
    if report["networkAccessAllowed"] is not False:
        raise ValueError("runtime baselines must block network access")
    generation = report["benchmarks"]["localDeterministicGeneration"]
    if generation["tokenMetricsApplicable"] is not False:
        raise ValueError("deterministic character throughput cannot be reported as token throughput")
    if generation["completedRequests"] < 1:
        raise ValueError("at least one measured local request must complete")
    if any(item["failed"] for item in generation["concurrency"]):
        raise ValueError("reference concurrency run contains failed requests")
    if report["derivedBudgets"]["sourceBenchmarkId"] != report["benchmarkId"]:
        raise ValueError("derived budgets must identify their measured source benchmark")
    json.dumps(report, allow_nan=False)


def _validate_schema_node(instance: Any, schema: dict[str, Any], path: str) -> None:
    expected_type = schema.get("type")
    type_map = {
        "object": dict,
        "array": list,
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "null": type(None),
    }
    if expected_type in type_map and not isinstance(instance, type_map[expected_type]):
        raise ValueError(f"{path} must have JSON type {expected_type}")
    if "const" in schema and instance != schema["const"]:
        raise ValueError(f"{path} must equal {schema['const']!r}")
    pattern = schema.get("pattern")
    if pattern is not None and isinstance(instance, str) and re.fullmatch(pattern, instance) is None:
        raise ValueError(f"{path} does not match {pattern!r}")
    if isinstance(instance, list):
        minimum = schema.get("minItems")
        if isinstance(minimum, int) and len(instance) < minimum:
            raise ValueError(f"{path} must contain at least {minimum} items")
    if not isinstance(instance, dict):
        return
    required = schema.get("required", [])
    missing = [name for name in required if name not in instance]
    if missing:
        raise ValueError(f"{path} is missing required properties: {missing}")
    properties = schema.get("properties", {})
    if schema.get("additionalProperties") is False:
        unexpected = sorted(set(instance) - set(properties))
        if unexpected:
            raise ValueError(f"{path} has unexpected properties: {unexpected}")
    for name, child_schema in properties.items():
        if name in instance and isinstance(child_schema, dict):
            _validate_schema_node(instance[name], child_schema, f"{path}.{name}")


def validate_schema_contract(
    report: dict[str, Any], schema_path: Path = DEFAULT_SCHEMA
) -> None:
    """Validate the report against the dependency-free checked-in schema subset."""
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if not isinstance(schema, dict):
        raise ValueError("runtime baseline schema must be a JSON object")
    _validate_schema_node(report, schema, "$")


def build_report(settings: BenchmarkSettings, probe_acceleration: bool = True) -> dict[str, Any]:
    settings.validate()
    generated_at = datetime.now(timezone.utc)
    benchmark_id = f"vfai-runtime-{generated_at.strftime('%Y%m%dT%H%M%SZ')}"
    accelerator = (
        probe_accelerator()
        if probe_acceleration
        else {
            "probeSkipped": True,
            "torchInstalled": _package_version("torch") is not None,
            "torchVersion": _package_version("torch"),
            "runtimeUsable": False,
            "devices": [],
        }
    )
    cold_start = probe_cold_start(settings.cold_start_runs)
    generation = benchmark_generation(settings)
    from model.artifact_registry import get_artifact_health

    model_health = get_artifact_health()
    report = {
        "schemaVersion": SCHEMA_VERSION,
        "benchmarkId": benchmark_id,
        "generatedAt": generated_at.isoformat().replace("+00:00", "Z"),
        "networkAccessAllowed": False,
        "revision": _revision(),
        "configuration": {
            "coldStartRuns": settings.cold_start_runs,
            "warmupRequests": settings.warmup_requests,
            "measuredRequests": settings.measured_requests,
            "concurrencyLevels": list(settings.concurrency_levels),
        },
        "hardware": hardware_inventory(accelerator),
        "software": software_inventory(),
        "model": {
            "health": model_health,
            "neuralGeneration": _neural_benchmark_state(model_health),
        },
        "benchmarks": {
            "coldStart": cold_start,
            "localDeterministicGeneration": generation,
        },
    }
    report["derivedBudgets"] = derive_cpu_budgets(
        benchmark_id, cold_start, generation
    )
    validate_report(report)
    validate_schema_contract(report)
    return report


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(output)


def _parse_concurrency(value: str) -> tuple[int, ...]:
    try:
        levels = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("concurrency must be comma-separated integers") from error
    if not levels:
        raise argparse.ArgumentTypeError("at least one concurrency level is required")
    return levels


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cold-start-runs", type=int, default=3)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--measured-requests", type=int, default=5)
    parser.add_argument("--concurrency", type=_parse_concurrency, default=(1, 2, 4))
    parser.add_argument("--skip-accelerator-probe", action="store_true")
    parser.add_argument("--worker", choices=("cold-start", "accelerator"), help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    if arguments.worker == "cold-start":
        print(json.dumps(cold_start_worker(), allow_nan=False))
        return 0
    if arguments.worker == "accelerator":
        print(json.dumps(accelerator_worker(), allow_nan=False))
        return 0
    settings = BenchmarkSettings(
        cold_start_runs=arguments.cold_start_runs,
        warmup_requests=arguments.warmup_requests,
        measured_requests=arguments.measured_requests,
        concurrency_levels=arguments.concurrency,
    )
    report = build_report(settings, probe_acceleration=not arguments.skip_accelerator_probe)
    output = arguments.output.resolve()
    write_report(report, output)
    print(json.dumps({"status": "ok", "benchmarkId": report["benchmarkId"], "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
