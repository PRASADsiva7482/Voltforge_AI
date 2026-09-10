"""Measure actual offline staging at two input sizes in fresh processes."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

AI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI))
from data_governance.ingestion import pipeline as p


def process_peak_bytes():
    if os.name != "nt":
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value if sys.platform == "darwin" else value * 1024
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                                                "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
    value = Counters()
    value.cb = ctypes.sizeof(value)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(value), value.cb):
        raise OSError("Process memory measurement unavailable")
    return value.PeakWorkingSetSize


def worker(count):
    with tempfile.TemporaryDirectory(prefix="vf-ingestion-scale-") as temp:
        root = Path(temp)
        path = root / "synthetic-lines.jsonl"
        with path.open("wb") as stream:
            for i in range(count):
                stream.write((f"Authored throughput fixture {i:08d}: " + "R = V/I; preserve SI notation and code whitespace. " * 20 + "\n").encode())
        p.exclusive_json(root / "catalog.json", {"fixtureOnly": True, "trainingAllowed": False, "files": [
            {"path": path.name, "sha256": p.sources.file_hash(path), "format": "jsonl", "mediaType": "text/plain", "encoding": "utf-8"}]})
        result = p.build(p.fixture_plan(root), root, root / "out", root / "work")
        manifest = p.read_json(Path(result["releasePath"]) / "manifest.json")
        return {"inputDocuments": count, "acceptedDocuments": result["acceptedDocuments"], "quarantinedDocuments": result["quarantinedDocuments"],
                "inputSha256": p.sources.file_hash(path), "manifestSha256": result["manifestSha256"], "performance": result["performance"],
                "parentProcessPeakWorkingSetBytes": process_peak_bytes(), "shards": len(manifest["files"]),
                "largestShardBytes": max(row["bytes"] for row in manifest["files"]), "fixtureOnly": True, "trainingAllowed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise ValueError("Performance receipt exists")
    if args.worker:
        result = worker(args.worker)
    else:
        runs = []
        with tempfile.TemporaryDirectory(prefix="vf-ingestion-measure-") as temp:
            for count in (1000, 10000):
                receipt = Path(temp) / f"{count}.json"
                subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--worker", str(count), "--report", str(receipt)],
                               cwd=AI, check=True, timeout=600, stdout=subprocess.DEVNULL,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                runs.append(p.read_json(receipt))
        small, large = runs
        limit = small["performance"]["pythonPeakAllocatedBytes"] * 2 + 2 * 1024 * 1024
        checks = {"allUniqueDocumentsAccepted": all(r["acceptedDocuments"] == r["inputDocuments"] and r["quarantinedDocuments"] == 0 for r in runs),
                  "tenfoldInput": large["performance"]["inputBytes"] == small["performance"]["inputBytes"] * 10,
                  "boundedPythonAllocation": large["performance"]["pythonPeakAllocatedBytes"] <= limit,
                  "boundedShards": all(r["largestShardBytes"] <= p.POLICY["shardBytes"] for r in runs)}
        result = {"taskId": "LLM-TASK-020", "status": "passed" if all(checks.values()) else "failed", "runs": runs,
                  "checks": checks, "largeRunPythonAllocationLimitBytes": limit,
                  "measurementScope": "Fresh-process, end-to-end JSONL staging including hashing, normalization, SQLite journal, exact dedup, shard publication and verification.",
                  "limits": "Synthetic development throughput only. Parent peak working set is observed, not a hard limit. No server capacity or PDF-child RSS claim.",
                  "trainingAllowed": False, "networkAccessed": False}
    p.exclusive_json(args.report, result)
    print(json.dumps({"status": result.get("status", "measured"), "report": str(args.report)}))
    if result.get("status") == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
