"""Task 023 source review, measured release and verified input handoff CLI."""
import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

AI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI))
from data_governance.pilot_corpus import release as r


def peak_resident():
    if os.name == "nt":
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [(name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise OSError(ctypes.get_last_error(), "Peak resident measurement failed")
        return counters.PeakWorkingSetSize
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def measure_worker():
    started = time.perf_counter()
    _, plan, score = r.compute()
    receipt = {"schemaVersion": 1, "scope": "fresh-process-full-input-normalization-proxy-tokenization-indexed-matching-and-quality-verification", "freshProcess": True,
               "planSha256": r.sha(r.canonical(plan)), "elapsedSeconds": round(time.perf_counter()-started, 6), "peakResidentBytes": peak_resident(),
               "platform": sys.platform, "pythonVersion": sys.version.split()[0], "candidateDocuments": score["candidateDocuments"], "normalizedInputBytes": score["normalizedInputBytes"],
               "retainedAcquisitionBytes": score["retainedAcquisitionBytes"], "networkAccessed": False,
               "acquisitionMeasurementScope": "retained exact acquisition byte budgets and full source hash/license/tree verification; network latency is not remeasured",
               "allInputProxyTokensRecounted": True, "protectedDescriptors": score["protectedDescriptors"]}
    if receipt["elapsedSeconds"] > r.POLICY["maximumPipelineSeconds"] or receipt["peakResidentBytes"] > r.POLICY["maximumPeakResidentBytes"]:
        raise ValueError("Real-input resource budget exceeded")
    receipt["contentId"] = r.sha(r.canonical(receipt))
    target = r.RESOURCE_OUTPUT / receipt["contentId"] / "measurement.json"
    r.write_immutable(target, r.data(receipt))
    print(json.dumps({"status": "passed-real-input-resource-budget", "receipt": target.relative_to(AI).as_posix(), "measurement": receipt}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("review-sources", "measure", "_measure-worker", "build", "verify", "require-use", "source-impact"))
    parser.add_argument("--resource", type=Path)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--usage")
    parser.add_argument("--split")
    parser.add_argument("--source-id")
    args = parser.parse_args()
    if args.command == "review-sources":
        print(json.dumps(r.record_source_decisions()))
    elif args.command == "measure":
        # The worker has its own process-lifetime peak and a hard elapsed timeout.
        result = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "_measure-worker"], cwd=AI, capture_output=True, text=True, encoding="utf-8", timeout=r.POLICY["maximumPipelineSeconds"]+15)
        if result.returncode:
            raise RuntimeError(result.stderr or result.stdout)
        print(result.stdout, end="")
    elif args.command == "_measure-worker":
        measure_worker()
    elif args.command == "build":
        if not args.resource:
            parser.error("build requires --resource")
        target, score = r.build(args.resource)
        print(json.dumps({"status": "released-admitted-pilot-input-corpus", "path": target.relative_to(AI).as_posix(), "scorecard": score}))
    else:
        if not args.release:
            parser.error("command requires --release")
        if args.command == "verify":
            manifest = r.verify(args.release, recompute=args.recompute)
            print(json.dumps({"status": "passed", "contentId": manifest["contentId"], "recomputed": args.recompute, "trainingRunApproved": False}))
        elif args.command == "require-use":
            result = r.require_use(args.release, args.usage, split=args.split)
            print(json.dumps({"status": "passed-input-use", "usage": result["usage"], "split": result["split"], "trainingRunApproved": False}))
        elif args.command == "source-impact":
            r.verify(args.release, recompute=False)
            if not args.source_id:
                parser.error("source-impact requires --source-id")
            rows = [row for row in r.rows_at(args.release) if row["sourceId"] == args.source_id]
            print(json.dumps({"sourceId": args.source_id, "scope": "this input release; downstream releases must record this manifest as lineage", "documents": [{"documentId": row["id"], "split": row["split"], "normalizedSha256": row["normalizedSha256"], "tokens": row["tokens"]} for row in rows], "count": len(rows)}))


if __name__ == "__main__":
    main()
