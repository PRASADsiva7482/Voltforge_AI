"""Bind LLM-TASK-016 cleanup evidence to source backups and verified receipts.

Run only after pytest XML, the runtime audit and cross-repository fast lane exist.
This creates a review diff and evidence index; it never approves a model release.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import difflib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from snapshot_llm_cleanup import AI, ROOT, descriptor, sha


REPORTS = AI / "evaluation/reports"
NEW_FILES = ["engine/deterministic_tools.py", "tests/test_llm_cleanup.py", "tools/record_llm_cleanup.py"]


def test_receipt(path: Path) -> dict:
    suites = ET.parse(path).getroot().findall("testsuite")
    counts = {key: sum(int(suite.get(key, "0")) for suite in suites)
              for key in ("tests", "failures", "errors", "skipped")}
    if not counts["tests"] or counts["failures"] or counts["errors"]:
        raise RuntimeError("Test receipt did not pass: " + str(path))
    cases = [case for suite in suites for case in suite.findall("testcase")]
    return {**descriptor(path), **counts, "tests": len(cases),
            "passed": sum(case.find("skipped") is None for case in cases),
            "additionalPassedSubtests": counts["tests"] - len(cases),
            "junitReportedTestsIncludingSubtests": counts["tests"],
            "seconds": sum(float(suite.get("time", "0")) for suite in suites)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    args = parser.parse_args()
    snapshot = args.snapshot.resolve()
    manifest_path = snapshot / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    changes, diff = [], []
    for item in manifest["source_files"]:
        saved = snapshot / "files" / item["path"]
        if sha(saved) != item["sha256"]:
            raise RuntimeError("Source backup mismatch: " + item["path"])
        current = ROOT / item["path"]
        if sha(current) != item["sha256"]:
            changes.append({**descriptor(current), "beforeSha256": item["sha256"]})
            diff.extend(difflib.unified_diff(
                saved.read_text(encoding="utf-8").splitlines(keepends=True),
                current.read_text(encoding="utf-8").splitlines(keepends=True),
                fromfile="before/" + item["path"], tofile="after/" + item["path"],
            ))
    for relative in NEW_FILES:
        current = AI / relative
        changes.append({**descriptor(current), "beforeSha256": None})
        diff.extend(difflib.unified_diff([], current.read_text(encoding="utf-8").splitlines(keepends=True),
                                         fromfile="/dev/null", tofile="after/Voltforge_AI/" + relative))
    for item in manifest["protected_files"]:
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError("Protected artifact mismatch: " + item["path"])
    audit_path = REPORTS / "llm-task-016-runtime-audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    for item in audit["source_fingerprints"]:
        if sha(AI / item["path"]) != item["sha256"]:
            raise RuntimeError("Audit source changed; rerun audit: " + item["path"])
    probes = audit["synthetic_probes"]
    if (audit["registry"]["activeArtifactId"] is not None
            or probes["runtime_start"]["ready"] is not False
            or probes["wrapper_health_without_loading_weights"]["ready"] is not False
            or probes["blocked_socket_attempts"]):
        raise RuntimeError("Audit does not match the unavailable/offline cleanup contract")
    chats = probes["chat_results"]
    if len(chats) != 8 or any(row["mode"] != "deterministic-fallback" or row["neural_attempted"] for row in chats):
        raise RuntimeError("Unexpected chat source attribution")
    quality_path = REPORTS / "llm-task-016-cross-repository-quality.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    if quality["lane"] != "fast" or not quality["checks"] or any(row["status"] != "pass" for row in quality["checks"]):
        raise RuntimeError("Cross-repository fast lane did not pass")
    tests = [test_receipt(REPORTS / name) for name in
             ("llm-task-016-ai-tests.xml", "llm-task-016-focused-tests.xml")]
    patch_path = REPORTS / "llm-task-016-cleanup.patch"
    patch_path.write_text("".join(diff), encoding="utf-8", newline="\n")
    report = {
        "schemaVersion": 1, "taskId": "LLM-TASK-016", "status": "verified-cleanup",
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "snapshot": descriptor(manifest_path),
        "verifiedSourceBackups": len(manifest["source_files"]),
        "unchangedProtectedFiles": len(manifest["protected_files"]),
        "cleanupDiff": descriptor(patch_path), "changedSources": changes,
        "tests": tests,
        "testScope": "Full suite plus focused regressions after final adapter fixes; suites overlap and counts must not be added.",
        "crossRepositoryFastLane": {**descriptor(quality_path), **quality["scorecard"]["summary"]},
        "runtimeAudit": {**descriptor(audit_path), "chatProbes": len(chats),
                         "deterministicFallbacks": len(chats), "neuralAttempts": 0,
                         "activeArtifactId": None, "ready": False},
        "verifiedChanges": [
            "Signed, approved, compatible package required before loader import; loaded identity and readiness are checked.",
            "Loader failures unload partial state and never substitute deterministic rules as neural output.",
            "Legacy generation facade is unavailable; no hardcoded model capacity, device, streaming or readiness.",
            "Activation shortcut is inert on import and exits nonzero without changing files.",
            "Compatibility and pinned-runtime bypass flags are ineffective; checkpoint security block is preserved.",
            "Plain prose cannot bypass structured generation validation; uncertainty cannot conceal unsafe instructions.",
            "Named deterministic tools retain evidenced hardware lookups and independently checked calculations.",
        ],
        "limits": [
            "No model training, release approval, activation, deployment or browser verification.",
            "No measured neural throughput or model capability acceptance; loader success tests use signed fixtures and a stub runtime.",
            "Registry, model binaries, tokenizer releases and synthetic releases remain unchanged; broader owner backup policy remains open.",
            "Existing routing, repeated grounding text, context/memory integration and true streaming remain in later backlog tasks.",
            "Cross-repository fast checks are development evidence, not the model release lane or deployed end-to-end acceptance.",
        ],
        "nextTask": "LLM-TASK-017",
    }
    output = REPORTS / "llm-task-016-cleanup.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output), "sha256": sha(output), "changedSources": len(changes),
                      "protectedFilesUnchanged": len(manifest["protected_files"])}))


if __name__ == "__main__":
    main()
