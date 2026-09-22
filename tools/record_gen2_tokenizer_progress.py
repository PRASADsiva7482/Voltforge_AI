"""Snapshot planning before task 024; record only verified preparation evidence."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone, timedelta
import hashlib
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

AI = Path(__file__).resolve().parents[1]
ROOT = AI.parent
PLANNING = (
    "VOLTFORGE_LOCAL_LLM_BUILD_BACKLOG.json", "VOLTFORGE_LOCAL_LLM_BUILD_BACKLOG.md",
    "AI_CHAT_ROOT_CAUSE_BACKLOG.json", "AI_CHAT_ROOT_CAUSE_ANALYSIS.md",
    "Voltforge_AI/docs/VOLTForge_AI_RUNBOOK.md", "Voltforge_AI/docs/runbook-contract.v1.json",
    "Voltforge_AI/.gitattributes",
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def descriptor(path):
    raw = path.read_bytes()
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def snapshot():
    target = ROOT / "backlog_history/llm-task-024-before"
    target.mkdir(parents=True, exist_ok=False)
    entries = []
    for name in PLANNING:
        source = ROOT / name
        backup = target / name
        backup.parent.mkdir(parents=True, exist_ok=True)
        with backup.open("xb") as stream:
            stream.write(source.read_bytes())
        entries.append({**descriptor(source), "backup": name})
    write(target / "manifest.json", {"taskId": "LLM-TASK-024", "files": entries})
    print(json.dumps({"status": "snapshotted", "files": len(entries)}))


def record():
    sys.path.insert(0, str(AI))
    from model.gen2_tokenizer.release import fingerprints, verify_fixture
    from data_governance.pretraining import corpus

    reports = AI / "evaluation/reports"
    receipt_path = reports / "llm-task-024-progress.json"
    checks_path = reports / "llm-task-024-checks.json"
    if receipt_path.exists() or checks_path.exists():
        raise ValueError("Use a new evidence version; task-024 receipts are write-once")
    comparison_path = reports / "llm-task-024-comparison.json"
    comparison = read(comparison_path)
    if comparison["status"] != "passed-fixture-comparison-only" or comparison["sourceFingerprints"] != fingerprints():
        raise ValueError("Comparison evidence does not match the implementation")
    fixtures_verified = []
    for item in comparison["candidates"]:
        tokenizer, manifest = verify_fixture(AI / item["fixtureRelease"], recompute=True)
        if tokenizer.binding() != item["binding"] or manifest["targetReached"] or tokenizer.vocab_size != 911:
            raise ValueError("Unexpected fixture result; review before recording")
        fixtures_verified.append(descriptor(AI / item["fixtureRelease"] / "manifest.json"))
    junit = reports / "llm-task-024-tests.xml"
    tree = ET.parse(junit).getroot()
    cases = list(tree.iter("testcase"))
    if len(cases) < 58 or any(list(tree.iter(tag)) for tag in ("failure", "error", "skipped")):
        raise ValueError("Task-024 regression evidence incomplete")
    fixture_tests = [case for case in cases if case.get("classname", "").endswith("test_gen2_tokenizer")]
    if len(fixture_tests) != 58:
        raise ValueError("Required tokenizer tests missing")
    for name in ("test_release_retrains_exactly_and_beats_legacy_with_lower_vocab_cost", "test_compiler_uses_verified_owned_tokenizer_and_exact_budget"):
        if not any(case.get("name") == name for case in cases):
            raise ValueError("Required historical compatibility regression missing")
    preserved = ROOT / "backlog_history/llm-task-024-before"
    for entry in read(preserved / "manifest.json")["files"]:
        if hashlib.sha256((preserved / entry["backup"]).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Planning snapshot changed")
    candidate = read(reports / "llm-task-023-corpus-build.json")
    # New code never fits this corpus. Verify frozen bytes/bindings; task 023's
    # unchanged full recomputation evidence remains in its own historical report.
    corpus_verified = corpus.verify(Path(candidate["releasePath"]), recompute=False)
    checks = []
    for command in (
        ["tools/verify_llm_foundation.py", "--check-preserved-history"],
        ["tools/verify_foundation_evaluation_binding.py"],
        ["tools/inventory_llm_sources.py", "verify", "--recompute"],
        ["tools/audit_data_governance.py", "--check"],
        ["tools/verify_runbook.py"],
        ["tools/scan_quality_boundaries.py", "--check"],
        ["tools/snapshot_llm_cleanup.py", "--verify", "../backlog_history/llm-task-016-before-20260906T054449Z"],
        ["tools/build_gen2_tokenizer.py", "check-corpus", "--release", candidate["releasePath"]],
    ):
        print("Checking " + " ".join(command), flush=True)
        result = subprocess.run([sys.executable, "-B", *command], cwd=AI, capture_output=True, text=True, encoding="utf-8")
        expected = 2 if command[0] == "tools/build_gen2_tokenizer.py" else 0
        if result.returncode != expected:
            raise ValueError("Required check failed: " + " ".join(command) + "\n" + result.stdout + result.stderr)
        output = json.loads(result.stdout)
        if expected == 2 and output.get("code") != "GEN2_CORPUS_NOT_ADMITTED":
            raise ValueError("Expected candidate fitting denial missing")
        checks.append({"command": [".toolchains/gen1/Scripts/python.exe", "-B", *command], "exitCode": result.returncode, "expectedExitCode": expected, "result": output})

    date = datetime.now(timezone(timedelta(hours=5, minutes=30))).date().isoformat()
    summary = (
        "Implemented a separate owned Gen2 byte-BPE codec, fixed 16 special-token IDs, versioned chat framing, "
        "exact context/output budgeting, shared model/template descriptors, strict incremental Unicode decoding, "
        "and immutable fixture artifact verification. Independent pair-recount and exhaustive-encoder tests "
        "check the reused BPE mathematics. Eight authored implementation inputs and six separate measurement "
        "fixtures compare 16,384/32,768 vocabulary ceilings with the existing 3,072-entry baseline. Both fixture "
        "fits stop at 911 entries; neither reaches its target and no production vocabulary is selected. "
        f"All {len(cases)} focused and historical-compatibility tests passed, including 58 new tokenizer controls. "
        "The current task-023 corpus is denied before shard reads; caller approval flags and fixture artifacts "
        "cannot enable fitting or serving. Task 024 remains IN_PROGRESS with dependency LLM-TASK-023: source "
        "admission, approved-corpus candidate fitting/comparison, production release and real Gen2 model/runtime "
        "integration remain open. Only authored fixture tokenizers were fitted; no approved-corpus fit, model "
        "training, activation or application routing change occurred."
    )
    remaining = [
        "Complete LLM-TASK-023 admitted pilot/input corpus, including independent language and validation coverage and source-use rights.",
        "Implement an evidence-verifying admitted-corpus adapter; fit only permitted training shards, with finite resource budgets and no validation/acceptance fitting.",
        "Compare actual 16,384/32,768 candidates on representative permitted measurements, select vocabulary size and recount corpus tokens.",
        "Publish a distinct immutable production 0.1.0 tokenizer with approved corpus lineage and complete roundtrip/marker evidence.",
        "Validate the selected tokenizer/template descriptor in actual Gen2 model and context/inference integration; fixture binding checks are interface evidence only.",
    ]
    relative_receipt = receipt_path.relative_to(ROOT).as_posix()
    evidence = [relative_receipt, comparison_path.relative_to(ROOT).as_posix(), junit.relative_to(ROOT).as_posix(), checks_path.relative_to(ROOT).as_posix(), "Voltforge_AI/docs/LLM_GEN2_TOKENIZER_PREPARATION.v1.md"] + [item["path"] for item in fixtures_verified]
    backlog = read(ROOT / PLANNING[0])
    task = next(item for item in backlog["tasks"] if item["id"] == "LLM-TASK-024")
    if task["status"] != "TODO" or task["depends_on"] != ["LLM-TASK-023"] or backlog["execution"]["next_task_id"] != "LLM-TASK-023":
        raise ValueError("Unexpected planning state; review before recording")
    task.update(
        status="IN_PROGRESS", started_on=date, implementation_state="verified-independent-tokenizer-preparation-corpus-dependency-open",
        verification_status="passed-fixture-and-contract-controls-not-production-tokenizer-acceptance",
        verification_result=summary, evidence_records=evidence, remaining_release_gates=remaining,
        blocked_by=["LLM-TASK-023"], preparation_authorization="User explicitly selected LLM-TASK-024; independent fixture work does not bypass its corpus dependency.",
        target_paths=["model/gen2_tokenizer/", "model/tokenizers/fixtures/gen2/", "context_compiler/gen2.py", "tools/build_gen2_tokenizer.py", "tests/test_gen2_tokenizer.py"],
    )
    backlog["updated_on"] = date
    backlog["scope_of_this_revision"] = "Tasks 015 through 022 completed; 023 corpus admission and 024 tokenizer release remain in progress. Task 024 independent fixture preparation is verified."
    counts = Counter(item["status"] for item in backlog["tasks"])
    backlog["status_summary"] = {key: counts[key] for key in ("TODO", "IN_PROGRESS", "COMPLETED")}
    next(item for item in backlog["milestones"] if item["id"] == "M2")["status"] = "IN_PROGRESS"
    backlog["current_checkpoint"].update(active_task_id="LLM-TASK-024", active_task_evidence=relative_receipt, active_task_summary=summary, blocking_task_id="LLM-TASK-023")
    backlog["tokenizer_preparation"] = {"version": "0.1.0-fixture.1", "candidate_targets": [16384, 32768], "actual_fixture_vocabularies": [911, 911], "production_release_accepted": False, "corpus_dependency": "LLM-TASK-023", "evidence": relative_receipt}
    markdown = (ROOT / PLANNING[1]).read_text(encoding="utf-8")
    markdown = markdown.replace("8 completed, 1 in progress (LLM-TASK-023), 32 TODO", "8 completed, 2 in progress (LLM-TASK-023 and LLM-TASK-024), 31 TODO", 1)
    start = markdown.index("**Continue: LLM-TASK-023")
    markdown = markdown[:start] + "**Requested work: LLM-TASK-024 — tokenizer preparation verified; corpus dependency remains open.**\n\n" + summary + "\n\n[Task 024 details](Voltforge_AI/docs/LLM_GEN2_TOKENIZER_PREPARATION.v1.md) | [Progress evidence](" + relative_receipt + ")\n\n" + markdown[start:]
    heading = "#### LLM-TASK-024 — " + task["title"]
    before, section = markdown.split(heading, 1)
    section = section.replace("**Status:** TODO.", "**Status:** IN_PROGRESS.", 1)
    markdown = before + heading + section
    markdown = markdown.replace("\n#### LLM-TASK-025", "\n**Verified preparation (" + date + "):** " + summary + "\n\n**Remaining release gates:**\n\n" + "\n".join("- " + item for item in remaining) + "\n\n**Evidence:** [Tokenizer preparation](Voltforge_AI/docs/LLM_GEN2_TOKENIZER_PREPARATION.v1.md), [comparison](Voltforge_AI/evaluation/reports/llm-task-024-comparison.json), [progress receipt](" + relative_receipt + ").\n\n#### LLM-TASK-025", 1)
    markdown = markdown.replace("| LLM-TASK-024 through LLM-TASK-026 | TODO |", "| LLM-TASK-024 through LLM-TASK-026 | IN_PROGRESS |", 1)
    markdown = re.sub(r"^Updated: [\d-]+\.", "Updated: " + date + ".", markdown, count=1, flags=re.MULTILINE)
    causes = read(ROOT / PLANNING[2])
    causes["updated_on"] = date
    causes["summary"].update(implementation_this_revision=backlog["scope_of_this_revision"], acceptance_limit=summary + " Root causes retain their previous open/partial status.")
    causes["tokenizer_preparation"] = backlog["tokenizer_preparation"]
    causes["current_checkpoint"].update(active_task_id="LLM-TASK-024", active_task_evidence=relative_receipt, blocking_task_id="LLM-TASK-023")
    for path in evidence:
        if path not in causes["evidence_files"]:
            causes["evidence_files"].append(path)
    analysis = (ROOT / PLANNING[3]).read_text(encoding="utf-8")
    analysis = analysis.replace("## Current work: LLM-TASK-023 in progress", "## Current work: LLM-TASK-024 preparation; corpus dependency open\n\n" + summary + "\n\n[Task 024 implementation and remaining gates](Voltforge_AI/docs/LLM_GEN2_TOKENIZER_PREPARATION.v1.md)\n\n## Blocking dependency: LLM-TASK-023 in progress", 1)
    write(checks_path, {"status": "passed", "checks": checks, "fixtureManifests": fixtures_verified, "task023ByteBindingVerification": corpus_verified, "task023RecomputedThisTurn": False})
    receipt = {"schemaVersion": 1, "taskId": "LLM-TASK-024", "status": "planning-check-pending", "recordedAtUtc": datetime.now(timezone.utc).isoformat(), "summary": summary,
        "tests": {"passed": len(cases), "newTokenizerControls": len(fixture_tests), "failed": 0, "skipped": 0, "junit": descriptor(junit)},
        "comparison": descriptor(comparison_path), "checks": descriptor(checks_path), "fixtureManifests": fixtures_verified,
        "taskCompleted": False, "approvedCorpusFitted": False, "tokenizerReleaseAccepted": False, "modelTrainingPerformed": False, "runtimeChanged": False,
        "remainingWork": remaining, "blockedBy": ["LLM-TASK-023"], "nextReadyTask": "LLM-TASK-023", "planningSnapshot": descriptor(preserved / "manifest.json")}
    write(receipt_path, receipt)
    write(ROOT / PLANNING[0], backlog)
    (ROOT / PLANNING[1]).write_text(markdown, encoding="utf-8", newline="\n")
    write(ROOT / PLANNING[2], causes)
    (ROOT / PLANNING[3]).write_text(analysis, encoding="utf-8", newline="\n")
    result = subprocess.run([sys.executable, "-B", "tools/verify_llm_backlogs.py"], cwd=AI, capture_output=True, text=True, encoding="utf-8")
    if result.returncode:
        raise ValueError("Planning check failed; receipt remains pending: " + result.stdout + result.stderr)
    receipt["planningCheck"] = json.loads(result.stdout)
    receipt["sourceFingerprints"] = [descriptor(ROOT / name) for name in PLANNING] + [descriptor(AI / name) for name in ("docs/LLM_GEN2_TOKENIZER_PREPARATION.v1.md", "tools/build_gen2_tokenizer.py", "tools/record_gen2_tokenizer_progress.py", "tests/test_gen2_tokenizer.py")]
    receipt["status"] = "in-progress-tokenizer-preparation-verified-corpus-release-pending"
    write(receipt_path, receipt)
    print(json.dumps({"status": receipt["status"], "testsPassed": len(cases), "nextReadyTask": receipt["nextReadyTask"]}))


if __name__ == "__main__":
    if sys.argv[1:] == ["snapshot"]:
        snapshot()
    elif sys.argv[1:] == ["record"]:
        record()
    else:
        raise SystemExit("Use snapshot or record")
