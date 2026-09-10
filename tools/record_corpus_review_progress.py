"""Snapshot and record task-023 leakage precision review without corpus admission."""
from pathlib import Path
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
    "Voltforge_AI/.gitattributes", "Voltforge_AI/docs/LLM_PRETRAINING_CORPUS_CANDIDATE.v1.md",
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def descriptor(path):
    raw = path.read_bytes()
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def snapshot():
    target = ROOT / "backlog_history/llm-task-023-precision-before"
    target.mkdir(parents=True, exist_ok=False)
    entries = []
    for name in PLANNING:
        source, backup = ROOT / name, target / name
        backup.parent.mkdir(parents=True, exist_ok=True)
        with backup.open("xb") as stream:
            stream.write(source.read_bytes())
        entries.append({**descriptor(source), "backup": name})
    write(target / "manifest.json", {"taskId": "LLM-TASK-023", "substep": "leakage-precision-review", "files": entries})
    print(json.dumps({"status": "snapshotted", "files": len(entries)}))


def record():
    sys.path.insert(0, str(AI))
    from data_governance.leakage_review import evaluation
    from model.gen2_tokenizer.release import verify_fixture
    reports = AI / "evaluation/reports"
    receipt_path = reports / "llm-task-023-precision-progress.json"
    checks_path = reports / "llm-task-023-precision-checks.json"
    if receipt_path.exists() or checks_path.exists():
        raise ValueError("Use a new progress evidence version")
    artifacts = sorted(evaluation.ARTIFACTS.glob("*/manifest.json"))
    if len(artifacts) != 1:
        raise ValueError("Select the exact precision review release before recording")
    target = artifacts[0].parent
    evaluation.verify(target, recompute=False)
    controls = read(target / "controls.json")
    real = read(target / "real-review.json")
    if controls["activeMatcherConfusionOnAuthoredControls"] != {"truePositive": 32, "falseNegative": 0, "falsePositive": 16, "trueNegative": 8}:
        raise ValueError("Unexpected precision controls; review before recording")
    if real["review"]["classificationCounts"] != {"lexical-review-required": 2} or real["quarantineFromFamilyPropagation"] != 5 or real["quarantinedDocuments"] != 7:
        raise ValueError("Unexpected real corpus impact")
    junit = reports / "llm-task-023-precision-tests.xml"
    tree = ET.parse(junit).getroot()
    cases = list(tree.iter("testcase"))
    if any(list(tree.iter(tag)) for tag in ("failure", "error", "skipped")):
        raise ValueError("Required regressions failed or skipped")
    focused = [case for case in cases if case.get("classname", "").endswith("test_corpus_leakage_review")]
    if len(focused) != 26 or not any(case.get("name") == "test_real_review_and_partition_preservation_reproduce" for case in cases):
        raise ValueError("Precision/recompute test evidence missing")
    snapshot_root = ROOT / "backlog_history/llm-task-023-precision-before"
    for item in read(snapshot_root / "manifest.json")["files"]:
        if hashlib.sha256((snapshot_root / item["backup"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Planning snapshot changed")
    tokenizer_evidence = read(reports / "llm-task-024-comparison.json")
    for row in tokenizer_evidence["candidates"]:
        verify_fixture(AI / row["fixtureRelease"], recompute=True)
    checks = []
    for command in (
        ["tools/verify_llm_foundation.py", "--check-preserved-history"],
        ["tools/verify_foundation_evaluation_binding.py"],
        ["tools/inventory_llm_sources.py", "verify", "--recompute"],
        ["tools/audit_data_governance.py", "--check"],
        ["tools/verify_runbook.py"],
        ["tools/scan_quality_boundaries.py", "--check"],
        ["tools/snapshot_llm_cleanup.py", "--verify", "../backlog_history/llm-task-016-before-20260906T054449Z"],
    ):
        print("Checking " + " ".join(command), flush=True)
        result = subprocess.run([sys.executable, "-B", *command], cwd=AI, capture_output=True, text=True, encoding="utf-8")
        if result.returncode:
            raise ValueError("Required check failed: " + " ".join(command) + "\n" + result.stdout + result.stderr)
        checks.append({"command": [".toolchains/gen1/Scripts/python.exe", "-B", *command], "exitCode": 0, "result": json.loads(result.stdout)})
    date = datetime.now(timezone(timedelta(hours=5, minutes=30))).date().isoformat()
    summary = (
        "Completed the independent leakage-precision review substep for task 023. The read-only reviewer "
        "separates identity/literal/code signals from lexical-only matches and exports counts without protected "
        "prompt text. Across 56 authored controls from 10 families, the unchanged matcher blocks all 32 protected "
        "positives but also 16 of 24 unrelated negatives. Two genuine paraphrase controls remain blocked for "
        "review, so word-overlap matches cannot simply be ignored. On the real 38-document corpus and 1,005 "
        "guards, both direct matches are lexical-only; family propagation quarantines five additional documents. "
        "The two real matches remain unadjudicated, and all seven existing quarantines and 50 inherited exclusions "
        "remain intact. Partition decisions reproduce exactly. "
        f"All {len(cases)} regression tests passed, including 26 precision-review tests and full review recomputation; "
        "governance, source, runbook, tokenizer and historical artifact checks passed. No matcher threshold, "
        "source permission, corpus admission, tokenizer fit, model or serving route changed. Task 023 remains "
        "IN_PROGRESS for exact source-use admission, independent language/validation coverage, corpus-scale "
        "measurements and admitted input-corpus release; task 024's production tokenizer remains blocked."
    )
    paths = [receipt_path, checks_path, junit, target / "manifest.json", target / "controls.json", target / "real-review.json", AI / "docs/LLM_CORPUS_LEAKAGE_PRECISION.v1.md"]
    evidence = [path.relative_to(ROOT).as_posix() for path in paths]
    relative_receipt = receipt_path.relative_to(ROOT).as_posix()
    backlog = read(ROOT / PLANNING[0])
    task = next(row for row in backlog["tasks"] if row["id"] == "LLM-TASK-023")
    if task["status"] != "IN_PROGRESS" or backlog["execution"]["next_task_id"] != "LLM-TASK-023":
        raise ValueError("Unexpected dependency state")
    task.update(verification_result=summary, verification_status="passed-precision-review-and-preservation-corpus-admission-open")
    task.setdefault("completed_substeps", []).append({"id": "independent-leakage-precision-review", "completed_on": date, "evidence": relative_receipt, "scope": "diagnostic review only; no matcher change or corpus readmission"})
    task["evidence_records"].extend(path for path in evidence if path not in task["evidence_records"])
    task["remaining_release_gates"] = [
        "Complete source-use admission for exact source bytes and retained license/privacy/retention/removal obligations; candidate acquisition is not training approval.",
        "Add independent general-language/technical, code, electronics and math families with populated validation coverage; current language training and validation coverage remain empty.",
        "Independent lexical-precision controls are verified. Before adopting a changed matcher for fresh data, freeze and validate its precision/recall and review process; do not automatically reverse old exclusions.",
        "Measure acquisition, normalization, tokenization and matching on the intended corpus scale with finite resource budgets.",
        "Publish an admitted pilot/input corpus with a provisional unique-token budget; Gen2 recounting belongs to task 024 and measured production acceptance remains in tasks 031/032.",
    ]
    backlog["updated_on"] = date
    backlog["scope_of_this_revision"] = "Tasks 015 through 022 completed; 023 independent precision review and 024 tokenizer preparation verified, with corpus and production tokenizer admission still open."
    backlog["current_checkpoint"].update(active_task_id="LLM-TASK-023", active_task_evidence=relative_receipt, active_task_summary=summary)
    backlog["current_checkpoint"].pop("blocking_task_id", None)
    precision = {"status": "completed-diagnostic-substep", "evidence": relative_receipt, "cases": 56, "families": 10, "protected_positive_controls_blocked": 32, "unrelated_negative_controls_matched": 16, "real_matches_require_review": 2, "corpus_decisions_changed": 0, "training_allowed": False}
    backlog["leakage_precision_review"] = precision
    markdown = (ROOT / PLANNING[1]).read_text(encoding="utf-8")
    markdown = re.sub(r"^Updated: [\d-]+\.", "Updated: " + date + ".", markdown, count=1, flags=re.MULTILINE)
    marker = "**Requested work: LLM-TASK-024 — tokenizer preparation verified; corpus dependency remains open.**"
    if marker not in markdown:
        raise ValueError("Expected tokenizer checkpoint missing from Markdown")
    markdown = markdown.replace(marker, "**Continue: LLM-TASK-023 — leakage precision review completed; corpus admission remains open.**\n\n" + summary + "\n\n[Precision review](Voltforge_AI/docs/LLM_CORPUS_LEAKAGE_PRECISION.v1.md) | [Verification](" + relative_receipt + ")\n\n**Previous checkpoint: LLM-TASK-024 tokenizer preparation verified; production release pending.**", 1)
    markdown = markdown.replace("**Continue: LLM-TASK-023 — pretraining corpus release remains IN_PROGRESS.**", "**Earlier checkpoint: LLM-TASK-023 candidate corpus assembled; release remains IN_PROGRESS.**", 1)
    start = markdown.index("#### LLM-TASK-023")
    end = markdown.index("#### LLM-TASK-024", start)
    markdown = markdown[:end] + "**Precision review (" + date + "):** " + summary + "\n\n**Remaining release work:**\n\n" + "\n".join("- " + item for item in task["remaining_release_gates"]) + "\n\n**Review evidence:** [Immutable review](" + (target / "manifest.json").relative_to(ROOT).as_posix() + "), [progress receipt](" + relative_receipt + ").\n\n" + markdown[end:]
    causes = read(ROOT / PLANNING[2])
    causes["updated_on"] = date
    causes["summary"].update(implementation_this_revision=backlog["scope_of_this_revision"], acceptance_limit=summary + " Root-cause statuses are unchanged.")
    causes["leakage_precision_review"] = precision
    causes["current_checkpoint"].update(active_task_id="LLM-TASK-023", active_task_evidence=relative_receipt)
    causes["current_checkpoint"].pop("blocking_task_id", None)
    causes["evidence_files"].extend(path for path in evidence if path not in causes["evidence_files"])
    analysis = (ROOT / PLANNING[3]).read_text(encoding="utf-8")
    analysis = analysis.replace("## Current work: LLM-TASK-024 preparation; corpus dependency open", "## Current work: LLM-TASK-023 leakage precision review\n\n" + summary + "\n\n[Precision findings and next steps](Voltforge_AI/docs/LLM_CORPUS_LEAKAGE_PRECISION.v1.md)\n\n## Previous checkpoint: LLM-TASK-024 preparation; corpus dependency open", 1)
    write(checks_path, {"status": "passed", "checks": checks, "reviewManifest": descriptor(target / "manifest.json"), "newTokenizerFixturesRecomputedUnchanged": True, "fullReviewRecomputationEvidence": descriptor(junit)})
    receipt = {"schemaVersion": 1, "taskId": "LLM-TASK-023", "substep": "independent-leakage-precision-review", "status": "planning-check-pending",
        "recordedAtUtc": datetime.now(timezone.utc).isoformat(), "summary": summary, "reviewManifest": descriptor(target / "manifest.json"),
        "tests": {"passed": len(cases), "precisionReviewTests": len(focused), "failed": 0, "skipped": 0, "junit": descriptor(junit)},
        "checks": descriptor(checks_path), "planningSnapshot": descriptor(snapshot_root / "manifest.json"),
        "substepCompleted": True, "taskCompleted": False, "corpusReleased": False, "matcherChanged": False, "trainingAllowed": False,
        "realMatchAdjudicationCompleted": False, "automaticReadmissions": 0, "nextTask": "LLM-TASK-023", "remainingWork": task["remaining_release_gates"]}
    write(receipt_path, receipt)
    write(ROOT / PLANNING[0], backlog)
    (ROOT / PLANNING[1]).write_text(markdown, encoding="utf-8", newline="\n")
    write(ROOT / PLANNING[2], causes)
    (ROOT / PLANNING[3]).write_text(analysis, encoding="utf-8", newline="\n")
    result = subprocess.run([sys.executable, "-B", "tools/verify_llm_backlogs.py"], cwd=AI, capture_output=True, text=True, encoding="utf-8")
    if result.returncode:
        raise ValueError("Planning check failed; receipt remains pending: " + result.stdout + result.stderr)
    receipt["planningCheck"] = json.loads(result.stdout)
    receipt["sourceFingerprints"] = [descriptor(ROOT / name) for name in PLANNING] + [descriptor(AI / name) for name in (
        "docs/LLM_CORPUS_LEAKAGE_PRECISION.v1.md", "tools/review_corpus_leakage.py", "tools/record_corpus_review_progress.py", "tests/test_corpus_leakage_review.py")]
    receipt["status"] = "precision-review-complete-task023-corpus-admission-in-progress"
    write(receipt_path, receipt)
    print(json.dumps({"status": receipt["status"], "testsPassed": len(cases), "realMatchesRequireReview": 2, "quarantinesPreserved": 7, "nextTask": "LLM-TASK-023"}))


if __name__ == "__main__":
    if sys.argv[1:] == ["snapshot"]:
        snapshot()
    elif sys.argv[1:] == ["record"]:
        record()
    else:
        raise SystemExit("Use snapshot or record")
