"""Record task-021 acceptance without confusing empty safe partitions with data readiness."""
from collections import Counter
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

AI = Path(__file__).resolve().parents[1]
ROOT = AI.parent
sys.path.insert(0, str(AI))
from data_governance.splitting import release
from data_governance.ingestion import pipeline as p

TASK = "LLM-TASK-021"
REPORTS = AI / "evaluation/reports"
REPORT = REPORTS / "llm-task-021-validation.json"
DOCS = ("VOLTFORGE_LOCAL_LLM_BUILD_BACKLOG.json", "VOLTFORGE_LOCAL_LLM_BUILD_BACKLOG.md", "AI_CHAT_ROOT_CAUSE_BACKLOG.json", "AI_CHAT_ROOT_CAUSE_ANALYSIS.md")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def descriptor(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": p.sources.file_hash(path), "bytes": path.stat().st_size}


def main():
    if REPORT.exists():
        raise ValueError("Completion evidence exists; preserve it")
    build = p.read_json(REPORTS / "llm-task-021-partition-build-v2.json")
    partition_path = Path(build["releasePath"])
    checked = release.verify(partition_path, recompute=True)
    score = checked["scorecard"]
    if score["splitCounts"] != {"train": 0, "validation": 0, "test": 0, "quarantine": 227} or score["trainingPoolAvailable"] is not False:
        raise ValueError("Unexpected current-corpus outcome; review before recording")
    fixtures = p.read_json(REPORTS / "llm-task-021-leakage-fixtures-v3.json")
    if fixtures["status"] != "passed" or fixtures["leakedFixturesRejected"] != 7 or not fixtures["unrelatedControlAccepted"] or not fixtures["deliberateCrossSplitCollisionRejected"]:
        raise ValueError("Required positive/negative leakage controls failed")
    if set(fixtures["nonemptyPartitionControl"]["splitCounts"]) != {"train", "validation", "test"}:
        raise ValueError("Nonempty split evidence is missing")
    for row in fixtures["sourceFingerprints"]:
        if p.sources.file_hash(AI / row["path"]) != row["sha256"]:
            raise ValueError("Fixture receipt uses different implementation bytes")
    junit = REPORTS / "llm-task-021-tests-v2.xml"
    tree = ET.parse(junit).getroot()
    cases = list(tree.iter("testcase"))
    suites = list(tree.iter("testsuite"))
    if len(cases) < 170 or not suites or any(list(tree.iter(tag)) for tag in ("failure", "error", "skipped")):
        raise ValueError("Final regression evidence incomplete")
    subtests = sum(int(row.get("tests", 0)) for row in suites) - len(cases)
    checks = []
    for command in (
        ["tools/inventory_llm_sources.py", "verify", "--recompute"],
        ["tools/verify_llm_foundation.py", "--check-preserved-history"],
        ["tools/verify_foundation_evaluation_binding.py"],
        ["tools/audit_data_governance.py", "--check"],
        ["tools/verify_runbook.py"],
        ["tools/scan_quality_boundaries.py", "--check"],
        ["tools/snapshot_llm_cleanup.py", "--verify", "../backlog_history/llm-task-016-before-20260906T054449Z"],
    ):
        result = subprocess.run([sys.executable, "-B", *command], cwd=AI, capture_output=True, text=True, encoding="utf-8")
        if result.returncode:
            raise ValueError("Required check failed: " + " ".join(command) + "\n" + result.stdout + result.stderr)
        checks.append({"command": [".toolchains/gen1/Scripts/python.exe", "-B", *command], "exitCode": 0, "result": json.loads(result.stdout)})
    for folder in ("llm-task-021-before", "llm-task-021-draft-implementation"):
        snapshot = ROOT / "backlog_history" / folder
        for row in p.read_json(snapshot / "manifest.json")["files"]:
            if p.sources.file_hash(snapshot / row.get("backup", row["path"])) != row["sha256"]:
                raise ValueError("Preserved snapshot changed")
    backlog = p.read_json(ROOT / DOCS[0])
    task = next(row for row in backlog["tasks"] if row["id"] == TASK)
    if task["status"] != "TODO" or backlog["execution"]["next_task_id"] != TASK:
        raise ValueError("Unexpected continuation checkpoint")
    date = datetime.now(timezone(timedelta(hours=5, minutes=30))).date().isoformat()
    manifest_path = (partition_path / "manifest.json").relative_to(ROOT).as_posix()
    score_path = (partition_path / "scorecard.json").relative_to(ROOT).as_posix()
    receipt_path = REPORT.relative_to(ROOT).as_posix()
    evidence = [manifest_path, score_path, "Voltforge_AI/docs/LLM_CORPUS_PARTITIONS.v1.md"] + [
        "Voltforge_AI/evaluation/reports/" + name for name in (
            "llm-task-021-partition-build-v2.json", "llm-task-021-leakage-fixtures-v3.json", "llm-task-021-tests-v2.xml",
            "llm-task-021-checks.json", "llm-task-021-validation.json")]
    evidence.extend(["backlog_history/llm-task-021-before/manifest.json", "backlog_history/llm-task-021-draft-implementation/manifest.json"])
    summary = ("Implemented immutable source/document/recipe/circuit family partitions, pre-render reservations, exact/lexical/code duplicate checks, "
        "protected evaluation guards and separate unique-token/exposure accounting. All 227 task-020 staging records are quarantined by this additional "
        "content/family policy: 114 exact protected-content matches and 113 direct protected-family matches in two connected components. "
        "Current train/validation/test partitions and eligible training tokens are zero; this is rejection evidence, not corpus adequacy. "
        "Seven deliberate leakage fixtures are rejected, unrelated controls pass, and a separate 100-family control produces 80/10/10 populated partitions "
        "with zero cross-split collisions. Final manifests reproduce from unchanged source and protected bytes. "
        + f"{len(cases)} regression tests passed with {subtests} additional passing subtests; governance, evaluation, runbook and preservation checks passed. "
        "No corpus acquisition, new source approval, tokenizer fitting, model training/activation or serving change.")
    limits = ["The current approved staging pool yields zero independent training records under the new policy. Task 022 must author new independent domain/problem families before variant rendering.",
              "Near matching is lexical and structural, not a guarantee against arbitrary semantic paraphrases. Source-specific review remains necessary.",
              "Task 023 must provide a reviewed indexed matcher with recall/resource evidence beyond the 5,000-document/2,000,000-comparison pilot bounds; checks fail closed at those limits.",
              "Pre-render reservations bind lineage and split assignments but do not grant source approval; expanded outputs must still pass post-render contamination checks.",
              "All partition artifacts retain trainingAllowed=false. Task 023 owns corpus release and task 024 owns the Gen2 tokenizer/token count.",
              "Unknown historical v1.0.0 text, separate confidential evaluation custody and owner backup/restore gates remain unresolved and unchanged."]
    task.update(status="COMPLETED", completed_on=date, implementation_state="verified-family-partitions-and-leakage-controls",
                verification_status="passed-controls-current-corpus-inadequate", verification_result=summary,
                evidence_records=evidence, remaining_release_gates=limits,
                target_paths=["data_governance/splitting/", "tools/partition_llm_corpus.py", "tools/evaluate_corpus_splits.py",
                              "corpus/partitions/v1/", "tests/test_corpus_splitting.py", "docs/LLM_CORPUS_PARTITIONS.v1.md"])
    backlog["updated_on"] = date
    backlog["scope_of_this_revision"] = "LLM-TASK-015 through 021 completed: audit, cleanup, frozen design/evaluation, source inventory, ingestion and family/leakage controls. The existing pool yields zero independent training records; new data, corpus release, model training and product acceptance remain open."
    backlog["execution"]["next_task_id"] = "LLM-TASK-022"
    counts = Counter(row["status"] for row in backlog["tasks"])
    backlog["status_summary"] = {key: counts[key] for key in ("TODO", "IN_PROGRESS", "COMPLETED")}
    backlog["current_checkpoint"] = {"completed_task_id": TASK, "evidence": receipt_path, "summary": summary, "active_artifact_id": None, "neural_ready": False}
    backlog["corpus_partitions"] = {"manifest": manifest_path, "scorecard": score_path, "input_staging_documents": 227,
        "split_counts": score["splitCounts"], "post_partition_training_proxy_tokens": 0, "independent_training_pool_available": False,
        "gen2_corpus_released": False, "training_allowed": False}
    task22 = next(row for row in backlog["tasks"] if row["id"] == "LLM-TASK-022")
    task22["input_artifacts"] = [manifest_path, "Voltforge_AI/data_governance/splitting/policy.v1.json"]
    task22["partition_handoff"] = "Create new independent recipes/circuit/problem families with explicit ancestry and pre-render reservations. Retain exact compiler/calculation receipts and post-render leakage checks. Add versioned source/ingestion adapters; do not mutate frozen tasks 019/020/021 or merely relabel their protected board/template variants."
    task23 = next(row for row in backlog["tasks"] if row["id"] == "LLM-TASK-023")
    task23["partition_scale_gate"] = "The current matcher fails closed above 5,000 descriptors or 2,000,000 record/guard comparisons. Before a larger corpus release, implement a versioned indexed scan and measure recall against exhaustive/adversarial controls plus throughput/memory; retain source/family grouping and content-free receipts."
    markdown = (ROOT / DOCS[1]).read_text(encoding="utf-8")
    markdown = markdown.replace("**41 active tasks: 6 completed (audit, cleanup, architecture, evaluation, source inventory and ingestion), 35 TODO. No trained conversational LLM release is accepted.**",
        "**41 active tasks: 7 completed (audit, cleanup, architecture, evaluation, source inventory, ingestion and leakage controls), 34 TODO. No trained conversational LLM release is accepted.**")
    start = markdown.index("**Next: LLM-TASK-021")
    end = markdown.index("\n\n", start)
    markdown = markdown[:start] + "**Next: LLM-TASK-022 — expand owned electronics and compiler-verified domain data.** The current pool yields zero independent training records; author new families with reserved partitions before rendering variants.\n\n" + summary + "\n\n[Partition implementation and corpus finding](Voltforge_AI/docs/LLM_CORPUS_PARTITIONS.v1.md) | [Task 021 evidence](Voltforge_AI/evaluation/reports/llm-task-021-validation.json)" + markdown[end:]
    start = markdown.index("#### LLM-TASK-021 — " + task["title"])
    end = markdown.index("\n#### LLM-TASK-022", start)
    section = markdown[start:end].replace("**Status:** TODO.", "**Status:** COMPLETED.", 1)
    a = section.index("**Target paths:**")
    b = section.index("\n", a)
    section = section[:a] + "**Target paths:** " + ", ".join("`" + name + "`" for name in task["target_paths"]) + "." + section[b:]
    section += f"**Verified result ({date}):** " + summary + "\n\n**Evidence:** [Partition manifest](" + manifest_path + "), [scorecard](" + score_path + "), [adversarial controls](Voltforge_AI/evaluation/reports/llm-task-021-leakage-fixtures-v3.json), [validation](Voltforge_AI/evaluation/reports/llm-task-021-validation.json).\n"
    markdown = markdown[:start] + section + markdown[end:]
    start = markdown.index("#### LLM-TASK-022 —")
    end = markdown.index("\n#### LLM-TASK-023", start)
    markdown = markdown[:end] + "\n**Task-021 handoff:** " + task22["partition_handoff"] + "\n" + markdown[end:]
    end = markdown.index("\n#### LLM-TASK-024")
    markdown = markdown[:end] + "\n**Partition scale gate:** " + task23["partition_scale_gate"] + "\n" + markdown[end:]
    causes = p.read_json(ROOT / DOCS[2])
    causes["updated_on"] = date
    causes["summary"]["implementation_this_revision"] = backlog["scope_of_this_revision"]
    causes["summary"]["acceptance_limit"] = summary + " Root causes remain open/partial until their remaining linked tasks pass."
    for cause in causes["root_causes"]:
        if TASK in cause["remediation_task_ids"]:
            cause["partition_leakage_progress"] = {"task_id": TASK, "evidence": receipt_path,
                "result": "Verified family/duplicate/protected controls; all 227 current staging documents are excluded from new training partitions. Independent data expansion and actual model training remain open."}
    causes["evidence_files"].extend(path for path in evidence if path not in causes["evidence_files"])
    causes["current_checkpoint"] = {"completed_task_id": TASK, "evidence": receipt_path, "next_task_id": "LLM-TASK-022"}
    analysis = (ROOT / DOCS[3]).read_text(encoding="utf-8")
    analysis = analysis.replace("## Current checkpoint: LLM-TASK-020 completed", "## Current checkpoint: LLM-TASK-021 completed\n\n" + summary +
        "\n\n[Partition findings](Voltforge_AI/docs/LLM_CORPUS_PARTITIONS.v1.md) | [Task 021 verification](Voltforge_AI/evaluation/reports/llm-task-021-validation.json)\n\n## Previous checkpoint: LLM-TASK-020 completed", 1)
    analysis = analysis.replace("**LLM-TASK-021 is next: leakage prevention and near-duplicate/family split controls.**", "**LLM-TASK-022 is next: independent owned domain data with pre-render partitions and compiler/calculation evidence.**")
    p.exclusive_json(REPORTS / "llm-task-021-checks.json", {"status": "passed", "partitionRecomputed": checked, "checks": checks})
    receipt = {"schemaVersion": 1, "taskId": TASK, "status": "partition-verified-planning-check-pending",
        "recordedAtUtc": datetime.now(timezone.utc).isoformat(), "summary": summary, "partitionVerification": checked,
        "tests": {"passed": len(cases), "failed": 0, "skipped": 0, "additionalPassingSubtests": subtests, "junit": descriptor(junit)},
        "fixtureEvidence": descriptor(REPORTS / "llm-task-021-leakage-fixtures-v3.json"), "limits": limits,
        "preservation": {"planningFilesSnapshotted": 7, "historicalContractBindingsUnchanged": 12, "protectedArtifactDataFilesUnchanged": 131,
                         "task018SuiteAndPolicyUnchanged": True, "task019InventoryUnchanged": True, "task020IngestionRecomputedUnchanged": True,
                         "preacceptanceCandidateAndImplementationRetained": True},
        "gen2CorpusReleased": False, "trainingAllowed": False, "modelReleaseApproved": False,
        "currentTrainingPoolAvailable": False, "networkAccessed": False, "nextTask": "LLM-TASK-022"}
    p.exclusive_json(REPORT, receipt)
    write_json(ROOT / DOCS[0], backlog)
    (ROOT / DOCS[1]).write_text(markdown, encoding="utf-8", newline="\n")
    write_json(ROOT / DOCS[2], causes)
    (ROOT / DOCS[3]).write_text(analysis, encoding="utf-8", newline="\n")
    result = subprocess.run([sys.executable, "-B", "tools/verify_llm_backlogs.py"], cwd=AI, capture_output=True, text=True, encoding="utf-8")
    if result.returncode:
        raise ValueError("Planning check failed; receipt remains pending: " + result.stdout + result.stderr)
    receipt["planningCheck"] = json.loads(result.stdout)
    paths = [ROOT / name for name in DOCS] + [AI / name for name in (
        "docs/LLM_CORPUS_PARTITIONS.v1.md", "docs/VOLTForge_AI_RUNBOOK.md", "docs/runbook-contract.v1.json", ".gitattributes",
        "tests/test_corpus_splitting.py", "tools/partition_llm_corpus.py", "tools/evaluate_corpus_splits.py", "tools/record_corpus_partitions.py")]
    receipt["sourceFingerprints"] = [descriptor(path) for path in paths]
    receipt["evidence"] = [descriptor(ROOT / path) for path in evidence if ROOT / path != REPORT]
    receipt["status"] = "completed-partition-controls-current-corpus-inadequate"
    write_json(REPORT, receipt)
    print(json.dumps({"status": receipt["status"], "testsPassed": len(cases), "passingSubtests": subtests,
                      "currentTrainingRecords": 0, "quarantinedRecords": 227, "nextTask": receipt["nextTask"]}))


if __name__ == "__main__":
    main()
