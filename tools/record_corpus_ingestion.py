"""Close task 020 only after immutable reproduction, review and regression evidence."""
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

AI = Path(__file__).resolve().parents[1]
ROOT = AI.parent
sys.path.insert(0, str(AI))
from data_governance.ingestion import pipeline as p

TASK = "LLM-TASK-020"
REPORT = AI / "evaluation/reports/llm-task-020-validation.json"
DOCS = ("VOLTFORGE_LOCAL_LLM_BUILD_BACKLOG.json", "VOLTFORGE_LOCAL_LLM_BUILD_BACKLOG.md", "AI_CHAT_ROOT_CAUSE_BACKLOG.json", "AI_CHAT_ROOT_CAUSE_ANALYSIS.md")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def descriptor(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": p.sources.file_hash(path), "bytes": path.stat().st_size}


def main():
    if REPORT.exists():
        raise ValueError("Task completion receipt exists; preserve it")
    reports = AI / "evaluation/reports"
    source_build = p.read_json(reports / "llm-task-020-ingestion-build.json")
    fixture_build = p.read_json(reports / "llm-task-020-fixture-build.json")
    source_release, fixture_release = Path(source_build["releasePath"]), Path(fixture_build["releasePath"])
    verified = {"source": p.verify(source_release, recompute=True), "fixtures": p.verify(fixture_release, recompute=True)}
    review = p.read_json(reports / "llm-task-020-extraction-review.json")
    memory = p.read_json(reports / "llm-task-020-memory.json")
    if review["status"] != "passed" or memory["status"] != "passed" or not all(memory["checks"].values()):
        raise ValueError("Extraction review or resource measurement did not pass")
    if review["sourceManifestSha256"] != verified["source"]["manifestSha256"] or review["fixtureManifestSha256"] != verified["fixtures"]["manifestSha256"]:
        raise ValueError("Extraction review binds different release bytes")
    junit_path = reports / "llm-task-020-tests.xml"
    tree = ET.parse(junit_path).getroot()
    cases = list(tree.iter("testcase"))
    suites = list(tree.iter("testsuite"))
    if len(cases) < 120 or not suites or any(list(tree.iter(tag)) for tag in ("failure", "error", "skipped")):
        raise ValueError("Required regression selection is not complete and passing")
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
    snapshot = ROOT / "backlog_history/llm-task-020-before"
    for row in p.read_json(snapshot / "manifest.json")["files"]:
        if p.sources.file_hash(snapshot / row["backup"]) != row["sha256"]:
            raise ValueError("Pre-edit planning snapshot changed")
    backlog = p.read_json(ROOT / DOCS[0])
    task = next(row for row in backlog["tasks"] if row["id"] == TASK)
    if task["status"] != "TODO" or backlog["execution"]["next_task_id"] != TASK:
        raise ValueError("Unexpected task-020 continuation checkpoint")
    date = datetime.now(timezone(timedelta(hours=5, minutes=30))).date().isoformat()
    source_manifest = (source_release / "manifest.json").relative_to(ROOT).as_posix()
    fixture_manifest = (fixture_release / "manifest.json").relative_to(ROOT).as_posix()
    receipt_path = REPORT.relative_to(ROOT).as_posix()
    evidence = [source_manifest, fixture_manifest, "Voltforge_AI/docs/LLM_CORPUS_INGESTION.v1.md"] + [
        "Voltforge_AI/evaluation/reports/" + name for name in (
            "llm-task-020-ingestion-build.json", "llm-task-020-fixture-build.json", "llm-task-020-extraction-review.json",
            "llm-task-020-memory.json", "llm-task-020-tests.xml", "llm-task-020-checks.json", "llm-task-020-validation.json")]
    evidence.append("backlog_history/llm-task-020-before/manifest.json")
    summary = ("Implemented offline resumable corpus ingestion with exact source/shard permission checks, document lineage, "
        "transactional resume, strict encoding and code/equation/SI-preserving extraction, secret/broken-text quarantine and exact normalized deduplication. "
        "Four current approved shards produce 227 immutable raw/normalized records; 23 held-out records are excluded before raw persistence. "
        "All 17 authored extraction fixtures have reviewed decisions (8 accepted, 9 quarantined), and eight actual source samples preserve their nested payloads. "
        "Both manifests reproduce byte-for-byte. Fresh-process 1,000/10,000-document measurements maintain about 2.1 MB of traced Python allocation "
        "and about 88 documents/second. " + f"{len(cases)} regression tests passed, plus {subtests} passing subtests; source/evaluation/runbook/preservation checks passed. "
        "No external corpus acquisition, new source rights, Gen2 corpus release, tokenizer fitting, training, activation or serving change.")
    limits = ["Task 021 must add semantic/near-duplicate checks, source/document/family partitions and cross-split leakage evidence before corpus release.",
              "The existing 227 records remain small structured synthetic data; fixture and staging copies add no new independent token credit.",
              "External source revisions and permissions remain unadmitted. A new approved inventory/adapter is required before acquisition can become a production input.",
              "PDF extraction review covers authored digital-text fixtures; complex layouts/OCR require source-specific review, and arbitrary external PDFs need aggregate worker resource controls.",
              "Secret patterns are heuristic; private-source exclusion and source-use/privacy review remain mandatory.",
              "Throughput/memory measurements are local synthetic JSONL observations, not server capacity or PDF-child RSS proof.",
              "No Gen2 corpus/training approval. Historical missing bytes, confidential evaluation custody and owner backup/restore gates remain unchanged."]
    task.update(status="COMPLETED", completed_on=date, implementation_state="verified-immutable-offline-ingestion",
                verification_status="passed-ingestion-no-training-corpus-release", verification_result=summary,
                evidence_records=evidence, remaining_release_gates=limits,
                target_paths=["data_governance/ingestion/", "tools/ingest_llm_corpus.py", "corpus/ingestion/v1/",
                              "tests/test_corpus_ingestion.py", "docs/LLM_CORPUS_INGESTION.v1.md"])
    backlog["updated_on"] = date
    backlog["scope_of_this_revision"] = "LLM-TASK-015 through 020 completed: audit, cleanup, frozen design/evaluation, source inventory and immutable ingestion. Corpus splits/release, model implementation/training and product acceptance remain open."
    backlog["execution"]["next_task_id"] = "LLM-TASK-021"
    counts = Counter(row["status"] for row in backlog["tasks"])
    backlog["status_summary"] = {key: counts[key] for key in ("TODO", "IN_PROGRESS", "COMPLETED")}
    backlog["current_checkpoint"] = {"completed_task_id": TASK, "evidence": receipt_path, "summary": summary, "active_artifact_id": None, "neural_ready": False}
    backlog["corpus_ingestion"] = {"source_manifest": source_manifest, "fixture_manifest": fixture_manifest,
        "accepted_source_documents": 227, "excluded_held_out_documents": 23, "reviewed_source_samples": 8,
        "reviewed_fixture_decisions": 17, "gen2_corpus_released": False, "training_allowed": False}
    next_task = next(row for row in backlog["tasks"] if row["id"] == "LLM-TASK-021")
    next_task["input_artifacts"] = [source_manifest, "Voltforge_AI/evaluation/foundation/manifest.v1.json"]
    next_task["ingestion_handoff"] = "Use immutable document/source/raw/normalized lineage; exclude extraction fixtures and retain historical held-out exclusions. Do not rewrite task-019 inventory or prior releases."
    markdown = (ROOT / DOCS[1]).read_text(encoding="utf-8")
    markdown = markdown.replace("**41 active tasks: 5 completed (audit, cleanup, architecture, evaluation and source inventory), 36 TODO. No trained conversational LLM release is accepted.**",
        "**41 active tasks: 6 completed (audit, cleanup, architecture, evaluation, source inventory and ingestion), 35 TODO. No trained conversational LLM release is accepted.**")
    start = markdown.index("**Next: LLM-TASK-020")
    end = markdown.index("\n\n", start)
    markdown = markdown[:start] + "**Next: LLM-TASK-021 — prevent train/evaluation leakage and near-duplicate inflation.** Build reproducible source/document/family partitions over the immutable ingestion output.\n\n" + summary + "\n\n[Ingestion implementation and limits](Voltforge_AI/docs/LLM_CORPUS_INGESTION.v1.md) | [Task 020 evidence](Voltforge_AI/evaluation/reports/llm-task-020-validation.json)" + markdown[end:]
    start = markdown.index("#### LLM-TASK-020 — " + task["title"])
    end = markdown.index("\n#### LLM-TASK-021", start)
    section = markdown[start:end].replace("**Status:** TODO.", "**Status:** COMPLETED.", 1)
    a = section.index("**Target paths:**")
    b = section.index("\n", a)
    section = section[:a] + "**Target paths:** " + ", ".join("`" + name + "`" for name in task["target_paths"]) + "." + section[b:]
    section += f"**Verified result ({date}):** " + summary + "\n\n**Evidence:** [Ingestion manifest](" + source_manifest + "), [extraction review](Voltforge_AI/evaluation/reports/llm-task-020-extraction-review.json), [memory/throughput](Voltforge_AI/evaluation/reports/llm-task-020-memory.json), [validation](Voltforge_AI/evaluation/reports/llm-task-020-validation.json).\n"
    markdown = markdown[:start] + section + markdown[end:]
    causes = p.read_json(ROOT / DOCS[2])
    causes["updated_on"] = date
    causes["summary"]["implementation_this_revision"] = backlog["scope_of_this_revision"]
    causes["summary"]["acceptance_limit"] = summary + " Root causes remain open/partial until their remaining linked implementation and acceptance tasks pass."
    for cause in causes["root_causes"]:
        if TASK in cause["remediation_task_ids"]:
            cause["corpus_ingestion_progress"] = {"task_id": TASK, "evidence": receipt_path,
                "result": "Immutable offline ingestion, source lineage, extraction review and resource evidence verified. Semantic leakage/family splits, broad corpus release and actual training remain open."}
    causes["evidence_files"].extend(path for path in evidence if path not in causes["evidence_files"])
    causes["current_checkpoint"] = {"completed_task_id": TASK, "evidence": receipt_path, "next_task_id": "LLM-TASK-021"}
    analysis = (ROOT / DOCS[3]).read_text(encoding="utf-8")
    analysis = analysis.replace("## Current checkpoint: LLM-TASK-019 completed", "## Current checkpoint: LLM-TASK-020 completed\n\n" + summary +
        "\n\n[Corpus ingestion](Voltforge_AI/docs/LLM_CORPUS_INGESTION.v1.md) | [Task 020 verification](Voltforge_AI/evaluation/reports/llm-task-020-validation.json)\n\n## Previous checkpoint: LLM-TASK-019 completed", 1)
    analysis = analysis.replace("**LLM-TASK-020 is next: reproducible corpus ingestion and normalization.**", "**LLM-TASK-021 is next: leakage prevention and near-duplicate/family split controls.**")
    p.exclusive_json(reports / "llm-task-020-checks.json", {"status": "passed", "ingestionRecomputed": verified, "checks": checks})
    receipt = {"schemaVersion": 1, "taskId": TASK, "status": "ingestion-verified-planning-check-pending",
               "recordedAtUtc": datetime.now(timezone.utc).isoformat(), "summary": summary, "verification": verified,
               "tests": {"passed": len(cases), "failed": 0, "skipped": 0, "additionalPassingSubtests": subtests, "junit": descriptor(junit_path)},
               "extractionReview": descriptor(reports / "llm-task-020-extraction-review.json"),
               "performance": descriptor(reports / "llm-task-020-memory.json"), "limits": limits,
               "preservation": {"planningFilesSnapshotted": 8, "historicalContractBindingsUnchanged": 12, "protectedArtifactDataFilesUnchanged": 131,
                                "task018SuiteAndPolicyUnchanged": True, "task019InventoryRecomputedUnchanged": True},
               "network": {"ingestionNetworkAccessed": False, "externalCorpusAcquired": False,
                           "separateBuildDependencyInstalled": "pypdf==6.17.0", "runtimeFrameworkChanged": False},
               "gen2CorpusReleased": False, "trainingAllowed": False, "modelReleaseApproved": False, "nextTask": "LLM-TASK-021"}
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
        "docs/LLM_CORPUS_INGESTION.v1.md", "docs/VOLTForge_AI_RUNBOOK.md", "docs/runbook-contract.v1.json", ".gitattributes", ".gitignore",
        "requirements-corpus.txt", "tests/test_corpus_ingestion.py", "tools/ingest_llm_corpus.py", "tools/build_ingestion_fixtures.py",
        "tools/review_ingestion_samples.py", "tools/evaluate_ingestion_memory.py", "tools/record_corpus_ingestion.py")]
    receipt["sourceFingerprints"] = [descriptor(path) for path in paths]
    receipt["evidence"] = [descriptor(ROOT / path) for path in evidence if ROOT / path != REPORT]
    receipt["status"] = "completed-immutable-ingestion-no-training-corpus-release"
    write_json(REPORT, receipt)
    print(json.dumps({"status": receipt["status"], "testsPassed": len(cases), "passingSubtests": subtests,
                      "acceptedSourceDocuments": 227, "excludedHeldOutDocuments": 23, "nextTask": receipt["nextTask"]}))


if __name__ == "__main__":
    main()
