"""Record task-022 evidence and synchronize planning after required checks pass."""
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
from synthetic_data.foundation_domain import release

REPORTS = AI / "evaluation/reports"
REPORT = REPORTS / "llm-task-022-validation.json"
DOCS = ("VOLTFORGE_LOCAL_LLM_BUILD_BACKLOG.json", "VOLTFORGE_LOCAL_LLM_BUILD_BACKLOG.md", "AI_CHAT_ROOT_CAUSE_BACKLOG.json", "AI_CHAT_ROOT_CAUSE_ANALYSIS.md")


def read(path): return json.loads(path.read_text(encoding="utf-8"))


def write(path, value): path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def descriptor(path): return {"path": path.relative_to(ROOT).as_posix(), "sha256": release.sha(path.read_bytes()), "bytes": path.stat().st_size}


def main():
    if REPORT.exists(): raise ValueError("Completion evidence exists; preserve it")
    built = read(REPORTS / "llm-task-022-domain-build.json")
    fresh = read(REPORTS / "llm-task-022-domain-recompile.json")
    checked = release.verify(Path(built["releasePath"]), recompute=True)
    score = checked["coverage"]
    if fresh["manifestSha256"] != checked["manifestSha256"] or fresh["freshCompileExecutions"] != 24 or fresh["coverage"] != score:
        raise ValueError("Fresh exact-compiler acceptance evidence is missing or changed")
    if score["candidateRecords"] != 67 or score["includedRecords"] != 17 or score["compileExecutions"] != 24 or any(score["leakageAudit"]["counts"].values()):
        raise ValueError("Unexpected domain coverage; review before recording")
    junit = REPORTS / "llm-task-022-tests.xml"
    tree = ET.parse(junit).getroot()
    tests = list(tree.iter("testcase"))
    if len(tests) < 280 or any(list(tree.iter(tag)) for tag in ("failure", "error", "skipped")): raise ValueError("Regression evidence incomplete")
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
        print("Checking " + " ".join(command), flush=True)
        result = subprocess.run([sys.executable, "-B", *command], cwd=AI, capture_output=True, text=True, encoding="utf-8")
        if result.returncode: raise ValueError("Required check failed: " + " ".join(command) + "\n" + result.stdout + result.stderr)
        checks.append({"command": [".toolchains/gen1/Scripts/python.exe", "-B", *command], "exitCode": 0, "result": json.loads(result.stdout)})
    snapshot = ROOT / "backlog_history/llm-task-022-before"
    for row in read(snapshot / "manifest.json")["files"]:
        if release.sha((snapshot / row["backup"]).read_bytes()) != row["sha256"]: raise ValueError("Preserved planning snapshot changed")
    backlog = read(ROOT / DOCS[0])
    task = next(row for row in backlog["tasks"] if row["id"] == "LLM-TASK-022")
    if task["status"] != "TODO" or backlog["execution"]["next_task_id"] != task["id"]: raise ValueError("Unexpected continuation checkpoint")
    date = datetime.now(timezone(timedelta(hours=5, minutes=30))).date().isoformat()
    path = Path(checked["releasePath"])
    manifest_path = (path / "manifest.json").relative_to(ROOT).as_posix()
    receipt_path = REPORT.relative_to(ROOT).as_posix()
    evidence = [manifest_path, (path / "coverage.json").relative_to(ROOT).as_posix(), (path / "calculation-review.jsonl").relative_to(ROOT).as_posix(),
                (path / "compiler-receipts.json").relative_to(ROOT).as_posix(), (path / "leakage.json").relative_to(ROOT).as_posix(),
                "Voltforge_AI/docs/LLM_OWNED_DOMAIN_DATA.v1.md", "backlog_history/llm-task-022-before/manifest.json"] + [
                "Voltforge_AI/evaluation/reports/" + name for name in ("llm-task-022-domain-build.json", "llm-task-022-domain-recompile.json", "llm-task-022-tests.xml", "llm-task-022-checks.json", "llm-task-022-validation.json")]
    summary = ("Added an immutable owned-domain candidate source with 67 records across 27 new families: 51 parameterized calculation examples across 17 families, "
        "12 broken/fixed firmware projects across six recipes and two exact AVR boards, and four exact-entity uncertainty examples. "
        "Independent algorithm review checks units, finite bounds and tolerances; all 153 deliberately corrupted calculation controls are rejected. "
        "All 24 pinned compiler cases pass their expected outcomes (12 fixed accepted, 12 broken rejected), including a fresh recompile. "
        "Pre-render reservations and the unchanged protected matcher yield 17 surviving records/families (12 train, 0 validation, 5 test), with zero audited cross-split/protected collisions. "
        "50 candidates remain quarantined: 15 cross-reservation bridges and 35 duplicates; all 12 firmware candidates are excluded. "
        f"{len(tests)} regression tests passed; governance, runbook and historical-preservation checks passed. "
        "The candidate release retains trainingAllowed=false and domainCorpusReady=false. No model was trained or activated.")
    limits = [
        "This is a domain candidate/data-generation milestone, not a released pretraining corpus. Only 12 records are in the candidate train split; validation is empty.",
        "All firmware records remain quarantined by protected board-name containment/shared question wording and cross-reservation family connections. Do not train on them or reduce thresholds to admit them.",
        "Task 023 must add an explicitly reviewed versioned source/matcher adapter with independent boilerplate-versus-answer/code contamination controls, expand independent families, and populate validation before corpus release.",
        "New candidate source rights are not inherited from task 019 approvals. Task 023 must admit the exact new source bytes and provenance before training use.",
        "Only Uno R3 and Mega 2560 Rev3 were newly compiled. All six recipes currently use an undeclared-function-call defect; broader defect classes and hardware/runtime verification remain follow-up work.",
        "Calculation review is independently implemented algorithmic verification, not human review or measured electrical evidence; hypothetical values never become part ratings.",
        "Protected matching is lexical/structural, not proof against arbitrary semantic paraphrases. Existing indexed-matcher scale, confidential evaluation custody and owner backup/restore gates remain open."]
    task.update(status="COMPLETED", completed_on=date, implementation_state="verified-owned-domain-candidates-corpus-inadequate",
                verification_status="passed-generation-and-compiler-controls-corpus-gates-open", verification_result=summary,
                evidence_records=evidence, remaining_release_gates=limits,
                target_paths=["synthetic_data/foundation_domain/", "tools/build_owned_domain_data.py", "corpus/owned-domain/v1/", "tests/test_owned_domain_data.py", "docs/LLM_OWNED_DOMAIN_DATA.v1.md"])
    backlog["updated_on"] = date
    backlog["scope_of_this_revision"] = "LLM-TASK-015 through 022 completed: foundational controls and immutable owned-domain candidates. Independent corpus breadth, admitted firmware, validation data, pretraining corpus release and model training remain open."
    backlog["execution"]["next_task_id"] = "LLM-TASK-023"
    counts = Counter(row["status"] for row in backlog["tasks"])
    backlog["status_summary"] = {key: counts[key] for key in ("TODO", "IN_PROGRESS", "COMPLETED")}
    backlog["current_checkpoint"] = {"completed_task_id": task["id"], "evidence": receipt_path, "summary": summary, "active_artifact_id": None, "neural_ready": False}
    backlog["owned_domain_candidates"] = {"manifest": manifest_path, "candidate_records": 67, "candidate_families": 27,
        "split_counts": score["splitCounts"], "included_firmware_records": 0, "domain_corpus_ready": False, "gen2_corpus_released": False, "training_allowed": False}
    next_task = next(row for row in backlog["tasks"] if row["id"] == "LLM-TASK-023")
    next_task["domain_candidate_inputs"] = [manifest_path, "Voltforge_AI/docs/LLM_OWNED_DOMAIN_DATA.v1.md"]
    next_task["domain_candidate_handoff"] = "Retain task-022 immutable provenance, reservations and all 50 exclusions. Admit new source rights explicitly; add independent families and a populated validation split. Firmware currently has zero eligible records. Before any revised admission, version the matcher/adapter with independent controls separating factual/protocol boilerplate from real answer/code leakage; do not mutate task-021 thresholds, strip genuine board-specific answers or tailor source text to evade protected guards. Broaden firmware defect classes and obtain runtime evidence separately from compiler acceptance."
    markdown = (ROOT / DOCS[1]).read_text(encoding="utf-8")
    old_count = "**41 active tasks: 7 completed (audit, cleanup, architecture, evaluation, source inventory, ingestion and leakage controls), 34 TODO. No trained conversational LLM release is accepted.**"
    if old_count not in markdown: raise ValueError("Markdown checkpoint count changed")
    markdown = markdown.replace(old_count, "**41 active tasks: 8 completed (audit through owned-domain candidate generation), 33 TODO. No trained conversational LLM release is accepted.**", 1)
    start = markdown.index("**Next: LLM-TASK-022")
    end = markdown.index("\n\n", start)
    markdown = markdown[:start] + "**Next: LLM-TASK-023 — release the general-language/code/domain pretraining corpus.** Candidate data is not yet adequate: 12 train records, no validation records and no eligible firmware.\n\n" + summary + "\n\n[Owned domain candidates](Voltforge_AI/docs/LLM_OWNED_DOMAIN_DATA.v1.md) | [Task 022 evidence](Voltforge_AI/evaluation/reports/llm-task-022-validation.json)" + markdown[end:]
    start = markdown.index("#### LLM-TASK-022 — " + task["title"])
    end = markdown.index("\n#### LLM-TASK-023", start)
    section = markdown[start:end].replace("**Status:** TODO.", "**Status:** COMPLETED.", 1)
    a = section.index("**Target paths:**")
    b = section.index("\n", a)
    section = section[:a] + "**Target paths:** " + ", ".join("`" + name + "`" for name in task["target_paths"]) + "." + section[b:]
    section += "\n**Verified result (" + date + "):** " + summary + "\n\n**Evidence:** [Domain manifest](" + manifest_path + "), [implementation and limits](Voltforge_AI/docs/LLM_OWNED_DOMAIN_DATA.v1.md), [validation](Voltforge_AI/evaluation/reports/llm-task-022-validation.json).\n"
    markdown = markdown[:start] + section + markdown[end:]
    end = markdown.index("\n#### LLM-TASK-024")
    markdown = markdown[:end] + "\n**Task-022 handoff:** " + next_task["domain_candidate_handoff"] + "\n" + markdown[end:]
    causes = read(ROOT / DOCS[2])
    causes["updated_on"] = date
    causes["summary"]["implementation_this_revision"] = backlog["scope_of_this_revision"]
    causes["summary"]["acceptance_limit"] = summary + " Root causes remain open/partial until all linked remediation tasks pass."
    for cause in causes["root_causes"]:
        if task["id"] in cause["remediation_task_ids"]:
            cause["owned_domain_progress"] = {"task_id": task["id"], "evidence": receipt_path, "result": "Owned candidates and exact compiler/calculation receipts are verified; admitted firmware, validation breadth, pretraining corpus and model quality remain open."}
    causes["evidence_files"].extend(name for name in evidence if name not in causes["evidence_files"])
    causes["current_checkpoint"] = {"completed_task_id": task["id"], "evidence": receipt_path, "next_task_id": "LLM-TASK-023"}
    analysis = (ROOT / DOCS[3]).read_text(encoding="utf-8")
    analysis = analysis.replace("## Current checkpoint: LLM-TASK-021 completed", "## Current checkpoint: LLM-TASK-022 completed\n\n" + summary + "\n\n[Task 022 domain data and limits](Voltforge_AI/docs/LLM_OWNED_DOMAIN_DATA.v1.md) | [Verification](Voltforge_AI/evaluation/reports/llm-task-022-validation.json)\n\n## Previous checkpoint: LLM-TASK-021 completed", 1)
    analysis = analysis.replace("**LLM-TASK-022 is next: independent owned domain data with pre-render partitions and compiler/calculation evidence.**", "**LLM-TASK-023 is next: approved broad corpus release, independent validation data and remaining domain-admission gates.**")
    release.write_immutable(REPORTS / "llm-task-022-checks.json", release.json_bytes({"status": "passed", "domainRecomputed": checked, "checks": checks}))
    receipt = {"schemaVersion": 1, "taskId": task["id"], "status": "verified-planning-check-pending", "recordedAtUtc": datetime.now(timezone.utc).isoformat(),
        "summary": summary, "domainVerification": checked, "freshCompilerEvidence": descriptor(REPORTS / "llm-task-022-domain-recompile.json"),
        "tests": {"passed": len(tests), "failed": 0, "skipped": 0, "junit": descriptor(junit)}, "limits": limits,
        "preservation": {"planningFilesSnapshotted": 7, "historicalContractBindingsUnchanged": 12, "protectedArtifactDataFilesUnchanged": 131,
                         "task018SuiteUnchanged": True, "task019InventoryUnchanged": True, "task020And021RecomputedUnchanged": True},
        "gen2CorpusReleased": False, "trainingAllowed": False, "domainCorpusReady": False, "modelReleaseApproved": False,
        "networkAccessDuringBuild": False, "documentationReferencesBrowsed": True, "thirdPartySourceBodiesImported": False, "nextTask": "LLM-TASK-023"}
    release.write_immutable(REPORT, release.json_bytes(receipt))
    write(ROOT / DOCS[0], backlog)
    (ROOT / DOCS[1]).write_text(markdown, encoding="utf-8", newline="\n")
    write(ROOT / DOCS[2], causes)
    (ROOT / DOCS[3]).write_text(analysis, encoding="utf-8", newline="\n")
    result = subprocess.run([sys.executable, "-B", "tools/verify_llm_backlogs.py"], cwd=AI, capture_output=True, text=True, encoding="utf-8")
    if result.returncode: raise ValueError("Planning consistency failed; receipt remains pending: " + result.stdout + result.stderr)
    receipt["planningCheck"] = json.loads(result.stdout)
    receipt["sourceFingerprints"] = [descriptor(ROOT / name) for name in DOCS] + [descriptor(AI / name) for name in (
        "docs/LLM_OWNED_DOMAIN_DATA.v1.md", "docs/VOLTForge_AI_RUNBOOK.md", "docs/runbook-contract.v1.json", ".gitattributes", "tests/test_owned_domain_data.py", "tools/build_owned_domain_data.py", "tools/record_owned_domain_data.py")]
    receipt["evidence"] = [descriptor(ROOT / name) for name in evidence if ROOT / name != REPORT]
    receipt["status"] = "completed-owned-domain-candidates-corpus-inadequate"
    write(REPORT, receipt)
    print(json.dumps({"status": receipt["status"], "testsPassed": len(tests), "candidateRecords": 67, "includedRecords": 17, "nextTask": receipt["nextTask"]}))


if __name__ == "__main__": main()
