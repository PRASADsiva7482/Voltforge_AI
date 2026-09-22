"""Record task-019 evidence and its dependency-ready continuation checkpoint once."""
from __future__ import annotations

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
from data_governance.foundation.inventory import file_hash, read, verify

REPORT = AI / "evaluation/reports/llm-task-019-validation.json"
SNAPSHOT = ROOT / "backlog_history/llm-task-019-before"
DOCS = ("VOLTFORGE_LOCAL_LLM_BUILD_BACKLOG.json", "VOLTFORGE_LOCAL_LLM_BUILD_BACKLOG.md", "AI_CHAT_ROOT_CAUSE_BACKLOG.json", "AI_CHAT_ROOT_CAUSE_ANALYSIS.md")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def root_descriptor(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": file_hash(path), "bytes": path.stat().st_size}


def main():
    if REPORT.exists() or (SNAPSHOT / "manifest.json").exists():
        raise ValueError("Completion evidence exists; preserve the prior version")
    verified = verify(recompute=True)
    inventory = read(AI / "data_governance/foundation/inventory.v1.json")
    junit_path = AI / "evaluation/reports/llm-task-019-tests.xml"
    tree = ET.parse(junit_path).getroot()
    suites = list(tree.iter("testsuite"))
    cases = list(tree.iter("testcase"))
    skipped = len(list(tree.iter("skipped")))
    if not suites or len(cases) < 80 or list(tree.iter("failure")) or list(tree.iter("error")) or skipped:
        raise ValueError("Final source/governance regression selection is not complete and passing")
    subtests = sum(int(row.get("tests", "0")) for row in suites) - len(cases)
    checks = []
    for command in (
        ["tools/verify_llm_foundation.py", "--check-preserved-history"],
        ["tools/verify_foundation_evaluation_binding.py"],
        ["tools/audit_data_governance.py", "--check"],
        ["tools/verify_runbook.py"],
        ["tools/scan_quality_boundaries.py", "--check"],
        ["tools/snapshot_llm_cleanup.py", "--verify", "../backlog_history/llm-task-016-before-20260906T054449Z"],
    ):
        completed = subprocess.run([sys.executable, "-B", *command], cwd=AI, capture_output=True, text=True, encoding="utf-8")
        if completed.returncode:
            raise ValueError("Required check failed: " + " ".join(command) + "\n" + completed.stdout + completed.stderr)
        checks.append({"command": [".toolchains/gen1/Scripts/python.exe", "-B", *command], "exitCode": 0, "result": json.loads(completed.stdout)})
    backlog_path = ROOT / DOCS[0]
    backlog = read(backlog_path)
    task = next(row for row in backlog["tasks"] if row["id"] == "LLM-TASK-019")
    if task["status"] != "TODO" or backlog["execution"]["next_task_id"] != task["id"]:
        raise ValueError("Unexpected source-inventory continuation checkpoint")
    date = datetime.now(timezone(timedelta(hours=5, minutes=30))).date().isoformat()
    evidence = ["Voltforge_AI/" + value for value in (
        "data_governance/foundation/policy.v1.json", "data_governance/foundation/proposed-sources.v1.json",
        "data_governance/foundation/external-evidence.v1.json", "data_governance/foundation/inventory.v1.json",
        "data_governance/foundation/manifest.v1.json", "docs/LLM_TRAINING_SOURCE_INVENTORY.v1.md",
        "evaluation/reports/llm-task-019-inventory-freeze.json", "evaluation/reports/llm-task-019-tests.xml",
        "evaluation/reports/llm-task-019-checks.json", "evaluation/reports/llm-task-019-validation.json")]
    evidence.append("backlog_history/llm-task-019-before/manifest.json")
    summary = ("Source inventory verified: 8 existing definitions, 4 approved for existing training-source use; "
        "4 approved synthetic shards with 250 records, 8 curated upstream packs with 61 facts, and 12 quarantined legacy datasets with 17,814 records. "
        "After excluding 23 held-out records, 227 distinct synthetic payloads measure 112,418 tokens with the existing owned tokenizer as a Gen2 planning proxy. "
        "Eight retained shard copies and upstream facts add no independent token credit. Seven external sources remain unadmitted proposals; "
        "private chats/projects, secrets, teacher outputs and evaluation material remain excluded. "
        f"{len(cases)} regression tests passed; offline recomputation, source/evaluation integrity, runbook and preservation checks passed. "
        "The pilot/base token gaps are quantified, with zero Gen2 release-ready tokens. No corpus acquisition, new source permission, training, model activation or deployment.")
    limits = ["External content revisions, file-level rights and any publisher permissions remain unacquired/unapproved; OpenStax explicitly requires LLM permission.",
              "The existing pool is small structured synthetic data. Source permission does not establish a broad language/code corpus or a Gen2 release.",
              "Tasks 020/021 must enforce document-to-source lineage, normalization, semantic deduplication and family-level splits; confidential evaluation custody is required before training.",
              "Task 024 must measure token counts with the owned Gen2 tokenizer; repetition cannot replace unique data.",
              "Historical v1.0.0 source bytes and external backup/restore limitations remain explicit; no historical lineage is rewritten."]
    task.update(status="COMPLETED", completed_on=date, implementation_state="verified-source-inventory-and-permission-gates",
                verification_status="passed-inventory-no-gen2-corpus-release", verification_result=summary, evidence_records=evidence,
                remaining_release_gates=limits, target_paths=["data_governance/foundation/", "tools/inventory_llm_sources.py",
                    "tests/test_foundation_source_inventory.py", "docs/LLM_TRAINING_SOURCE_INVENTORY.v1.md"])
    backlog["updated_on"] = date
    backlog["scope_of_this_revision"] = "LLM-TASK-015 through 019 completed: audit, cleanup, frozen design/evaluation and source inventory. Gen2 corpus ingestion/release, model implementation, training and product acceptance remain open."
    backlog["execution"]["next_task_id"] = "LLM-TASK-020"
    counts = Counter(row["status"] for row in backlog["tasks"])
    backlog["status_summary"] = {key: counts[key] for key in ("TODO", "IN_PROGRESS", "COMPLETED")}
    next(row for row in backlog["milestones"] if row["id"] == "M1")["status"] = "IN_PROGRESS"
    receipt_path = REPORT.relative_to(ROOT).as_posix()
    backlog["current_checkpoint"] = {"completed_task_id": "LLM-TASK-019", "evidence": receipt_path, "summary": summary,
                                     "active_artifact_id": None, "neural_ready": False}
    backlog["training_source_inventory"] = {"path": evidence[3], "manifest": evidence[4], "version": "1.0.0",
        "existing_unique_eligible_records": 227, "existing_pool_proxy_tokens": 112418, "proxy_tokenizer_version": "1.1.0",
        "gen2_release_ready_tokens": 0, "gap_scenarios": inventory["tokenGap"], "external_sources_admitted": 0}
    markdown = (ROOT / DOCS[1]).read_text(encoding="utf-8")
    markdown = markdown.replace("**41 active tasks: 4 completed (audit, cleanup, architecture and independent evaluation), 37 TODO. No trained conversational LLM release is accepted.**",
        "**41 active tasks: 5 completed (audit, cleanup, architecture, evaluation and source inventory), 36 TODO. No trained conversational LLM release is accepted.**")
    start = markdown.index("**Next: LLM-TASK-019")
    end = markdown.index("\n\n", start)
    markdown = markdown[:start] + "**Next: LLM-TASK-020 — build reproducible corpus ingestion and normalization.** Admit only source revisions with verified permitted-use and provenance evidence.\n\n" + summary + "\n\n[Source inventory and token gap](Voltforge_AI/docs/LLM_TRAINING_SOURCE_INVENTORY.v1.md) | [Task 019 evidence](Voltforge_AI/evaluation/reports/llm-task-019-validation.json)" + markdown[end:]
    markdown = markdown.replace("| LLM-TASK-019 through LLM-TASK-023 | TODO |", "| LLM-TASK-019 through LLM-TASK-023 | IN_PROGRESS |")
    heading = "#### LLM-TASK-019 — " + task["title"]
    start = markdown.index(heading)
    end = markdown.index("\n#### LLM-TASK-020", start)
    section = markdown[start:end].replace("**Status:** TODO.", "**Status:** COMPLETED.", 1)
    a, b = section.index("**Target paths:**"), section.index("\n", section.index("**Target paths:**"))
    section = section[:a] + "**Target paths:** " + ", ".join("`" + path + "`" for path in task["target_paths"]) + "." + section[b:]
    section += f"**Verified result ({date}):** " + summary + "\n\n**Evidence:** [Permission and source inventory](Voltforge_AI/data_governance/foundation/inventory.v1.json), [manifest](Voltforge_AI/data_governance/foundation/manifest.v1.json), [token-gap plan](Voltforge_AI/docs/LLM_TRAINING_SOURCE_INVENTORY.v1.md), [validation](Voltforge_AI/evaluation/reports/llm-task-019-validation.json).\n"
    markdown = markdown[:start] + section + markdown[end:]
    causes = read(ROOT / DOCS[2])
    causes["updated_on"] = date
    causes["summary"]["implementation_this_revision"] = backlog["scope_of_this_revision"]
    causes["summary"]["acceptance_limit"] = summary + " All root causes remain open or partial until the remaining implementation and acceptance tasks pass."
    for cause in causes["root_causes"]:
        if "LLM-TASK-019" in cause["remediation_task_ids"]:
            cause["source_inventory_progress"] = {"task_id": "LLM-TASK-019", "evidence": receipt_path,
                "result": "Source rights, corpus quarantine and token-gap evidence recorded. Broad Gen2 data acquisition/release and actual training remain open."}
    causes["evidence_files"].extend(path for path in evidence if path not in causes["evidence_files"])
    causes["current_checkpoint"] = {"completed_task_id": "LLM-TASK-019", "evidence": receipt_path, "next_task_id": "LLM-TASK-020"}
    analysis = (ROOT / DOCS[3]).read_text(encoding="utf-8")
    analysis = analysis.replace("## Current checkpoint: LLM-TASK-018 completed", "## Current checkpoint: LLM-TASK-019 completed\n\n" + summary +
        "\n\n[Source inventory and gap plan](Voltforge_AI/docs/LLM_TRAINING_SOURCE_INVENTORY.v1.md) | [Task 019 verification](Voltforge_AI/evaluation/reports/llm-task-019-validation.json)\n\n## Previous checkpoint: LLM-TASK-018 completed", 1)
    analysis = analysis.replace("**LLM-TASK-019 is next: inventory training sources and permitted uses.**", "**LLM-TASK-020 is next: reproducible corpus ingestion and normalization.**")
    # All substantive tests and checks passed before completion status is recorded.
    snapshot_files = []
    for name in (*DOCS, "VOLTForge_AI_RUNBOOK.md", "runbook-contract.v1.json", ".gitattributes"):
        path = SNAPSHOT / name
        snapshot_files.append({"path": name, "sha256": file_hash(path), "bytes": path.stat().st_size})
    write_json(SNAPSHOT / "manifest.json", {"taskId": "LLM-TASK-019", "files": snapshot_files,
        "scope": "Seven planning/runbook/attributes files saved before task-019 edits. Earlier protected registry/data bytes are verified against the task-016 snapshot.",
        "restore": "Review later changes before copying any individual backup; no automatic reset or deletion."})
    write_json(AI / "evaluation/reports/llm-task-019-checks.json", {"status": "passed", "inventoryRecomputed": verified, "checks": checks})
    receipt = {"schemaVersion": 1, "taskId": "LLM-TASK-019", "status": "inventory-verified-planning-check-pending",
        "recordedAtUtc": datetime.now(timezone.utc).isoformat(), "summary": summary, "inventoryVerification": verified,
        "tests": {"passed": len(cases), "failed": 0, "skipped": skipped, "additionalPassingSubtests": subtests,
                  "junitReportedTotalIncludingSubtests": len(cases) + subtests, "junit": root_descriptor(junit_path)},
        "existingPool": {key: inventory["existingPool"][key] for key in ("inputRecords", "uniqueEligibleRecords", "proxyTokens", "excluded")},
        "tokenGap": inventory["tokenGap"], "limits": limits, "network": {"inventoryNetworkAccessed": False,
            "separatePermissionMetadataFetches": 7, "rawPageBodiesStored": False, "externalCorpusAcquired": False},
        "preservation": {"savedPlanningFiles": 7, "historicalContractBindingsUnchanged": 12, "protectedArtifactDataFilesUnchanged": 131,
                         "task018SuiteAndPolicyUnchanged": True},
        "gen2CorpusReleased": False, "modelReleaseApproved": False, "trainingAllowed": False, "nextTask": "LLM-TASK-020"}
    write_json(REPORT, receipt)
    write_json(backlog_path, backlog)
    (ROOT / DOCS[1]).write_text(markdown, encoding="utf-8", newline="\n")
    write_json(ROOT / DOCS[2], causes)
    (ROOT / DOCS[3]).write_text(analysis, encoding="utf-8", newline="\n")
    completed = subprocess.run([sys.executable, "-B", "tools/verify_llm_backlogs.py"], cwd=AI, capture_output=True, text=True, encoding="utf-8")
    if completed.returncode:
        raise ValueError("Planning check failed; receipt remains explicitly pending:\n" + completed.stdout + completed.stderr)
    receipt["planningCheck"] = json.loads(completed.stdout)
    paths = [ROOT / name for name in DOCS] + [AI / name for name in (
        "docs/LLM_TRAINING_SOURCE_INVENTORY.v1.md", "docs/VOLTForge_AI_RUNBOOK.md", "docs/runbook-contract.v1.json", ".gitattributes",
        "tools/inventory_llm_sources.py", "tools/capture_foundation_source_evidence.py", "tools/record_llm_source_inventory.py",
        "tests/test_foundation_source_inventory.py")]
    # root_descriptor accepts workspace planning files as well as AI source files.
    receipt["sourceFingerprints"] = [root_descriptor(path) for path in paths]
    receipt["evidence"] = [{"path": path, "sha256": file_hash(ROOT / path), "bytes": (ROOT / path).stat().st_size} for path in evidence if ROOT / path != REPORT]
    receipt["status"] = "completed-source-inventory-no-gen2-corpus-release"
    write_json(REPORT, receipt)
    print(json.dumps({"status": receipt["status"], "testsPassed": len(cases), "passingSubtests": subtests,
                      "existingPoolProxyTokens": 112418, "gen2ReleaseReadyTokens": 0, "nextTask": receipt["nextTask"]}))


if __name__ == "__main__":
    main()
