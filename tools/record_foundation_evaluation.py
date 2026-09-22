"""Record the verified task-018 checkpoint and synchronize its two backlogs once."""
from __future__ import annotations

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
from evaluation.foundation.suite import canonical, read, sha, verify_suite
from tools.verify_foundation_evaluation_binding import verify as verify_binding

REPORT = AI / "evaluation/reports/llm-task-018-validation.json"
SNAPSHOT = ROOT / "backlog_history/llm-task-018-before"
TESTS = AI / "evaluation/reports/llm-task-018-tests.xml"
ROOT_DOCS = ("VOLTFORGE_LOCAL_LLM_BUILD_BACKLOG.json", "VOLTFORGE_LOCAL_LLM_BUILD_BACKLOG.md",
             "AI_CHAT_ROOT_CAUSE_BACKLOG.json", "AI_CHAT_ROOT_CAUSE_ANALYSIS.md")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def descriptor(path, relative_to=ROOT):
    return {"path": path.relative_to(relative_to).as_posix(), "sha256": sha(path.read_bytes()), "bytes": path.stat().st_size}


def main():
    if REPORT.exists() or (SNAPSHOT / "manifest.json").exists():
        raise ValueError("Completion evidence already exists; preserve history and create a new receipt for later changes")
    date = datetime.now(timezone(timedelta(hours=5, minutes=30))).date().isoformat()
    suite, binding = verify_suite(), verify_binding()
    baseline = read(AI / "evaluation/reports/llm-task-018-baseline.json")
    if baseline["suite"] != suite or baseline["runtime"]["ready"] or baseline["observedNeuralForwardCalls"] or baseline["observedModelTokens"]:
        raise ValueError("Expected exact frozen no-model baseline")
    if len(baseline["scorecards"]) != 12 or any(not row["fullCohort"] or len(row["rows"]) != 700 or row["releaseApproved"] for row in baseline["scorecards"]):
        raise ValueError("Incomplete baseline cohorts")
    if any(row["decision"] != ("diagnostic-only" if row["lane"] == "rule_only" else "blocked") for row in baseline["scorecards"]):
        raise ValueError("Unexpected baseline decision")
    test_tree = ET.parse(TESTS).getroot()
    test_suites = list(test_tree.iter("testsuite"))
    if not test_suites or any(int(row.get("failures", "0")) or int(row.get("errors", "0")) for row in test_suites):
        raise ValueError("Final task-018 tests did not all pass")
    # pytest's suite total includes passing unittest subtests; testcase nodes do not.
    count = len(list(test_tree.iter("testcase")))
    subtests = sum(int(row.get("tests", "0")) for row in test_suites) - count
    skipped = sum(int(row.get("skipped", "0")) for row in test_suites)
    if count < 115 or skipped:
        raise ValueError("Final regression selection is incomplete")
    checks = []
    for command in (
        ["tools/verify_llm_foundation.py", "--check-preserved-history"],
        ["tools/verify_runbook.py"],
        ["tools/scan_quality_boundaries.py", "--check"],
        ["tools/snapshot_llm_cleanup.py", "--verify", "../backlog_history/llm-task-016-before-20260906T054449Z"],
    ):
        result = subprocess.run([sys.executable, "-B", *command], cwd=AI, capture_output=True, text=True, encoding="utf-8")
        if result.returncode:
            raise ValueError("Required check failed: " + " ".join(command) + "\n" + result.stdout + result.stderr)
        checks.append({"command": [".toolchains/gen1/Scripts/python.exe", "-B", *command], "exitCode": 0,
                       "result": json.loads(result.stdout)})
    old_design = read(AI / "evaluation/reports/llm-task-017-foundation-contract.json")
    for row in old_design["inputs"]:
        if sha((AI / row["path"]).read_bytes()) != row["sha256"]:
            raise ValueError("Task-017 historical design or verifier changed")
    sources = []
    for name in ROOT_DOCS:
        path = SNAPSHOT / name
        sources.append({"originalPath": name, **descriptor(path, SNAPSHOT)})
    for name in ("VOLTForge_AI_RUNBOOK.md", "runbook-contract.v1.json"):
        path = SNAPSHOT / name
        sources.append({"originalPath": "Voltforge_AI/docs/" + name, **descriptor(path, SNAPSHOT)})
    write_json(SNAPSHOT / "manifest.json", {"taskId": "LLM-TASK-018", "files": sources,
        "scope": "Six planning/runbook files copied before task-018 edits; registry/data preservation is checked against the earlier task-016 snapshot.",
        "restore": "Review later changes and copy individual saved files; no automatic reset or deletion."})
    evidence = [
        "Voltforge_AI/evaluation/foundation/manifest.v1.json", "Voltforge_AI/evaluation/foundation/acceptance-policy.v1.json",
        "Voltforge_AI/foundation/evaluation-binding.v1.json", "Voltforge_AI/docs/LLM_FOUNDATION_EVALUATION.v1.md",
        "Voltforge_AI/evaluation/reports/llm-task-018-suite-freeze.json", "Voltforge_AI/evaluation/reports/llm-task-018-baseline.json",
        "Voltforge_AI/evaluation/reports/llm-task-018-evaluation-binding.json", "Voltforge_AI/evaluation/reports/llm-task-018-tests.xml",
        "Voltforge_AI/evaluation/reports/llm-task-018-preserved-design.json", "Voltforge_AI/evaluation/reports/llm-task-018-checks.json",
        "Voltforge_AI/evaluation/reports/llm-task-018-validation.json", "backlog_history/llm-task-018-before/manifest.json",
    ]
    summary = (f"Frozen independent evaluation: 100 development, 100 validation, 500 core acceptance cases across 10 categories, "
        f"100 adversarial and 100 exact-board firmware cases. The core has 100 scenario families and 73 template families. "
        f"{count} tests passed with no failures or skips, plus {subtests} passing subtests; these are harness/contract regressions, not model quality. "
        "Twelve baseline scorecards retain all 700 acceptance cases per lane/seed; actual neural forward/token counters are zero. "
        "LED/OLED confusion and repeated uncertainty remain reproduced; changed-value arithmetic passes. "
        "Frozen design, 12 historical bindings and 131 protected artifact/data files unchanged. "
        "No model training, acceptance, activation, browser verification or server performance result.")
    backlog_path = ROOT / ROOT_DOCS[0]
    backlog = read(backlog_path)
    task = next(row for row in backlog["tasks"] if row["id"] == "LLM-TASK-018")
    if task["status"] != "TODO" or backlog["execution"]["next_task_id"] != "LLM-TASK-018":
        raise ValueError("Unexpected continuation checkpoint")
    task.update(status="COMPLETED", implementation_state="verified-independent-harness-and-frozen-policy",
                verification_status="passed-harness-and-baseline-no-model-acceptance", completed_on=date,
                verification_result=summary, evidence_records=evidence,
                target_paths=["evaluation/foundation/", "tools/evaluate_llm_foundation.py", "tools/verify_foundation_evaluation_binding.py",
                              "tests/test_foundation_evaluation.py", "tests/test_foundation_evaluation_binding.py", "docs/LLM_FOUNDATION_EVALUATION.v1.md"],
                remaining_release_gates=["Evaluation-only confidential custody before candidate training and Gen2 ingestion leakage enforcement.",
                    "Owned Gen2 tokenizer/model/trainer, attested candidate and validated native checkpoint runtime.",
                    "Actual candidate outputs, independent signed expert/compiler/claim/retrieval receipts and all three seeds.",
                    "Actual product/browser/server measurements, independent release review and governed activation."])
    backlog["updated_on"] = date
    backlog["scope_of_this_revision"] = "LLM-TASK-015 through 018 completed: audit, cleanup, frozen architecture and independent evaluation foundation. Corpus acquisition, owned Gen2 implementation, training and product acceptance remain open."
    backlog["execution"]["next_task_id"] = "LLM-TASK-019"
    backlog["status_summary"] = {key: Counter(row["status"] for row in backlog["tasks"])[key] for key in ("TODO", "IN_PROGRESS", "COMPLETED")}
    next(row for row in backlog["milestones"] if row["id"] == "M0")["status"] = "COMPLETED"
    backlog["acceptance_policy"].update(state="frozen v1 in LLM-TASK-018; engineering targets, no candidate acceptance",
        policy_path=evidence[1], suite_manifest=evidence[0], design_binding=evidence[2],
        suite_sha256=suite["suiteSha256"], policy_sha256=suite["policySha256"],
        seal_boundary="Checksum locked, not confidential; evaluation-only custody and training leakage audit remain mandatory before candidate training/release.")
    backlog["acceptance_policy"]["capability_targets"][1] = "At least 90% of all 100 predeclared eligible exact-FQBN firmware cases pass functional review and compilation. Missing reviews/receipts block completion; unsupported claims cannot shrink this frozen denominator."
    backlog["acceptance_policy"]["capability_targets"].append("All 100 mandatory adversarial cases pass the frozen independent rubric; no critical failures.")
    backlog["current_checkpoint"] = {"completed_task_id": "LLM-TASK-018", "evidence": evidence[-2], "summary": summary, "active_artifact_id": None, "neural_ready": False}
    backlog["foundation_contract"]["evaluation_binding"] = evidence[2]
    markdown = (ROOT / ROOT_DOCS[1]).read_text(encoding="utf-8")
    markdown = markdown.replace("Updated: 2026-09-06.", f"Updated: {date}.", 1)
    markdown = markdown.replace("**41 active tasks: 3 completed (audit, cleanup and architecture contract), 38 TODO. No trained conversational LLM release is accepted.**",
        "**41 active tasks: 4 completed (audit, cleanup, architecture and independent evaluation), 37 TODO. No trained conversational LLM release is accepted.**")
    start = markdown.index("**Next: LLM-TASK-018")
    end = markdown.index("\n\n", start)
    markdown = markdown[:start] + "**Next: LLM-TASK-019 — inventory training sources and permitted uses.** Establish source rights, provenance and the approved corpus/token gap before ingestion or training.\n\n" + summary + "\n\n[Evaluation design](Voltforge_AI/docs/LLM_FOUNDATION_EVALUATION.v1.md) | [Task 018 evidence](Voltforge_AI/evaluation/reports/llm-task-018-validation.json)" + markdown[end:]
    markdown = markdown.replace("| LLM-TASK-015 through LLM-TASK-018 | IN_PROGRESS |", "| LLM-TASK-015 through LLM-TASK-018 | COMPLETED |")
    markdown = markdown.replace("Freeze the following proposed v1 targets in LLM-TASK-018 before evaluating candidates. They are engineering targets, not current results.",
        "The following v1 targets are now frozen in the task-018 policy and suite. They are engineering targets, not current model results. Acceptance files are checksum locked; confidential evaluation custody and training leakage enforcement remain required before candidate training.")
    markdown = markdown.replace("- At least 90% compilation success on eligible exact-FQBN firmware cases; unsupported cases use a separate denominator.",
        "- At least 90% of all 100 predeclared eligible exact-FQBN firmware cases pass functional review and compilation. Missing receipts block completion; unsupported claims cannot shrink this denominator.")
    markdown = markdown.replace("- All critical action authorization, identity isolation, fabricated execution and neural-source checks pass on the declared suite.",
        "- All 100 mandatory adversarial cases and all critical action authorization, identity isolation, fabricated execution and neural-source checks pass on the declared suite.")
    heading = "#### LLM-TASK-018 — " + task["title"]
    start = markdown.index(heading)
    end = markdown.index("\n### M1:", start)
    section = markdown[start:end].replace("**Status:** TODO.", "**Status:** COMPLETED.", 1)
    target_start = section.index("**Target paths:**")
    target_end = section.index("\n", target_start)
    section = section[:target_start] + "**Target paths:** " + ", ".join("`" + p + "`" for p in task["target_paths"]) + "." + section[target_end:]
    section += f"**Verified result ({date}):** " + summary + "\n\n**Evidence:** [Frozen suite](Voltforge_AI/evaluation/foundation/manifest.v1.json), [policy](Voltforge_AI/evaluation/foundation/acceptance-policy.v1.json), [baseline](Voltforge_AI/evaluation/reports/llm-task-018-baseline.json), [validation](Voltforge_AI/evaluation/reports/llm-task-018-validation.json).\n"
    markdown = markdown[:start] + section + markdown[end:]
    causes = read(ROOT / ROOT_DOCS[2])
    causes["updated_on"] = date
    causes["summary"]["implementation_this_revision"] = backlog["scope_of_this_revision"]
    causes["summary"]["acceptance_limit"] = summary + " All 14 root causes remain open or partial until their remaining implementation/acceptance tasks pass."
    for row in causes["root_causes"]:
        if "LLM-TASK-018" in row["remediation_task_ids"]:
            row["evaluation_progress"] = {"task_id": "LLM-TASK-018", "evidence": evidence[-2],
                "result": "Independent frozen suite, neural-source checks and no-model baseline recorded. This implements evaluation coverage, not the remaining runtime/model repair."}
    causes["evidence_files"].extend(path for path in evidence if path not in causes["evidence_files"])
    causes["current_checkpoint"] = {"completed_task_id": "LLM-TASK-018", "evidence": evidence[-2], "next_task_id": "LLM-TASK-019"}
    causes["initial_audit_note"] += " Task-018 evaluation_progress records new coverage without closing runtime defects."
    analysis = (ROOT / ROOT_DOCS[3]).read_text(encoding="utf-8")
    analysis = analysis.replace("## Current checkpoint: LLM-TASK-017 completed", "## Current checkpoint: LLM-TASK-018 completed\n\n" + summary +
        "\n\n[Evaluation design](Voltforge_AI/docs/LLM_FOUNDATION_EVALUATION.v1.md) | [Task 018 verification](Voltforge_AI/evaluation/reports/llm-task-018-validation.json)\n\n## Previous checkpoint: LLM-TASK-017 completed", 1)
    analysis = analysis.replace("**LLM-TASK-018 is next.**", "**LLM-TASK-019 is next: inventory training sources and permitted uses.**")
    # All acceptance tests and artifact checks have passed before status changes.
    write_json(AI / "evaluation/reports/llm-task-018-checks.json", {"status": "passed", "checks": checks, "binding": binding})
    receipt = {"schemaVersion": 1, "taskId": "LLM-TASK-018", "status": "harness-verified-planning-check-pending",
        "recordedAtUtc": datetime.now(timezone.utc).isoformat(), "summary": summary, "suite": suite,
        "tests": {"passed": count, "failed": 0, "skipped": skipped, "additionalPassingSubtests": subtests, "junit": descriptor(TESTS)},
        "baseline": {"report": descriptor(AI / "evaluation/reports/llm-task-018-baseline.json"), "scorecards": 12,
                     "casesPerScorecard": 700, "observedNeuralForwardCalls": 0, "observedModelTokens": 0,
                     "regressions": baseline["regressions"]},
        "instrumentationProbe": {"source": "test_actual_random_decoder_probe_has_no_release_credit", "randomUntrainedModel": True,
                                 "forwardCalls": 3, "lastPositionLogitsElements": 96, "generatedTokenIds": 3, "releaseEligible": False},
        "preservation": {"savedPlanningFiles": 6, "task017DesignAndVerifierFilesUnchanged": len(old_design["inputs"]),
                         "historicalBindingsUnchanged": 12, "protectedArtifactDataFilesUnchanged": 131},
        "modelReleaseApproved": False, "activationAllowed": False, "nextTask": "LLM-TASK-019",
        "limits": task["remaining_release_gates"], "earlierTestReports": "Draft/prefreeze receipts retain debugging history; counts overlap and must not be added."}
    write_json(REPORT, receipt)
    write_json(backlog_path, backlog)
    (ROOT / ROOT_DOCS[1]).write_text(markdown, encoding="utf-8", newline="\n")
    write_json(ROOT / ROOT_DOCS[2], causes)
    (ROOT / ROOT_DOCS[3]).write_text(analysis, encoding="utf-8", newline="\n")
    result = subprocess.run([sys.executable, "-B", "tools/verify_llm_backlogs.py"], cwd=AI, capture_output=True, text=True, encoding="utf-8")
    if result.returncode:
        raise ValueError("Planning validation failed; completion receipt remains explicitly pending:\n" + result.stdout + result.stderr)
    receipt["planningCheck"] = json.loads(result.stdout)
    receipt["status"] = "completed-independent-harness-no-model-release"
    paths = [ROOT / p for p in ROOT_DOCS] + [AI / p for p in (
        "docs/VOLTForge_AI_RUNBOOK.md", "docs/runbook-contract.v1.json", "docs/LLM_FOUNDATION_EVALUATION.v1.md",
        "tools/evaluate_llm_foundation.py", "tools/verify_foundation_evaluation_binding.py", "tools/record_foundation_evaluation.py",
        "tests/test_foundation_evaluation.py", "tests/test_foundation_evaluation_binding.py", "tests/test_llm_foundation_contract.py",
        "foundation/evaluation-binding.v1.json", ".gitattributes")]
    receipt["sourceFingerprints"] = [descriptor(path) for path in paths]
    receipt["evidence"] = [descriptor(ROOT / p) for p in evidence if ROOT / p != REPORT]
    write_json(REPORT, receipt)
    print(json.dumps({"status": receipt["status"], "testsPassed": count, "nextTask": receipt["nextTask"], "report": str(REPORT)}))


def correct_test_accounting():
    """Append corrected metadata; preserve the original receipt and frozen suite."""
    target = REPORT.with_name("llm-task-018-validation-v2.json")
    if target.exists():
        raise ValueError("Corrected receipt exists; never overwrite")
    previous = read(REPORT)
    tree = ET.parse(TESTS).getroot()
    cases = len(list(tree.iter("testcase")))
    total = sum(int(row.get("tests", "0")) for row in tree.iter("testsuite"))
    if (previous["tests"]["passed"], cases, total) != (120, 115, 120) or list(tree.iter("failure")) or list(tree.iter("error")):
        raise ValueError("Unexpected accounting correction inputs")
    old_phrase = "120 tests passed with no failures or skips, plus 5 passing subtests"
    new_phrase = "115 tests passed with no failures or skips, plus 5 passing subtests"
    old_path, new_path = REPORT.relative_to(ROOT).as_posix(), target.relative_to(ROOT).as_posix()
    for path in [ROOT / name for name in ROOT_DOCS] + [AI / "docs/LLM_FOUNDATION_EVALUATION.v1.md"]:
        text = path.read_text(encoding="utf-8").replace(old_phrase, new_phrase)
        text = text.replace("llm-task-018-validation.json", "llm-task-018-validation-v2.json")
        path.write_text(text, encoding="utf-8", newline="\n")
    # JSON roundtrip changes prose references but retains the original immutable JUnit and suite.
    corrected = json.loads(json.dumps(previous).replace(old_phrase, new_phrase).replace(old_path, new_path))
    corrected["receiptVersion"] = 2
    corrected["supersedes"] = descriptor(REPORT)
    corrected["correction"] = "Metadata accounting only: JUnit suite tests=120 includes 115 testcase nodes plus 5 passing subtests. The original receipt counted those subtests twice in prose; no fixtures, results, thresholds or model state changed."
    corrected["tests"].update(passed=cases, additionalPassingSubtests=total - cases, junitReportedTotalIncludingSubtests=total)
    corrected["summary"] = corrected["summary"].replace(old_phrase, new_phrase)
    corrected["recordedAtUtc"] = datetime.now(timezone.utc).isoformat()
    corrected["sourceFingerprints"] = [descriptor(ROOT / row["path"]) for row in previous["sourceFingerprints"]]
    corrected["evidence"] = [descriptor(ROOT / row["path"]) for row in previous["evidence"]]
    corrected["status"] = "harness-verified-planning-check-pending"
    write_json(target, corrected)
    result = subprocess.run([sys.executable, "-B", "tools/verify_llm_backlogs.py"], cwd=AI, capture_output=True, text=True, encoding="utf-8")
    if result.returncode:
        raise ValueError(result.stdout + result.stderr)
    corrected["planningCheck"] = json.loads(result.stdout)
    corrected["suite"] = verify_suite()
    verify_binding()
    corrected["status"] = "completed-independent-harness-no-model-release"
    write_json(target, corrected)
    print(json.dumps({"status": corrected["status"], "testsPassed": cases, "passingSubtests": total - cases, "report": str(target)}))


if __name__ == "__main__":
    if sys.argv[1:] == ["--correct-test-accounting"]:
        correct_test_accounting()
    elif sys.argv[1:]:
        raise ValueError("Unknown argument")
    else:
        main()
