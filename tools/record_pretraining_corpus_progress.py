"""Record verified task-023 progress without claiming a released training corpus."""
from collections import Counter
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
AI = Path(__file__).resolve().parents[1]
ROOT = AI.parent
sys.path.insert(0, str(AI))
from data_governance.pretraining import corpus

REPORTS = AI / "evaluation/reports"
REPORT = REPORTS / "llm-task-023-progress.json"
DOCS = ("VOLTFORGE_LOCAL_LLM_BUILD_BACKLOG.json", "VOLTFORGE_LOCAL_LLM_BUILD_BACKLOG.md", "AI_CHAT_ROOT_CAUSE_BACKLOG.json", "AI_CHAT_ROOT_CAUSE_ANALYSIS.md")


def read(path): return json.loads(path.read_text(encoding="utf-8"))


def write(path, value): path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def descriptor(path): return {"path": path.relative_to(ROOT).as_posix(), "sha256": corpus.sha(path.read_bytes()), "bytes": path.stat().st_size}


def main():
    if REPORT.exists(): raise ValueError("Progress receipt already exists; use a new version for further work")
    built = read(REPORTS / "llm-task-023-corpus-build.json")
    # The required integration test recomputes this exact immutable candidate.
    # Recheck byte/code bindings here without repeating the full token count.
    checked = corpus.verify(Path(built["releasePath"]), recompute=False)
    score = checked["scorecard"]
    if score["trainingAllowed"] is not False or score["trainingUniqueProxyTokens"] != 722234 or score["splitCounts"] != {"train": 26, "validation": 0, "test": 5, "quarantine": 7}: raise ValueError("Unexpected candidate result; review before recording")
    index = read(REPORTS / "llm-task-023-index-evaluation.json")
    if index["status"] != "passed" or index["fixtureRecall"] != 1 or index["realInputRecall"] != 1 or index["maximumComparedFixtureDocuments"] != 6000: raise ValueError("Index evidence incomplete")
    for entry in index["sourceFingerprints"]:
        if corpus.sha((AI / entry["path"]).read_bytes()) != entry["sha256"]: raise ValueError("Index evidence code changed")
    junit = REPORTS / "llm-task-023-tests.xml"
    tree = ET.parse(junit).getroot()
    cases = list(tree.iter("testcase"))
    if len(cases) < 330 or any(list(tree.iter(tag)) for tag in ("failure", "error", "skipped")): raise ValueError("Regression evidence incomplete")
    if not any(case.get("name") == "test_candidate_reproduces_and_preserves_all_old_exclusions" for case in cases): raise ValueError("Full corpus recomputation test missing")
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
    snapshot = ROOT / "backlog_history/llm-task-023-before"
    for entry in read(snapshot / "manifest.json")["files"]:
        if corpus.sha((snapshot / entry["backup"]).read_bytes()) != entry["sha256"]: raise ValueError("Preserved planning snapshot changed")
    backlog = read(ROOT / DOCS[0])
    task = next(row for row in backlog["tasks"] if row["id"] == "LLM-TASK-023")
    if task["status"] != "TODO" or backlog["execution"]["next_task_id"] != task["id"]: raise ValueError("Unexpected continuation checkpoint")
    date = datetime.now(timezone(timedelta(hours=5, minutes=30))).date().isoformat()
    manifest = (Path(checked["releasePath"]) / "manifest.json").relative_to(ROOT).as_posix()
    receipt_path = REPORT.relative_to(ROOT).as_posix()
    evidence = [manifest, "Voltforge_AI/docs/LLM_PRETRAINING_CORPUS_CANDIDATE.v1.md", "backlog_history/llm-task-023-before/manifest.json"] + ["Voltforge_AI/evaluation/reports/" + name for name in (
        "llm-task-023-acquisition.json", "llm-task-023-acquisition-v2.json", "llm-task-023-corpus-build.json", "llm-task-023-index-evaluation.json", "llm-task-023-tests.xml", "llm-task-023-checks.json", "llm-task-023-progress.json")]
    summary = ("Implemented bounded pinned-source acquisition, exact per-file license evidence, immutable mixed corpus candidates, indexed leakage checking, source/domain/language/split token accounting, deterministic 40/30/20/10 mixture quotas and three-pass repetition limits. "
        "Acquired 21 selected source files and two license files from exact Python devguide and FreeRTOS commits; one additional file is denied by the inherited text-quality gate. "
        "The 38-record candidate combines those source files with 17 previously surviving owned-domain examples; all 50 previous exclusions remain excluded. "
        "Seven language documents are quarantined by the unchanged protected/family policy. Surviving partitions contain 26 train records with 722,234 existing-tokenizer proxy tokens, zero validation records and five test records. "
        "Indexed matches equal exhaustive matches on 93 fixture documents and all 38 real inputs against 1,005 real guards. A separate 6,000-document fixture benchmark and common-word resource-failure controls pass; no large-corpus or semantic-completeness claim. "
        f"{len(cases)} regression tests passed, along with governance, runbook and preservation checks. "
        "Task 023 remains IN_PROGRESS: language/validation coverage, independent source breadth, explicit source-use admission and corpus acceptance are incomplete. No tokenizer fitting, model training or activation occurred.")
    remaining = [
        "Complete source-use admission for exact new source bytes and retained license/privacy/retention/removal obligations; candidate acquisition is not a training approval.",
        "Add independent general-language/technical, code, electronics and math families with populated validation coverage; the current surviving language pool is empty and code has one repository family.",
        "Review lexical containment precision using independent controls before any new versioned admission policy; retain current protected evaluation bytes and all existing exclusions.",
        "Measure ingestion, tokenization and indexed matching on the intended corpus scale; the 6,000-document disjoint-vocabulary scan fixture excludes feature-construction cost and does not close production scale.",
        "Publish an admitted pilot/input corpus with a declared provisional token budget; Gen2 recounting belongs to task 024 and measured production budget/mixture acceptance remains in tasks 031/032."]
    # Correct the existing acceptance-order loop without dropping its measured
    # budget requirement: the input corpus precedes tokenizer/pilots; measured
    # production corpus adequacy is retained explicitly in task 032.
    previous_dod = task["definition_of_done"][0]
    new_dod = "A reproducible pilot/input corpus release covers each declared capability, passes its source/quality/split gates, and records a provisional unique-token budget for tokenizer and scaling experiments; measured production-budget acceptance remains mandatory in LLM-TASK-031/032."
    task["definition_of_done"][0] = new_dod
    task32 = next(row for row in backlog["tasks"] if row["id"] == "LLM-TASK-032")
    production_dod = "Before approving the production training recipe, verify that the immutable general-language/code/domain corpus covers each declared capability and meets the measured unique-token/exposure budget chosen from LLM-TASK-031 scaling evidence; a pilot/input corpus release does not satisfy this gate."
    task32["definition_of_done"].append(production_dod)
    task32["inherited_corpus_budget_gate"] = {"from_task": task["id"], "original_requirement": previous_dod, "reason": "Resolve input-corpus/scaling acceptance-order loop while retaining measured production data adequacy."}
    task.update(status="IN_PROGRESS", started_on=date, implementation_state="verified-candidate-assembly-release-gates-open",
                verification_status="passed-candidate-controls-not-corpus-acceptance", verification_result=summary,
                evidence_records=evidence, remaining_release_gates=remaining,
                target_paths=["data_governance/pretraining/", "corpus/acquisition/", "corpus/pretraining-candidates/", "tools/build_pretraining_corpus.py", "tests/test_pretraining_corpus.py", "docs/LLM_PRETRAINING_CORPUS_CANDIDATE.v1.md"],
                acceptance_staging={"input_corpus_owner": "LLM-TASK-023", "gen2_tokenizer_owner": "LLM-TASK-024", "measured_scaling_owner": "LLM-TASK-031", "production_corpus_budget_owner": "LLM-TASK-032", "original_requirement_preserved_in": "LLM-TASK-032.inherited_corpus_budget_gate"})
    backlog["updated_on"] = date
    backlog["scope_of_this_revision"] = "LLM-TASK-015 through 022 completed; task 023 candidate corpus assembly/indexed controls are verified, but source admission, independent language/validation breadth and corpus acceptance remain in progress."
    counts = Counter(row["status"] for row in backlog["tasks"])
    backlog["status_summary"] = {key: counts[key] for key in ("TODO", "IN_PROGRESS", "COMPLETED")}
    backlog["current_checkpoint"].update(active_task_id=task["id"], active_task_evidence=receipt_path, active_task_summary=summary)
    backlog["pretraining_corpus_candidate"] = {"manifest": manifest, "candidate_records": 38, "split_counts": score["splitCounts"], "candidate_train_proxy_tokens": 722234, "gen2_corpus_released": False, "training_allowed": False}
    markdown = (ROOT / DOCS[1]).read_text(encoding="utf-8")
    markdown = re.sub(r"^Updated: [\d-]+\.", "Updated: " + date + ".", markdown, count=1, flags=re.MULTILINE)
    markdown = markdown.replace("**41 active tasks: 8 completed (audit through owned-domain candidate generation), 33 TODO. No trained conversational LLM release is accepted.**", "**41 active tasks: 8 completed, 1 in progress (LLM-TASK-023), 32 TODO. No trained conversational LLM release is accepted.**", 1)
    start = markdown.index("**Next: LLM-TASK-023")
    end = markdown.index("\n\n", start)
    markdown = markdown[:start] + "**Continue: LLM-TASK-023 — pretraining corpus release remains IN_PROGRESS.** Candidate assembly and controls are verified; source admission, language/validation breadth and corpus adequacy remain open.\n\n" + summary + "\n\n[Task 023 candidate corpus and gates](Voltforge_AI/docs/LLM_PRETRAINING_CORPUS_CANDIDATE.v1.md) | [Progress evidence](Voltforge_AI/evaluation/reports/llm-task-023-progress.json)" + markdown[end:]
    start = markdown.index("#### LLM-TASK-023 — " + task["title"])
    end = markdown.index("\n#### LLM-TASK-024", start)
    section = markdown[start:end].replace("**Status:** TODO.", "**Status:** IN_PROGRESS.", 1).replace(previous_dod, new_dod, 1)
    a = section.index("**Target paths:**")
    b = section.index("\n", a)
    section = section[:a] + "**Target paths:** " + ", ".join("`" + name + "`" for name in task["target_paths"]) + "." + section[b:]
    section += "\n**Verified progress (" + date + "):** " + summary + "\n\n**Remaining:** " + " ".join(remaining) + "\n\n**Evidence:** [Candidate manifest](" + manifest + "), [progress receipt](Voltforge_AI/evaluation/reports/llm-task-023-progress.json).\n"
    markdown = markdown[:start] + section + markdown[end:]
    start = markdown.index("#### LLM-TASK-032 —")
    end = markdown.index("\n#### LLM-TASK-033", start)
    milestone_boundary = markdown.find("\n### ", start, end)
    if milestone_boundary >= 0: end = milestone_boundary
    markdown = markdown[:end] + "\n**Retained production corpus acceptance:** " + production_dod + "\n\nThe original task-023 measured-budget condition is retained here so input corpus preparation can precede tokenizer and scaling experiments.\n" + markdown[end:]
    causes = read(ROOT / DOCS[2])
    causes["updated_on"] = date
    causes["summary"]["implementation_this_revision"] = backlog["scope_of_this_revision"]
    causes["summary"]["acceptance_limit"] = summary + " Root causes remain open/partial until their linked tasks pass."
    for cause in causes["root_causes"]:
        if task["id"] in cause["remediation_task_ids"]:
            cause["pretraining_corpus_progress"] = {"task_id": task["id"], "status": "IN_PROGRESS", "evidence": receipt_path, "result": "Candidate assembly, indexed leakage and accounting verified; source-use admission, independent language/validation and training corpus release incomplete."}
    causes["evidence_files"].extend(name for name in evidence if name not in causes["evidence_files"])
    causes["current_checkpoint"].update(active_task_id=task["id"], active_task_evidence=receipt_path, next_task_id=task["id"])
    analysis = (ROOT / DOCS[3]).read_text(encoding="utf-8")
    analysis = analysis.replace("## Current checkpoint: LLM-TASK-022 completed", "## Current work: LLM-TASK-023 in progress\n\n" + summary + "\n\n[Candidate corpus evidence and remaining gates](Voltforge_AI/docs/LLM_PRETRAINING_CORPUS_CANDIDATE.v1.md)\n\n## Last completed checkpoint: LLM-TASK-022 completed", 1)
    analysis = analysis.replace("**LLM-TASK-023 is next: approved broad corpus release, independent validation data and remaining domain-admission gates.**", "**Continue LLM-TASK-023: source-use admission, independent language/validation coverage and admitted input-corpus release.**")
    corpus.write_immutable(REPORTS / "llm-task-023-checks.json", corpus.data({"status": "passed", "corpusByteVerification": checked, "recomputeEvidence": {"junit": descriptor(junit), "test": "test_candidate_reproduces_and_preserves_all_old_exclusions"}, "checks": checks}))
    receipt = {"schemaVersion": 1, "taskId": task["id"], "status": "verified-progress-planning-check-pending", "recordedAtUtc": datetime.now(timezone.utc).isoformat(),
        "summary": summary, "corpusVerification": checked, "indexEvaluation": descriptor(REPORTS / "llm-task-023-index-evaluation.json"),
        "tests": {"passed": len(cases), "failed": 0, "skipped": 0, "junit": descriptor(junit)}, "remainingWork": remaining,
        "planningCorrection": {"originalTask023Requirement": previous_dod, "inputCorpusRequirement": new_dod, "retainedTask032Requirement": production_dod},
        "preservation": {"planningFilesSnapshotted": 7, "historicalContractBindingsUnchanged": 12, "protectedArtifactDataFilesUnchanged": 131, "task019Through022RecomputedUnchanged": True},
        "taskCompleted": False, "gen2CorpusReleased": False, "trainingAllowed": False, "modelReleaseApproved": False,
        "explicitPinnedSourceAcquisitionPerformed": True, "assemblyNetworkAccessed": False, "nextTask": task["id"]}
    corpus.write_immutable(REPORT, corpus.data(receipt))
    write(ROOT / DOCS[0], backlog)
    (ROOT / DOCS[1]).write_text(markdown, encoding="utf-8", newline="\n")
    write(ROOT / DOCS[2], causes)
    (ROOT / DOCS[3]).write_text(analysis, encoding="utf-8", newline="\n")
    result = subprocess.run([sys.executable, "-B", "tools/verify_llm_backlogs.py"], cwd=AI, capture_output=True, text=True, encoding="utf-8")
    if result.returncode: raise ValueError("Planning check failed; receipt remains pending: " + result.stdout + result.stderr)
    receipt["planningCheck"] = json.loads(result.stdout)
    receipt["sourceFingerprints"] = [descriptor(ROOT / name) for name in DOCS] + [descriptor(AI / name) for name in (
        "docs/LLM_PRETRAINING_CORPUS_CANDIDATE.v1.md", "docs/VOLTForge_AI_RUNBOOK.md", "docs/runbook-contract.v1.json", ".gitattributes", "tests/test_pretraining_corpus.py", "tools/build_pretraining_corpus.py", "tools/acquire_pretraining_sources.py", "tools/evaluate_pretraining_index.py", "tools/record_pretraining_corpus_progress.py")]
    receipt["evidence"] = [descriptor(ROOT / name) for name in evidence if ROOT / name != REPORT]
    receipt["status"] = "in-progress-candidate-controls-verified-corpus-release-incomplete"
    write(REPORT, receipt)
    print(json.dumps({"status": receipt["status"], "testsPassed": len(cases), "candidateTrainingProxyTokens": 722234, "nextTask": receipt["nextTask"]}))


def finalize_pending():
    receipt = read(REPORT)
    if receipt["status"] != "verified-progress-planning-check-pending": raise ValueError("Only a pending planning receipt may be finalized")
    corpus.verify(Path(receipt["corpusVerification"]["releasePath"]), recompute=False)
    if descriptor(ROOT / receipt["tests"]["junit"]["path"]) != receipt["tests"]["junit"]: raise ValueError("Accepted regression evidence changed")
    if read(REPORTS / "llm-task-023-checks.json")["status"] != "passed": raise ValueError("Required checks are incomplete")
    result = subprocess.run([sys.executable, "-B", "tools/verify_llm_backlogs.py"], cwd=AI, capture_output=True, text=True, encoding="utf-8")
    if result.returncode: raise ValueError(result.stdout + result.stderr)
    receipt["planningCheck"] = json.loads(result.stdout)
    evidence = next(row for row in read(ROOT / DOCS[0])["tasks"] if row["id"] == "LLM-TASK-023")["evidence_records"]
    receipt["sourceFingerprints"] = [descriptor(ROOT / name) for name in DOCS] + [descriptor(AI / name) for name in (
        "docs/LLM_PRETRAINING_CORPUS_CANDIDATE.v1.md", "docs/VOLTForge_AI_RUNBOOK.md", "docs/runbook-contract.v1.json", ".gitattributes", "tests/test_pretraining_corpus.py", "tools/build_pretraining_corpus.py", "tools/acquire_pretraining_sources.py", "tools/evaluate_pretraining_index.py", "tools/record_pretraining_corpus_progress.py")]
    receipt["evidence"] = [descriptor(ROOT / name) for name in evidence if ROOT / name != REPORT]
    impact = read(REPORTS / "llm-task-023-source-impact.json")
    if len(impact["affectedDocuments"]) != 14 or sum(row["tokens"] for row in impact["affectedDocuments"]) != 718908 or impact["action"] != "report-only-no-files-deleted": raise ValueError("Source impact evidence mismatch")
    receipt["sourceImpactEvidence"] = descriptor(REPORTS / "llm-task-023-source-impact.json")
    receipt["planningFinalization"] = "Moved the retained task-032 criterion into its correct Markdown section and refreshed the stale continuation instruction; all planning checks now pass."
    receipt["status"] = "in-progress-candidate-controls-verified-corpus-release-incomplete"
    write(REPORT, receipt)
    print(json.dumps({"status": receipt["status"], "testsPassed": receipt["tests"]["passed"], "nextTask": receipt["nextTask"]}))


if __name__ == "__main__":
    if sys.argv[1:] == ["--finalize-pending"]: finalize_pending()
    elif sys.argv[1:]: raise ValueError("Unsupported arguments")
    else: main()
