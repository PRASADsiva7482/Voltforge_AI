"""Record task-023 source expansion with explicit source/corpus admission states."""
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
    target = ROOT / "backlog_history/llm-task-023-expansion-before"
    target.mkdir(parents=True, exist_ok=False)
    entries = []
    for name in PLANNING:
        source, backup = ROOT / name, target / name
        backup.parent.mkdir(parents=True, exist_ok=True)
        with backup.open("xb") as stream:
            stream.write(source.read_bytes())
        entries.append({**descriptor(source), "backup": name})
    write(target / "manifest.json", {"taskId": "LLM-TASK-023", "substep": "source-expansion", "files": entries})
    print(json.dumps({"status": "snapshotted", "files": len(entries)}))


def record():
    sys.path.insert(0, str(AI))
    from data_governance.source_expansion import acquisition, admission, candidate
    from model.gen2_tokenizer.release import verify_fixture
    from data_governance.leakage_review.evaluation import verify as verify_precision

    reports = AI / "evaluation/reports"
    receipt_path = reports / "llm-task-023-expansion-progress.json"
    checks_path = reports / "llm-task-023-expansion-checks.json"
    impact_path = reports / "llm-task-023-expansion-source-impact.json"
    if any(path.exists() for path in (receipt_path, checks_path, impact_path)):
        raise ValueError("Use a new expansion evidence version")
    manifests = list(candidate.OUTPUT.glob("*/manifest.json"))
    if len(manifests) != 1:
        raise ValueError("Select exact expanded candidate before recording")
    target = manifests[0].parent
    candidate.verify(target, recompute=False)
    score = candidate.read(target / "scorecard.json")
    if score["splitCounts"] != {"train": 42, "validation": 7, "test": 5, "quarantine": 8} or score["trainingUniqueProxyTokens"] != 919275 or score["validationUniqueProxyTokens"] != 31097:
        raise ValueError("Unexpected expanded corpus measurements")
    decision, packet = admission.verify()
    junit = reports / "llm-task-023-expansion-tests.xml"
    tree = ET.parse(junit).getroot()
    cases = list(tree.iter("testcase"))
    if any(list(tree.iter(tag)) for tag in ("failure", "error", "skipped")):
        raise ValueError("Required expansion tests failed or skipped")
    focused = [case for case in cases if case.get("classname", "").endswith("test_source_expansion")]
    if not focused or not any(case.get("name") == "test_expanded_candidate_reproduces_and_preserves_old_quarantines" for case in cases):
        raise ValueError("Full expanded candidate recomputation evidence missing")
    snapshot_root = ROOT / "backlog_history/llm-task-023-expansion-before"
    for row in read(snapshot_root / "manifest.json")["files"]:
        if hashlib.sha256((snapshot_root / row["backup"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Planning snapshot changed")
    for row in read(reports / "llm-task-024-comparison.json")["candidates"]:
        verify_fixture(AI / row["fixtureRelease"], recompute=True)
    precision = read(reports / "llm-task-023-precision-progress.json")
    verify_precision((ROOT / precision["reviewManifest"]["path"]).parent, recompute=False)
    checks = []
    for command in (
        ["tools/verify_llm_foundation.py", "--check-preserved-history"],
        ["tools/verify_foundation_evaluation_binding.py"],
        ["tools/inventory_llm_sources.py", "verify", "--recompute"],
        ["tools/audit_data_governance.py", "--check"],
        ["tools/verify_runbook.py"],
        ["tools/scan_quality_boundaries.py", "--check"],
        ["tools/snapshot_llm_cleanup.py", "--verify", "../backlog_history/llm-task-016-before-20260906T054449Z"],
        ["tools/expand_pretraining_sources.py", "verify-admission"],
        ["tools/expand_pretraining_sources.py", "require-use", "--release", str(target), "--usage", "tokenizer-fitting"],
        ["tools/expand_pretraining_sources.py", "source-impact", "--release", str(target), "--source-id", "vf-g2-mkdocs-9d65447e"],
    ):
        print("Checking " + " ".join(command), flush=True)
        result = subprocess.run([sys.executable, "-B", *command], cwd=AI, capture_output=True, text=True, encoding="utf-8")
        expected = 2 if "require-use" in command else 0
        if result.returncode != expected:
            raise ValueError("Required check failed: " + " ".join(command) + "\n" + result.stdout + result.stderr)
        output = json.loads(result.stdout)
        if expected == 2 and output.get("code") != "EXPANDED_CORPUS_RELEASE_GATES_UNMET":
            raise ValueError("Expected corpus fitting denial missing")
        if "source-impact" in command:
            if len(output["affectedDocuments"]) != 6 or sum(row["tokens"] for row in output["affectedDocuments"]) != 34053 or output["action"] != "report-only-no-deletion":
                raise ValueError("Unexpected source impact evidence")
            write(impact_path, output)
        checks.append({"command": [".toolchains/gen1/Scripts/python.exe", "-B", *command], "exitCode": result.returncode, "expectedExitCode": expected, "result": output})
    date = datetime.now(timezone(timedelta(hours=5, minutes=30))).date().isoformat()
    summary = (
        "Recorded exact source-input decisions for 24 selected files from five independently pinned repositories: "
        "Learn Go with Tests, the Rust book, MkDocs Material, cJSON and log.c. The 546,522-byte acquisition retains "
        "MIT notices, selected Git blob identities and ancestor license evidence. Only selected bytes/uses are "
        "approved; two repository families are reserved for validation and cannot enter fitting. The new immutable "
        "candidate has 62 records: 42 train, seven validation, five test and eight quarantine. Existing-tokenizer "
        "proxy counts are 919,275 train and 31,097 validation tokens. Technical English training grows from zero "
        "to 108,047 tokens across two independent families; language/code validation is now populated. One new "
        "near-duplicate file is excluded. All seven previous corpus quarantines and 50 earlier domain exclusions "
        "remain excluded; matcher thresholds and old split reservations are unchanged. "
        f"All {len(cases)} regression tests passed, including {len(focused)} source-expansion tests and full candidate "
        "recomputation. Governance, source, runbook, tokenizer and historical checks pass. Task 023 remains "
        "IN_PROGRESS for inherited FreeRTOS/owned-domain input-use decisions, independent math/electronics "
        "validation families, measured corpus resources/budget and admitted input-corpus release. Source-input "
        "approval does not approve corpus fitting: trainingAllowed=false, no model training or activation."
    )
    relative_receipt = receipt_path.relative_to(ROOT).as_posix()
    evidence_paths = [receipt_path, checks_path, impact_path, junit, target / "manifest.json", target / "scorecard.json", admission.DECISION_PATH,
                      AI / decision["acquisitionPath"] / "manifest.json", AI / "docs/LLM_SOURCE_EXPANSION.v1.md"]
    evidence = [path.relative_to(ROOT).as_posix() for path in evidence_paths]
    remaining = [
        "Record exact input-use decisions for surviving inherited FreeRTOS and owned-domain sources, retaining their notices and generator/verification evidence. The excluded Python source remains excluded.",
        "Add independently checked math and electronics validation families; preserve the new technical English/C validation reservations and zero protected/cross-split collisions.",
        "Establish a provisional pilot/input-corpus budget and realistic bounded acquisition/normalization/tokenization/matching measurements; technical prose coverage is not broad language or model-quality acceptance.",
        "Publish an admitted pilot/input corpus only after its applicable source/quality/split/coverage gates pass; then complete task 024 real tokenizer fitting and recounting. Measured production budget/mixture acceptance remains in tasks 031/032.",
        "The precision review remains diagnostic. Any changed matcher for future data requires new precision/recall evidence and a versioned policy; no historical exclusion is automatically reversed.",
    ]
    backlog = read(ROOT / PLANNING[0])
    task = next(row for row in backlog["tasks"] if row["id"] == "LLM-TASK-023")
    if task["status"] != "IN_PROGRESS" or backlog["execution"]["next_task_id"] != "LLM-TASK-023":
        raise ValueError("Unexpected task dependency state")
    task.update(verification_result=summary, verification_status="passed-expanded-candidate-and-exact-source-input-decisions-corpus-release-open", remaining_release_gates=remaining)
    task.setdefault("completed_substeps", []).append({"id": "five-source-input-decisions-and-language-code-validation-expansion", "completed_on": date, "evidence": relative_receipt, "scope": "Exact new source-input permissions and candidate expansion; no corpus/training release"})
    task["evidence_records"].extend(path for path in evidence if path not in task["evidence_records"])
    task["target_paths"].extend(path for path in ("data_governance/source_expansion/", "tools/expand_pretraining_sources.py", "tests/test_source_expansion.py", "docs/LLM_SOURCE_EXPANSION.v1.md") if path not in task["target_paths"])
    backlog["updated_on"] = date
    backlog["scope_of_this_revision"] = "Tasks 015 through 022 completed; task 023 has five new exact source-input decisions and a candidate with populated technical English/C validation. Math/electronics validation, inherited source admission and corpus release remain open; task 024 production tokenizer is pending."
    backlog["current_checkpoint"].update(active_task_id="LLM-TASK-023", active_task_evidence=relative_receipt, active_task_summary=summary)
    previous_manifest = backlog["pretraining_corpus_candidate"]["manifest"]
    backlog["pretraining_corpus_candidate"] = {"manifest": (target / "manifest.json").relative_to(ROOT).as_posix(), "previous_candidate_manifest": previous_manifest,
        "candidate_records": 62, "split_counts": score["splitCounts"], "candidate_train_proxy_tokens": 919275, "candidate_validation_proxy_tokens": 31097, "gen2_corpus_released": False, "training_allowed": False}
    progress = {"evidence": relative_receipt, "sources_with_recorded_new_input_decisions": 5, "selected_files": 24, "technical_english_training_proxy_tokens": 108047,
                "validation_proxy_tokens": 31097, "new_exclusions": 1, "old_exclusions_preserved": True, "corpus_release_approved": False}
    backlog["source_expansion_progress"] = progress
    markdown = (ROOT / PLANNING[1]).read_text(encoding="utf-8")
    marker = "**Continue: LLM-TASK-023 — leakage precision review completed; corpus admission remains open.**"
    if marker not in markdown:
        raise ValueError("Expected precision checkpoint missing")
    markdown = markdown.replace(marker, "**Continue: LLM-TASK-023 — source expansion verified; math/electronics validation and corpus admission remain open.**\n\n" + summary + "\n\n[New source decisions and expanded corpus](Voltforge_AI/docs/LLM_SOURCE_EXPANSION.v1.md) | [Verification](" + relative_receipt + ")\n\n**Previous checkpoint: leakage precision review completed; existing exclusions preserved.**", 1)
    markdown = re.sub(r"^Updated: [\d-]+\.", "Updated: " + date + ".", markdown, count=1, flags=re.MULTILINE)
    start = markdown.index("#### LLM-TASK-023")
    end = markdown.index("#### LLM-TASK-024", start)
    markdown = markdown[:end] + "**Source expansion (" + date + "):** " + summary + "\n\n**Current remaining release work:**\n\n" + "\n".join("- " + item for item in remaining) + "\n\n**Expansion evidence:** [Source decisions](Voltforge_AI/data_governance/source_expansion/admission.v1.json), [expanded corpus](" + (target / "manifest.json").relative_to(ROOT).as_posix() + "), [verification](" + relative_receipt + ").\n\n" + markdown[end:]
    markdown = re.sub(r"Continue with \*\*LLM-TASK-023\*\*\..*$", "Continue with **LLM-TASK-023**. The new exact source-input decisions and populated technical English/C validation are verified. Complete inherited FreeRTOS/owned-domain admission, independent math/electronics validation and the bounded pilot/input-corpus release before production tokenizer fitting. Preserve every existing exclusion and all prior artifacts.", markdown, flags=re.MULTILINE)
    causes = read(ROOT / PLANNING[2])
    causes["updated_on"] = date
    causes["summary"].update(implementation_this_revision=backlog["scope_of_this_revision"], acceptance_limit=summary + " Root-cause statuses remain unchanged.")
    causes["source_expansion_progress"] = progress
    causes["current_checkpoint"].update(active_task_id="LLM-TASK-023", active_task_evidence=relative_receipt)
    causes["evidence_files"].extend(path for path in evidence if path not in causes["evidence_files"])
    analysis = (ROOT / PLANNING[3]).read_text(encoding="utf-8")
    analysis = analysis.replace("## Current work: LLM-TASK-023 leakage precision review", "## Current work: LLM-TASK-023 source expansion and admission\n\n" + summary + "\n\n[Source decisions, measurements and next steps](Voltforge_AI/docs/LLM_SOURCE_EXPANSION.v1.md)\n\n## Previous checkpoint: LLM-TASK-023 leakage precision review", 1)
    write(checks_path, {"status": "passed", "checks": checks, "candidateManifest": descriptor(target / "manifest.json"), "sourceDecision": descriptor(admission.DECISION_PATH),
                        "gen2FixtureArtifactsRecomputedUnchanged": True, "precisionReviewVerifiedUnchanged": True, "fullCandidateRecomputationEvidence": descriptor(junit)})
    receipt = {"schemaVersion": 1, "taskId": "LLM-TASK-023", "substep": "source-expansion-and-exact-input-decisions", "status": "planning-check-pending", "recordedAtUtc": datetime.now(timezone.utc).isoformat(), "summary": summary,
        "tests": {"passed": len(cases), "sourceExpansionTests": len(focused), "failed": 0, "skipped": 0, "junit": descriptor(junit)},
        "candidateManifest": descriptor(target / "manifest.json"), "scorecard": score, "sourceDecision": descriptor(admission.DECISION_PATH),
        "acquisitionManifest": descriptor(AI / decision["acquisitionPath"] / "manifest.json"), "checks": descriptor(checks_path), "sourceImpact": descriptor(impact_path),
        "planningSnapshot": descriptor(snapshot_root / "manifest.json"), "newSourceInputDecisionsRecorded": 5,
        "taskCompleted": False, "corpusReleased": False, "trainingAllowed": False, "matcherChanged": False, "modelActivated": False, "nextTask": "LLM-TASK-023", "remainingWork": remaining}
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
        "docs/LLM_SOURCE_EXPANSION.v1.md", "tools/expand_pretraining_sources.py", "tools/record_source_expansion_progress.py", "tests/test_source_expansion.py")]
    receipt["status"] = "expanded-candidate-and-source-input-decisions-verified-task023-in-progress"
    write(receipt_path, receipt)
    print(json.dumps({"status": receipt["status"], "testsPassed": len(cases), "trainProxyTokens": 919275, "validationProxyTokens": 31097, "nextTask": "LLM-TASK-023"}))


if __name__ == "__main__":
    if sys.argv[1:] == ["snapshot"]:
        snapshot()
    elif sys.argv[1:] == ["record"]:
        record()
    else:
        raise SystemExit("Use snapshot or record")
