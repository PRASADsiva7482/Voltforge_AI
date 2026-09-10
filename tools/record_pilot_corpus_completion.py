"""Snapshot planning and record the fully verified task 023 input release."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import xml.etree.ElementTree as ET
import re

AI = Path(__file__).resolve().parents[1]
ROOT = AI.parent
sys.path.insert(0, str(AI))
from tools.record_source_expansion_progress import PLANNING
from data_governance.pilot_corpus import release as r


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def descriptor(path):
    raw = path.read_bytes()
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def snapshot():
    target = ROOT / "backlog_history/llm-task-023-completion-before"
    target.mkdir(parents=True, exist_ok=False)
    names = (*PLANNING, "Voltforge_AI/docs/LLM_SOURCE_EXPANSION.v1.md", "Voltforge_AI/docs/LLM_GEN2_TOKENIZER_PREPARATION.v1.md")
    entries = []
    for name in names:
        source, backup = ROOT / name, target / name
        backup.parent.mkdir(parents=True, exist_ok=True)
        with backup.open("xb") as stream:
            stream.write(source.read_bytes())
        entries.append({**descriptor(source), "backup": name})
    write(target / "manifest.json", {"taskId": "LLM-TASK-023", "files": entries})
    print(json.dumps({"status": "snapshotted", "files": len(entries)}))


def selected():
    manifests = list(r.OUTPUT.glob("*/manifest.json"))
    if len(manifests) != 1:
        raise ValueError("Select the one verified v1 input corpus")
    target = manifests[0].parent
    manifest = r.verify(target, recompute=False)
    return target, manifest, r.read(target / "scorecard.json")


def archive_prepublication():
    target = ROOT / "backlog_history/llm-task-023-prepublication-path-fix"
    target.mkdir(parents=True, exist_ok=False)
    paths = [*sorted(r.ROOT.glob("*.py")), *sorted(r.ROOT.glob("*.json")), AI / "tools/release_pilot_corpus.py"]
    entries = []
    for path in paths:
        backup = target / path.relative_to(ROOT)
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_bytes(path.read_bytes())
        entries.append(descriptor(path))
    write(target / "manifest.json", {"taskId": "LLM-TASK-023", "status": "prepublication-code-snapshot", "reason": "Measured pipeline passed, but CLI publication rejected a relative resource path. No corpus was published. Preserve this measured implementation before fixing path resolution and obtaining new evidence.", "files": entries, "supersededResourceReceipt": "Voltforge_AI/corpus/pilot-input/resources-v1/c6c15dae1be08757a3a3c23064c472a0b91de6d9f3a46c7f02805fa316c07e93/measurement.json"})


def prepare_docs():
    target, manifest, score = selected()
    resource = r.read(AI / manifest["resourceMeasurement"]["path"])
    relative = target.relative_to(AI).as_posix()
    command = ".toolchains\\gen1\\Scripts\\python.exe -B tools/release_pilot_corpus.py"
    verify_command = command + " verify --release " + relative + " --recompute"
    table = "\n".join(f"| {domain} | {row['train']['uniqueProxyTokens']:,} | {row['train']['families']} | {row['validation']['uniqueProxyTokens']:,} | {row['validation']['families']} |" for domain, row in score["coverage"].items())
    doc = f"""# Task 023 admitted pilot/input corpus v1

This release closes the source, quality, split, coverage and provisional-budget
requirements recorded for task 023. It supplies verified inputs for later owned
tokenizer/scaling work. No tokenizer fit, model training, deployment or activation
is performed by this task. Production corpus and mixture acceptance remain in
tasks 031/032 under the already recorded acceptance staging.

The immutable [manifest](../{relative}/manifest.json),
[scorecard](../{relative}/scorecard.json), [mixture](../{relative}/mixture.json)
and [resource receipt](../{manifest['resourceMeasurement']['path']}) bind the exact
source snapshots, code, policies, exclusions, notices and per-file checksums.
Repeated computation reproduces every shard and manifest using the retained
resource receipt. A new measurement gets its own immutable receipt and release
identity; elapsed time itself is not expected to be deterministic.

## Completed implementation plan

1. Review and record exact surviving FreeRTOS and owned-domain input permissions,
   alongside the five previously reviewed public source repositories.
2. Add independent, predeclared validation families for math and electronics.
3. Reapply normalization, secret checks, full proxy token counting, duplicate and
   protected-content checks, family reservations and source-use restrictions.
4. Enforce the finite input budget, source/domain/language accounting, explicit
   40/30/20/10 scheduling experiment and three-pass repetition limit.
5. Measure the full pipeline in a fresh process at the actual input scale,
   publish immutable corpus bytes, and verify consumer handoffs and rejection
   controls before recording task completion.

## Corpus and token accounting

There are **{score['candidateDocuments']} records: 42 train, 11 validation, five test
and eight quarantine**. All eight previous corpus exclusions and 50 older
owned-domain exclusions remain excluded. No source was edited to match or evade
protected prompts. The unchanged matcher checks {score['protectedDescriptors']:,}
protected descriptors; included protected matches, cross-split duplicate edges
and cross-split lineage keys are all zero.

| Domain | Train unique proxy tokens | Train families | Validation unique proxy tokens | Validation families |
| --- | ---: | ---: | ---: | ---: |
{table}

Total training input is **{score['trainingUniqueProxyTokens']:,} unique proxy tokens**;
validation has **{score['validationUniqueProxyTokens']:,}**. Counts use the owned Gen1
v1.1.0 tokenizer on every document, not a pretrained tokenizer or a guessed
character ratio. Gen2 token counts remain zero until task 024 recounts its own
selected tokenizer. The scorecard also reports per-source, domain, language and
split counts; unique means token positions in admitted deduplicated documents,
not the number of vocabulary types. Actual training exposures are zero.

The provisional input budget was declared before supplement rendering: 900,000
to 1,200,000 unique train proxy tokens, at least 31,000 validation tokens, at
least two train families and one validation family per domain, and at least 200
validation tokens in each domain. This is sufficient to exercise real corpus
input and tokenizer plumbing at this finite scale. It is not a statistical
claim of adequate base-model language learning.

The bounded scheduling experiment exposes exactly **20,000 proxy tokens**:
8,000 language, 6,000 code, 4,000 electronics and 2,000 math. No document exceeds
three passes. Small math/electronics pools constrain larger balanced mixtures;
the remaining language/code bytes are available for tokenizer input. Prefix
sampling and these weights are experimental choices requiring later measured
model results. Infeasible budgets return no schedule and cannot be admitted.

## Source permissions and validation independence

[Exact input decisions](../data_governance/pilot_corpus/source-decisions.v1.json)
approve 58 specific surviving source records. They bind 14 FreeRTOS files with
complete retained MIT grants, 17 original owned-domain survivors, 23 surviving
files from the five expanded repositories and four new original references.
The excluded Python files and MkDocs duplicate receive no new input permission.
Existing candidate and source-evidence artifacts are preserved unchanged.

The original validation references cover polynomial integration, finite
expectation, capacitor charge redistribution and a two-node conductance system.
Their family roles are declared before rendering. Separate exact-arithmetic
identities check each calculation: Simpson weights, equiprobable enumeration,
charge/energy identities, and branch-current/power balances. Corrupted answers
are rejected. These are algorithmically checked reference examples, not human
review or measured component ratings. Their text is original; no evaluation
prompts, private projects, external teacher completions or downloaded weights
are generation inputs.

The permission review records Codex as the reviewer under the user's explicit
task authorization. It asserts no separate owner signature. Source grants,
corpus input admission, training-run approval and model release are distinct.

## Resource evidence and limits

The fresh-process pipeline checked **{score['normalizedInputBytes']:,} normalized
bytes**, with **{score['retainedAcquisitionBytes']:,} retained acquisition bytes**,
in **{resource['elapsedSeconds']:.3f} seconds**, using a process peak of
**{resource['peakResidentBytes']:,} resident bytes**. Limits are 96 documents,
512 KiB per normalized document, 2 MiB normalized input, 8 MiB retained
acquisition, 300 seconds and 2 GiB peak process memory. Both acquisition sources
already enforce bounded selected-file download policies. This measurement
rechecks retained hashes/licenses/tree evidence offline; it does not remeasure
network download latency or imply server/GPU training performance.

Technical English and C dominate this input corpus. The math/electronics pools
are small. No broad conversational, general-knowledge, semantic-contamination,
server-capacity or trained-model adequacy claim is made. Owner backup/restore
acceptance remains required for later tokenizer/model release.

## Reproduce and consume

From `Voltforge_AI`, verify without network access:

```powershell
rtk proxy {verify_command}
rtk proxy {command} require-use --release {relative} --usage tokenizer-fitting-input --split train
rtk proxy {command} source-impact --release {relative} --source-id vf-owned-pilot-reference-v1
```

Rebuild the same bytes using the bound resource measurement:

```powershell
rtk proxy {command} build --resource {manifest['resourceMeasurement']['path']}
```

The Python consumer is `data_governance.pilot_corpus.release.read_inputs`.
It recomputes admission before returning data and binds the bytes again during
handoff. `tokenizer-fitting-input` exposes only train records. Validation/test
require their respective input uses and cannot enter fitting. The bounded
pilot input use returns only the scheduled 20,000 Gen1 proxy token positions,
with document IDs, pass numbers and explicit `trainingRunApproved=false`.
It does not supply a training command. Production-training and runtime uses
are denied. The frozen task-024 fixture adapter still denies corpus fitting;
its eventual integration is task 024 work and was not started here.

For source removal, use the metadata-only impact command, append the affected
source/corpus identity and reason to
`data_governance/pilot-input-revocations.v1.json`, and publish a newly reviewed
replacement release. `require_use` fails closed on relevant revocations or a
missing/malformed ledger. Never erase old audit bytes or mutate an immutable
release; future derived artifacts must retain this manifest as lineage.

Implementation and acceptance controls are in
[release.py](../data_governance/pilot_corpus/release.py),
[policy](../data_governance/pilot_corpus/policy.v1.json) and
[tests](../tests/test_pilot_corpus.py). Historical candidate, precision-review,
source-expansion and Gen2 fixture checks remain part of the regression run.
"""
    (AI / "docs/LLM_PILOT_INPUT_CORPUS.v1.md").write_text(doc, encoding="utf-8", newline="\n")
    runbook_path = AI / "docs/VOLTForge_AI_RUNBOOK.md"
    runbook = runbook_path.read_text(encoding="utf-8")
    start = runbook.index("Task 023 is in progress.")
    end = runbook.index("\n```powershell", start)
    runbook = runbook[:start] + ("Task 023 supplies an [admitted pilot/input corpus](LLM_PILOT_INPUT_CORPUS.v1.md):\n"
        f"42 train / 11 validation / five test / eight quarantine records, with {score['trainingUniqueProxyTokens']:,}\n"
        f"unique train and {score['validationUniqueProxyTokens']:,} validation proxy tokens. Source, family, quality,\n"
        "finite-budget and real-input resource gates pass. All older exclusions remain\n"
        "excluded. Admitted pilot inputs do not approve a training run or production model release.\n"
        "Production corpus/mixture acceptance remains in tasks 031/032. The older\n"
        "[candidate artifacts](LLM_PRETRAINING_CORPUS_CANDIDATE.v1.md) remain historical.\n"
        "Pretraining corpus candidates do not authorize tokenizer fitting or model training.\n") + runbook[end:]
    start = runbook.index("Task 023 now has [five exact")
    end = runbook.index("\n```powershell", start)
    runbook = runbook[:start] + ("The task-023 admitted input release binds exact source-use decisions, retained\n"
        "notices, checked math/electronics validation references and all source/split\n"
        "gates. Use its verifier and train-only handoff; source permission alone is\n"
        "insufficient. Source-input admission does not authorize corpus fitting or model training.\n\n"
        "```powershell\nrtk proxy " + verify_command + "\n```\n") + runbook[end:]
    runbook = runbook.replace("Task 023's corpus is still unadmitted. The 16k/32k fixture ceilings are not trained", "Task 023 now supplies verified inputs through its separate admission API. The\n16k/32k fixture ceilings are not trained")
    runbook = runbook.replace("release approval. Complete source/split admission before adding real corpus fitting.", "release approval. Task 024 must integrate the admitted-input verifier before real corpus fitting.")
    runbook = runbook.replace("Continue with task 023 source admission, independent data expansion and corpus\nrelease gates.", "Task 023 subsequently completed source admission, independent validation and\npilot/input corpus release gates.")
    runbook_path.write_text(runbook, encoding="utf-8", newline="\n")
    contract_path = AI / "docs/runbook-contract.v1.json"
    contract = r.read(contract_path)
    contract["requiredCommands"].insert(0, verify_command)
    contract["requiredClaims"].insert(0, "Admitted pilot inputs do not approve a training run or production model release.")
    contract["requiredSourceFiles"][:0] = ["docs/LLM_PILOT_INPUT_CORPUS.v1.md", "data_governance/pilot_corpus/release.py", "data_governance/pilot_corpus/policy.v1.json", "data_governance/pilot_corpus/source-decisions.v1.json", "data_governance/pilot_corpus/supplement.py", "data_governance/pilot-input-revocations.v1.json", "tools/release_pilot_corpus.py", relative + "/manifest.json"]
    write(contract_path, contract)
    for name in ("LLM_SOURCE_EXPANSION.v1.md", "LLM_PRETRAINING_CORPUS_CANDIDATE.v1.md"):
        path = AI / "docs" / name
        content = path.read_text(encoding="utf-8")
        head, rest = content.split("\n", 1)
        content = head + "\n\nThis document records a historical candidate checkpoint. The current task-023\n[admitted pilot/input release](LLM_PILOT_INPUT_CORPUS.v1.md) closes its input\nadmission gaps while preserving the candidate bytes and original decisions.\n" + rest
        if name == "LLM_SOURCE_EXPANSION.v1.md":
            start = content.index("Continue task 023 with:")
            end = content.index("\nNo model training", start)
            content = content[:start] + "The inherited source decisions, independent math/electronics validation,\nprovisional budget, real-input measurements and admitted corpus release have\nnow been completed in the separate pilot/input release linked above. These\nsource-expansion artifacts remain unchanged and candidate-only.\n" + content[end:]
        path.write_text(content, encoding="utf-8", newline="\n")
    path = AI / "docs/LLM_GEN2_TOKENIZER_PREPARATION.v1.md"
    content = path.read_text(encoding="utf-8")
    content = content.replace("1. Complete task 023: admitted pilot/input corpus with independent language,\n   code/domain breadth and validation families; preserve protected exclusions.", "1. Task 023 input dependency is now complete; see the [admitted input release](LLM_PILOT_INPUT_CORPUS.v1.md). No new task-024 implementation is included in that completion.")
    content = content.replace("exists. Source admission remains task 023's responsibility.", "exists. Task 023 now supplies a separate verified input API; integrating it here remains task 024 work.")
    path.write_text(content, encoding="utf-8", newline="\n")
    attributes = AI / ".gitattributes"
    with attributes.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write("data_governance/pilot_corpus/** -text\ntools/release_pilot_corpus.py -text\n")
    print(json.dumps({"status": "prepared-release-docs-and-runbook-contract", "manifest": relative + "/manifest.json"}))


def run_checks():
    checks = []
    commands = [
        ["tools/verify_llm_foundation.py", "--check-preserved-history"],
        ["tools/verify_foundation_evaluation_binding.py"],
        ["tools/inventory_llm_sources.py", "verify", "--recompute"],
        ["tools/audit_data_governance.py", "--check"],
        ["tools/verify_runbook.py"],
        ["tools/scan_quality_boundaries.py", "--check"],
        ["tools/snapshot_llm_cleanup.py", "--verify", "../backlog_history/llm-task-016-before-20260906T054449Z"],
    ]
    for command in commands:
        result = subprocess.run([sys.executable, "-B", *command], cwd=AI, capture_output=True, text=True, encoding="utf-8", timeout=300)
        try:
            output = json.loads(result.stdout)
        except json.JSONDecodeError:
            output = result.stdout.strip()
        checks.append({"command": command, "exitCode": result.returncode, "output": output})
        print(json.dumps({"command": command[0], "exitCode": result.returncode}), flush=True)
        if result.returncode:
            raise ValueError("Acceptance check failed: " + result.stdout + result.stderr)
    from model.gen2_tokenizer.release import verify_fixture
    from data_governance.leakage_review.evaluation import verify as verify_precision
    fixtures = sorted((AI / "model/tokenizers/fixtures/gen2/v1").glob("*/manifest.json"))
    if len(fixtures) != 2:
        raise ValueError("Gen2 fixture inventory changed")
    for path in fixtures:
        verify_fixture(path.parent, recompute=True)
    precision = AI / "corpus/leakage-reviews/v1/72e42051c6187257f2e5954a9e0056e006915ded4b9c168a2e7f2c1b426be5ec"
    verify_precision(precision, recompute=False)
    target, manifest, score = selected()
    r.check_revocations(manifest)
    write(AI / "evaluation/reports/llm-task-023-completion-checks.json", {"status": "passed", "checks": checks, "gen2FixturesRecomputedUnchanged": [descriptor(path) for path in fixtures],
          "precisionReviewVerifiedUnchanged": descriptor(precision / "manifest.json"), "corpusManifest": descriptor(target / "manifest.json"), "scorecard": descriptor(target / "scorecard.json"), "resourceMeasurement": descriptor(AI / manifest["resourceMeasurement"]["path"])})


def complete():
    target, manifest, score = selected()
    reports = AI / "evaluation/reports"
    junit = reports / "llm-task-023-completion-tests.xml"
    suite = ET.parse(junit).getroot()
    cases = list(suite.iter("testcase"))
    if not cases or any(list(suite.iter(tag)) for tag in ("failure", "error", "skipped")):
        raise ValueError("Required corpus regression tests failed or skipped")
    focused = [case for case in cases if case.get("classname", "").endswith("test_pilot_corpus")]
    if len(focused) < 60:
        raise ValueError("Missing complete pilot corpus regression coverage")
    checks_path = reports / "llm-task-023-completion-checks.json"
    checks = r.read(checks_path)
    fast_path = reports / "llm-task-023-completion-cross-repository-quality.json"
    fast = r.read(fast_path)
    if checks["status"] != "passed" or checks["corpusManifest"] != descriptor(target / "manifest.json") or fast["scorecard"]["summary"]["failed"] != 0:
        raise ValueError("Required acceptance or cross-repository checks failed")
    resource = r.read(AI / manifest["resourceMeasurement"]["path"])
    receipt_path = reports / "llm-task-023-completion.json"
    if receipt_path.exists():
        raise ValueError("Completion evidence is write-once; choose a new revision")
    summary = (f"Completed task 023's recorded pilot/input-corpus acceptance: immutable release with 66 records (42 train, 11 validation, five test, eight quarantine), {score['trainingUniqueProxyTokens']:,} unique train and {score['validationUniqueProxyTokens']:,} validation proxy tokens. "
        "All four domains have independent train/validation families. Exact permissions cover 58 source records, including surviving inherited FreeRTOS/owned-domain inputs and four independently checked new references. "
        "All eight older corpus quarantines and 50 earlier domain exclusions remain excluded; the unchanged matcher checks 1,005 protected descriptors with zero included protected/cross-split collisions. "
        f"The full real-input pipeline recomputes {score['normalizedInputBytes']:,} normalized bytes in {resource['elapsedSeconds']:.3f} seconds with {resource['peakResidentBytes']:,} peak resident bytes. "
        "The provisional 900,000-1,200,000 unique-token input budget and bounded 20,000-exposure 40/30/20/10 plan pass, with at most three passes per document. "
        f"All {len(cases)} regression tests passed ({len(focused)} pilot-input controls), along with governance, runbook, historical preservation and {fast['scorecard']['summary']['passed']} fast cross-repository checks. "
        "Verified consumers deny missing/revoked rights, changed shards, split violations, contaminated inputs and excessive exposure schedules before returning data. "
        "No tokenizer fitting, model training or activation occurred. Production data/mixture acceptance remains mandatory in tasks 031/032; no upcoming task was implemented.")
    evidence = [descriptor(path)["path"] for path in (target / "manifest.json", target / "scorecard.json", target / "mixture.json", target / "leakage.json", target / "reference-checks.json",
        AI / manifest["resourceMeasurement"]["path"], r.DECISION, AI / "docs/LLM_PILOT_INPUT_CORPUS.v1.md", junit, checks_path, fast_path)]
    evidence += [receipt_path.relative_to(ROOT).as_posix(), "backlog_history/llm-task-023-completion-before/manifest.json", "backlog_history/llm-task-023-prepublication-path-fix/manifest.json"]
    backlog = r.read(ROOT / PLANNING[0])
    task = next(item for item in backlog["tasks"] if item["id"] == "LLM-TASK-023")
    task.update(status="COMPLETED", completed_on="2026-09-09", implementation_state="admitted-pilot-input-corpus-released-and-reproduced", verification_status="passed-all-recorded-task023-input-acceptance", verification_result=summary, remaining_release_gates=[])
    task["evidence_records"].extend(name for name in evidence if name not in task["evidence_records"])
    task["target_paths"] += ["data_governance/pilot_corpus/", "corpus/pilot-input/", "tools/release_pilot_corpus.py", "tests/test_pilot_corpus.py", "docs/LLM_PILOT_INPUT_CORPUS.v1.md"]
    task["completed_substeps"].append({"id": "admitted-pilot-input-corpus-and-complete-acceptance", "completed_on": "2026-09-09", "evidence": receipt_path.relative_to(ROOT).as_posix(), "scope": "All recorded task023 input-corpus requirements; no task024/031/032 implementation"})
    task["partition_scale_gate"] = "Closed for the declared finite input release: indexed matcher plus exhaustive/adversarial controls and full 66-record/1,005-guard input pipeline measurement. Larger datasets require new resource/quality evidence; limits are not raised by this completion."
    task["domain_candidate_handoff"] = "Closed for this input release: exact inherited rights reviewed, independent validation added, all 50 old domain exclusions and eight corpus exclusions preserved. Diagnostic precision review never readmitted quarantined data or changed matcher thresholds. Firmware runtime/broader defects remain separately scoped follow-ups."
    task["acceptance_limits"] = score["limits"] + ["Actual training exposures and released Gen2 token counts remain zero; owner backup/restore and run approval are not asserted."]
    next_task = next(item for item in backlog["tasks"] if item["id"] == "LLM-TASK-024")
    completed_dependency_gate = "Complete LLM-TASK-023 admitted pilot/input corpus, including independent language and validation coverage and source-use rights."
    next_task["remaining_release_gates"] = [item for item in next_task["remaining_release_gates"] if item != completed_dependency_gate]
    next_task["blocked_by"] = [item for item in next_task.get("blocked_by", []) if item != "LLM-TASK-023"]
    next_task["corpus_dependency_handoff"] = {"status": "input-dependency-completed", "task_id": "LLM-TASK-023", "evidence": receipt_path.relative_to(ROOT).as_posix(), "implementation_started_this_revision": False}
    backlog["status_summary"] = dict(Counter(item["status"] for item in backlog["tasks"]))
    backlog["execution"]["next_task_id"] = "LLM-TASK-024"
    backlog["execution"]["current_user_scope"] = "Task 023 completed; stop here per user instruction. Task 024 is the next ready item, not started in this completion."
    for milestone in backlog["milestones"]:
        if milestone["id"] == "M1":
            milestone["status"] = "COMPLETED"
    backlog["current_checkpoint"] = {"completed_task_id": "LLM-TASK-023", "evidence": receipt_path.relative_to(ROOT).as_posix(), "summary": summary, "active_artifact_id": None, "neural_ready": False,
                                     "active_task_id": None, "next_ready_task_id": "LLM-TASK-024", "user_scope": "Stop after task023; no upcoming implementation"}
    backlog["admitted_pilot_input_corpus"] = {"manifest": (target / "manifest.json").relative_to(ROOT).as_posix(), "split_counts": score["splitCounts"], "train_unique_proxy_tokens": score["trainingUniqueProxyTokens"], "validation_unique_proxy_tokens": score["validationUniqueProxyTokens"], "input_use_allowed": True, "training_run_approved": False, "production_training_allowed": False, "evidence": receipt_path.relative_to(ROOT).as_posix()}
    # Resolve dependency metadata only; task024 implementation stays unchanged.
    markdown = (ROOT / PLANNING[1]).read_text(encoding="utf-8")
    start = markdown.index("**41 active tasks:")
    end = markdown.index("**Previous checkpoint: leakage precision", start)
    markdown = markdown[:start] + ("**41 active tasks: 9 completed, 1 in progress (LLM-TASK-024), 31 TODO. No trained conversational LLM release is accepted.**\n\n"
        "**Completed: LLM-TASK-023 — all recorded pilot/input-corpus acceptance checks pass.** Work stops at this task as requested. Task 024 is the next ready item; no upcoming task was implemented.\n\n"
        + summary + "\n\n[Corpus release and completed plan](Voltforge_AI/docs/LLM_PILOT_INPUT_CORPUS.v1.md) | [Acceptance evidence](Voltforge_AI/evaluation/reports/llm-task-023-completion.json)\n\n") + markdown[end:]
    start = markdown.index("#### LLM-TASK-023 —")
    end = markdown.index("#### LLM-TASK-024 —", start)
    section = (f"#### {task['id']} — {task['title']}\n\n**Status:** COMPLETED. **Priority:** {task['priority']}. **Depends on:** {', '.join(task['depends_on'])}.\n\n"
        "**Work:**\n\n" + "\n".join("- " + item for item in task["work"]) + "\n\n**Definition of done:**\n\n" + "\n".join("- " + item for item in task["definition_of_done"])
        + "\n\n**Required evidence:** " + "; ".join(task["required_evidence"]) + ".\n\n**Verified completion (2026-09-09):** " + summary
        + "\n\n**Task-023 remaining acceptance work:** none under its recorded input-corpus scope. Production-scale budget and measured mixtures remain in tasks 031/032, unchanged.\n\n"
        "**Evidence:** [Immutable corpus](" + (target / "manifest.json").relative_to(ROOT).as_posix() + "), [coverage and full plan](Voltforge_AI/docs/LLM_PILOT_INPUT_CORPUS.v1.md), [acceptance receipt](Voltforge_AI/evaluation/reports/llm-task-023-completion.json). All previous evidence paths remain in the canonical JSON task and planning snapshots.\n\n"
        "### M2: Owned tokenizer and decoder architecture\n\n")
    markdown = markdown[:start] + section + markdown[end:]
    markdown = markdown.replace("- " + completed_dependency_gate + "\n", "")
    marker = "**Verified preparation (2026-09-08):** Implemented a separate owned Gen2 byte-BPE codec"
    markdown = markdown.replace(marker, "**Dependency handoff (2026-09-09):** Task 023 input admission is complete. The following preparation report is historical; no new task-024 implementation occurred.\n\n" + marker, 1)
    markdown = markdown.replace("| M1 | Approved diverse corpus and leakage prevention | LLM-TASK-019 through LLM-TASK-023 | IN_PROGRESS |", "| M1 | Approved diverse corpus and leakage prevention | LLM-TASK-019 through LLM-TASK-023 | COMPLETED |")
    causes = r.read(ROOT / PLANNING[2])
    causes["summary"]["implementation_this_revision"] = "Tasks 015 through 023 completed. Task 023 releases admitted pilot/input data with all domain validation and real-input resource evidence. No subsequent implementation or trained-model acceptance is claimed."
    causes["summary"]["acceptance_limit"] = summary + " Root-cause statuses remain unchanged; linked model/runtime acceptance is still open."
    causes["evidence_files"].extend(name for name in evidence if name not in causes["evidence_files"])
    causes["pilot_input_corpus_completion"] = {"task_id": "LLM-TASK-023", "status": "COMPLETED", "evidence": receipt_path.relative_to(ROOT).as_posix(), "manifest": (target / "manifest.json").relative_to(ROOT).as_posix(), "training_run_approved": False}
    analysis = (ROOT / PLANNING[3]).read_text(encoding="utf-8")
    start = analysis.index("## Current work: LLM-TASK-023")
    end = analysis.index("## Previous checkpoint:", start)
    analysis = analysis[:start] + "## Completed: LLM-TASK-023 admitted pilot/input corpus\n\n" + summary + "\n\n[Corpus release and acceptance scope](Voltforge_AI/docs/LLM_PILOT_INPUT_CORPUS.v1.md) | [Verification](Voltforge_AI/evaluation/reports/llm-task-023-completion.json)\n\n" + analysis[end:]
    receipt = {"schemaVersion": 1, "taskId": "LLM-TASK-023", "status": "planning-verification-pending", "recordedAtUtc": datetime.now(timezone.utc).isoformat(), "summary": summary,
        "taskCompleted": True, "inputCorpusReleased": True, "trainingRunApproved": False, "productionTrainingAllowed": False, "actualTrainingExposures": 0, "modelActivated": False, "upcomingTasksImplemented": [],
        "corpusManifest": descriptor(target / "manifest.json"), "scorecard": score, "resourceMeasurement": resource,
        "tests": {"passed": len(cases), "pilotInputTests": len(focused), "failed": 0, "skipped": 0, "junit": descriptor(junit)}, "checks": descriptor(checks_path), "crossRepositoryQuality": descriptor(fast_path),
        "planningSnapshot": descriptor(ROOT / "backlog_history/llm-task-023-completion-before/manifest.json"), "remainingTask023Work": [], "nextReadyTask": "LLM-TASK-024", "stopAfterTask023": True}
    write(receipt_path, receipt)
    write(ROOT / PLANNING[0], backlog)
    (ROOT / PLANNING[1]).write_text(markdown, encoding="utf-8", newline="\n")
    write(ROOT / PLANNING[2], causes)
    (ROOT / PLANNING[3]).write_text(analysis, encoding="utf-8", newline="\n")
    result = subprocess.run([sys.executable, "-B", "tools/verify_llm_backlogs.py"], cwd=AI, capture_output=True, text=True, encoding="utf-8")
    if result.returncode:
        raise ValueError("Planning verifier failed: " + result.stdout + result.stderr)
    receipt["planningCheck"] = json.loads(result.stdout)
    receipt["status"] = "completed-all-recorded-task023-pilot-input-acceptance"
    receipt["planningAndDocumentationBindings"] = [descriptor(ROOT / name) for name in PLANNING] + [descriptor(AI / name) for name in ("docs/LLM_PILOT_INPUT_CORPUS.v1.md", "docs/LLM_SOURCE_EXPANSION.v1.md", "docs/LLM_GEN2_TOKENIZER_PREPARATION.v1.md", "tests/test_pilot_corpus.py", "tools/record_pilot_corpus_completion.py")]
    write(receipt_path, receipt)
    print(json.dumps({"status": receipt["status"], "testsPassed": len(cases), "fastChecksPassed": fast["scorecard"]["summary"]["passed"], "nextReadyTask": "LLM-TASK-024", "stopAfterTask023": True}))


if __name__ == "__main__":
    if sys.argv[1:] == ["snapshot"]:
        snapshot()
    elif sys.argv[1:] == ["prepare-docs"]:
        prepare_docs()
    elif sys.argv[1:] == ["archive-prepublication"]:
        archive_prepublication()
    elif sys.argv[1:] == ["checks"]:
        run_checks()
    elif sys.argv[1:] == ["complete"]:
        complete()
    else:
        raise SystemExit("Use snapshot or prepare-docs")
