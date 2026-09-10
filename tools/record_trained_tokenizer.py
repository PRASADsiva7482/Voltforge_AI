"""Snapshot and record task024 corpus-trained tokenizer acceptance."""
from pathlib import Path
import hashlib
import json
import sys
import subprocess
from collections import Counter
from datetime import datetime, timezone
import re
import xml.etree.ElementTree as ET

AI = Path(__file__).resolve().parents[1]
ROOT = AI.parent
sys.path.insert(0, str(AI))
from model.gen2_tokenizer_training import release as r, evaluation
from model.gen2_tokenizer_training.corpus import CORPUS, inputs


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def descriptor(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}


def snapshot():
    target = ROOT / "backlog_history/llm-task-024-trained-before"
    target.mkdir(parents=True, exist_ok=False)
    names = ["VOLTFORGE_LOCAL_LLM_BUILD_BACKLOG.json", "VOLTFORGE_LOCAL_LLM_BUILD_BACKLOG.md", "AI_CHAT_ROOT_CAUSE_BACKLOG.json", "AI_CHAT_ROOT_CAUSE_ANALYSIS.md",
             "Voltforge_AI/docs/VOLTForge_AI_RUNBOOK.md", "Voltforge_AI/docs/runbook-contract.v1.json", "Voltforge_AI/docs/LLM_GEN2_TOKENIZER_PREPARATION.v1.md", "Voltforge_AI/docs/LLM_PILOT_INPUT_CORPUS.v1.md", "Voltforge_AI/.gitattributes"]
    entries = []
    for name in names:
        source, backup = ROOT / name, target / name
        backup.parent.mkdir(parents=True, exist_ok=True)
        with backup.open("xb") as stream:
            stream.write(source.read_bytes())
        entries.append({**descriptor(source), "backup": name})
    write(target / "manifest.json", {"taskId": "LLM-TASK-024", "files": entries})
    print(json.dumps({"status": "snapshotted", "files": len(entries)}))


def selected():
    version = r.read_json(r.VERSION_RECORD)
    path = (AI / version["releaseManifest"]["path"]).parent
    tokenizer, manifest = r.load(path)
    report_path = AI / manifest["comparison"]["path"]
    return tokenizer, manifest, path, r.read_json(report_path)


def recount():
    tokenizer, manifest, path, comparison = selected()
    chosen = next(item for item in comparison["candidates"] if item["targetVocabSize"] == tokenizer.target_vocab_size)
    # Training and validation counts are already bound and independently replayed
    # in the comparison. Count the test split only after vocabulary selection.
    test = inputs.read_inputs(CORPUS, "test-input", split="test")
    counts = {"train": chosen["train"], "validation": chosen["validation"], "test": evaluation.record_metrics(tokenizer, test)}
    report = {"schemaVersion": 1, "taskId": "LLM-TASK-024", "status": "passed-admitted-corpus-gen2-token-recount", "tokenizerManifest": r.binding(path / "manifest.json"), "corpusManifest": r.binding(CORPUS / "manifest.json"),
              "comparison": manifest["comparison"], "sourceFingerprints": r.fingerprints(), "producer": descriptor(Path(__file__)), "splits": counts,
              "trainingUniqueGen2Tokens": counts["train"]["totalTokens"], "validationUniqueGen2Tokens": counts["validation"]["totalTokens"], "testUniqueGen2Tokens": counts["test"]["totalTokens"],
              "testUsedForFittingOrSelection": False, "testCountedAfterTokenizerFreeze": True, "quarantineUsed": False, "actualModelTrainingExposures": 0,
              "trainValidationCountBasis": "reuse exact selected candidate counts from verified comparison; test independently encoded only after the final tokenizer version was registered"}
    report = r.identity(report)
    target = AI / "corpus/token-counts/gen2/v1" / report["contentId"]
    with r.build_lock(target.parent / ".build.lock"):
        r.write_immutable(target / "report.json", r.data(report))
    print(json.dumps({"status": report["status"], "path": target.relative_to(AI).as_posix(), "trainTokens": report["trainingUniqueGen2Tokens"], "validationTokens": report["validationUniqueGen2Tokens"], "testTokens": report["testUniqueGen2Tokens"]}))


def supporting_report(namespace):
    paths = list((AI / namespace).glob("*/report.json"))
    if len(paths) != 1:
        raise ValueError("Expected one exact accepted report in " + namespace)
    value = read(paths[0])
    if r.identity({key: item for key, item in value.items() if key != "contentId"}) != value or paths[0].parent.name != value["contentId"]:
        raise ValueError("Supporting report identity changed")
    if value["sourceFingerprints"] != r.fingerprints():
        raise ValueError("Supporting implementation changed")
    return paths[0], value


def prepare_docs():
    tokenizer, manifest, path, comparison = selected()
    count_path, counts = supporting_report("corpus/token-counts/gen2/v1")
    integration_path, integration = supporting_report("model/tokenizers/gen2/integration/v1")
    relative = path.relative_to(AI).as_posix()
    command = ".toolchains\\gen1\\Scripts\\python.exe -B tools/train_gen2_tokenizer.py"
    verify_command = command + " verify --release " + relative + " --recompute"
    table = "\n".join(f"| {candidate['targetVocabSize']:,} | {candidate['actualVocabSize']:,} | {candidate['train']['totalTokens']:,} | {candidate['validation']['totalTokens']:,} |" for candidate in comparison["candidates"])
    metric_table = "\n".join(f"| {category} | {comparison['baseline']['metrics'][category]['tokens']} | {comparison['candidates'][0]['metrics'][category]['tokens']} | {comparison['candidates'][1]['metrics'][category]['tokens']} |" for category in comparison["baseline"]["metrics"])
    text = f"""# Task 024 owned corpus-trained tokenizer release

The selected **vfdlm-g2-byte-bpe 0.1.0** tokenizer has **{tokenizer.vocab_size:,}
entries**. It was fitted from raw bytes of the task-023 admitted training split:
42 documents, 1,475,271 bytes. It imports no pretrained vocabulary, merges or
weights. Both requested candidate targets were reached from the same exact
training data. All old Gen1 tokenizers and Gen2 fixture files remain immutable.

[Release manifest](../{relative}/manifest.json) |
[Candidate comparison](../{manifest['comparison']['path']}) |
[Gen2 corpus counts](../{count_path.relative_to(AI).as_posix()}) |
[Real model/context integration](../{integration_path.relative_to(AI).as_posix()})

## Completed task plan

1. Add an evidence-verifying adapter for task 023's exact admitted corpus;
   recompute source/split admission before merge fitting and enforce byte limits.
2. Fit both 16,384 and 32,768 ceilings in fresh processes, retaining exact train
   lineage, code, policy, vocabulary, merges and resource measurements.
3. Freeze both candidate artifacts before reading permitted validation data;
   compare per-domain compression, fertility, exact round trips, SI/Unicode,
   long-code cost, literal markers and codec throughput with the 3,072 baseline.
4. Apply the predeclared vocabulary selection rule, publish a write-once 0.1.0
   version record and immutable release, and recount the admitted corpus.
5. Feed compiled IDs into real randomly initialized decoder embeddings and
   logits with the same tokenizer/template descriptor. Check mismatches before
   forward execution, and preserve all historical release evidence.

## Candidate comparison and selection

| Target entries | Actual entries | Train tokens | Validation tokens |
| ---: | ---: | ---: | ---: |
{table}

The policy selects 32k only when its equal-domain validation token reduction
over 16k is at least 8%, with no domain regressing by more than 10%. Otherwise
it retains 16k. The measured macro reduction is
**{comparison['selection']['macroValidationReductionOf32768Versus16384']:.2%}**;
the selected target is **{comparison['selection']['targetVocabSize']:,}**.
This criterion selects a vocabulary for this input corpus, not an optimal
model architecture or a demonstrated language-quality result. Validation text
is measured only after both merge sets are frozen; it never enters fitting.
Test and sealed acceptance data do not select the tokenizer. Source/leakage
verifiers still inspect their protected references as required by corpus gates.

| Public measurement category | Gen1 3,072 tokens | Gen2 16,384 tokens | Gen2 32,768 tokens |
| --- | ---: | ---: | ---: |
{metric_table}

These public implementation measurements supplement actual train/validation
counts. Whitespace fertility is a proxy. The historical baseline had different
fitting data and is diagnostic. The report records three-iteration codec
throughput; those host timings establish no server serving capacity.

## Token contract, framing and model integration

The existing 16 special IDs, all 256 byte values, no-normalization UTF-8 with
surrogatepass, and `vfdlm-g2-chat-v1` template version 1.0.0 are retained.
Literal `<|assistant|>` and other marker-looking content use byte tokens;
only trusted role framing inserts structural IDs. Exact content marker costs
are measured rather than approximated or rewritten. All byte values, SI text,
Unicode, surrogate cases and incremental decoding are covered by checks.

`context_compiler/gen2_release.py` reuses the frozen ID compiler and passes its
exact binding to `model/gen2_tokenizer_training/model_boundary.py`. The latter
constructs a real decoder from the existing owned Gen1 tensor mathematics under
a new Gen2 tokenizer binding. The integration profile has
**{integration['model']['parameterCount']:,} actual parameters**, embeddings of
shape **{integration['model']['embeddingShape']}**, and logits of shape
**{integration['model']['logitShape']}**. Cached final-token logits match the
full-prefix result. A clean process imports pinned PyTorch
**{integration['nativeRuntime']['torchVersion']}** before tensor allocation.

This proves model/context consumption of the selected version, beyond a
dictionary-only interface check. The finite random-weight integration profile
does not complete task 025's configurable architecture/profile acceptance or
claim learned 4k context. No model optimizer step, checkpoint load/save or
application activation occurs. Existing application routing remains unchanged.

## Corpus recount and resource limits

The selected tokenizer counts **{counts['trainingUniqueGen2Tokens']:,} unique train
tokens**, **{counts['validationUniqueGen2Tokens']:,} validation tokens** and
**{counts['testUniqueGen2Tokens']:,} test tokens**. Train/validation counts reuse
the selected comparison's exact checked records; test text is encoded only
after final vocabulary registration, with no selection feedback. Quarantines
remain excluded. These are token positions in deduplicated documents, not
distinct vocabulary types. Actual model-training exposures remain zero.

Each candidate receipt records measured fit time and process peak memory.
Limits were declared before fitting: at most 96 train documents, 2 MiB train
bytes, 300 fit seconds, 2 GiB peak process memory, 32 MiB vocabulary bytes and
minimum pair frequency two. Source lineage and byte/token counts are immutable.
Production data/compute budget selection remains in tasks 031/032.

## Release and retention boundary

The 0.1.0 record approves this immutable offline tokenizer artifact for later
training-input integration. It does not approve a model training run, serving,
deployment or external-retention acceptance. The inherited VFAI-FU-015 policy
is an accepted-for-later Gen1 synthetic-history follow-up; the foundation model
release policy requires backup/restore in integrated-assistant acceptance.
Owner-approved encrypted external backup/restore requirements remain intact
for deployment in tasks 041/054/055. No owner signature, backup or restore
receipt is fabricated, and the historical retention policy is not changed.

The authoritative task-024 definition requires the immutable tokenizer plus
model/context version consumption. Those are the acceptance criteria here;
deployment-only gates do not turn an offline tokenizer fit into a serving
approval. The existing byte-codec fixture runtime gate stays closed.

## Commands

From `Voltforge_AI`:

```powershell
rtk proxy {verify_command}
rtk proxy {command} integration --release {relative}
rtk proxy .toolchains\\gen1\\Scripts\\python.exe -B -m pytest -q tests/test_trained_gen2_tokenizer.py tests/test_gen2_tokenizer.py
```

New candidate revisions require new immutable namespaces/policies; do not rerun
fits into the accepted candidate pair or overwrite the 0.1.0 version record.
Changing bytes, source permissions, split ownership, merge rules, special IDs,
template or model vocabulary dimensions fails verification. Source revocation
continues to block the tokenizer's corpus lineage through task 023's verifier.
"""
    (AI / "docs/LLM_TRAINED_GEN2_TOKENIZER.v1.md").write_text(text, encoding="utf-8", newline="\n")
    old_doc = AI / "docs/LLM_GEN2_TOKENIZER_PREPARATION.v1.md"
    text = old_doc.read_text(encoding="utf-8")
    start = text.index("Task **LLM-TASK-024 is IN_PROGRESS**.")
    end = text.index("\n## Implemented contract", start)
    text = text[:start] + "Task 024's [corpus-trained tokenizer release](LLM_TRAINED_GEN2_TOKENIZER.v1.md)\nnow supplies the selected 0.1.0 artifact and real model/context integration.\nThis document preserves the earlier implementation-fixture checkpoint; its\nfixture artifacts remain unchanged and cannot stand in for the trained release.\n" + text[end:]
    start = text.index("## Remaining acceptance work")
    text = text[:start] + "## Completed follow-through\n\nThe admitted-corpus adapter, real candidate fits and comparison, immutable\nrelease, corpus recount and real tensor/context binding have been implemented\nin the separate release linked above. The frozen fixture-only `check-corpus`\ncommand continues to reject corpus fitting; use `train_gen2_tokenizer.py` for\nthe verified corpus-trained artifact. No legacy gate or artifact was rewritten.\n"
    old_doc.write_text(text, encoding="utf-8", newline="\n")
    pilot_doc = AI / "docs/LLM_PILOT_INPUT_CORPUS.v1.md"
    pilot_text = pilot_doc.read_text(encoding="utf-8")
    first, remaining = pilot_text.split("\n", 1)
    pilot_doc.write_text(first + "\n\nFollow-through: task 024 now supplies the [owned 0.1.0 tokenizer and Gen2 token recount](LLM_TRAINED_GEN2_TOKENIZER.v1.md). This task-023 document retains its original proxy counts and acceptance scope.\n" + remaining, encoding="utf-8", newline="\n")
    runbook_path = AI / "docs/VOLTForge_AI_RUNBOOK.md"
    book = runbook_path.read_text(encoding="utf-8")
    start = book.index("Task 024 adds [Gen2 tokenizer preparation]")
    end = book.index("\n```powershell", start)
    book = book[:start] + (f"Task 024 now supplies a [corpus-trained owned tokenizer](LLM_TRAINED_GEN2_TOKENIZER.v1.md)\nwith {tokenizer.vocab_size:,} entries, immutable 0.1.0 lineage and actual decoder/context\nversion checks. The older [fixture preparation](LLM_GEN2_TOKENIZER_PREPARATION.v1.md)\nremains historical. The trained tokenizer does not approve model training, serving or external-retention acceptance.\n\n"
        "```powershell\nrtk proxy " + verify_command + "\n```\n") + book[end:]
    book = book.replace("release approval. Task 024 must integrate the admitted-input verifier before real corpus fitting.", "release approval. The separate corpus-trained release uses the admitted-input verifier.")
    runbook_path.write_text(book, encoding="utf-8", newline="\n")
    contract_path = AI / "docs/runbook-contract.v1.json"
    contract = read(contract_path)
    contract["requiredCommands"].insert(0, verify_command)
    contract["requiredClaims"].insert(0, "The trained tokenizer does not approve model training, serving or external-retention acceptance.")
    contract["requiredSourceFiles"][:0] = ["docs/LLM_TRAINED_GEN2_TOKENIZER.v1.md", "model/gen2_tokenizer_training/policy.v1.json", "model/gen2_tokenizer_training/corpus.py", "model/gen2_tokenizer_training/fitting.py", "model/gen2_tokenizer_training/release.py", "model/gen2_tokenizer_training/evaluation.py", "model/gen2_tokenizer_training/model_boundary.py", "model/gen2_tokenizer_training/integration.py", "context_compiler/gen2_release.py", "tools/train_gen2_tokenizer.py", relative + "/manifest.json", r.VERSION_RECORD.relative_to(AI).as_posix()]
    write(contract_path, contract)
    with (AI / ".gitattributes").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write("model/gen2_tokenizer_training/** -text\nmodel/tokenizers/gen2/** -text\ncontext_compiler/gen2_release.py -text\ntools/train_gen2_tokenizer.py -text\n")
    print(json.dumps({"status": "prepared-trained-tokenizer-docs", "vocabSize": tokenizer.vocab_size}))


def checks():
    results = []
    commands = [["tools/verify_llm_foundation.py", "--check-preserved-history"], ["tools/verify_foundation_evaluation_binding.py"], ["tools/inventory_llm_sources.py", "verify", "--recompute"],
                ["tools/audit_data_governance.py", "--check"], ["tools/verify_runbook.py"], ["tools/scan_quality_boundaries.py", "--check"],
                ["tools/snapshot_llm_cleanup.py", "--verify", "../backlog_history/llm-task-016-before-20260906T054449Z"]]
    for command in commands:
        result = subprocess.run([sys.executable, "-B", *command], cwd=AI, capture_output=True, text=True, encoding="utf-8", timeout=300)
        try:
            output = json.loads(result.stdout)
        except json.JSONDecodeError:
            output = result.stdout.strip()
        results.append({"command": command, "exitCode": result.returncode, "output": output})
        print(json.dumps({"command": command[0], "exitCode": result.returncode}), flush=True)
        if result.returncode:
            raise ValueError("Required check failed: " + result.stdout + result.stderr)
    from model.gen2_tokenizer.release import verify_fixture
    fixtures = list((AI / "model/tokenizers/fixtures/gen2/v1").glob("*/manifest.json"))
    if len(fixtures) != 2:
        raise ValueError("Historical fixture inventory changed")
    for path in fixtures:
        verify_fixture(path.parent, recompute=True)
    inputs.verify(CORPUS, recompute=False)
    _, _, path, _ = selected()
    count_path, count_report = supporting_report("corpus/token-counts/gen2/v1")
    integration_path, integration = supporting_report("model/tokenizers/gen2/integration/v1")
    if count_report["producer"] != descriptor(Path(__file__)):
        raise ValueError("Recount producer changed")
    write(AI / "evaluation/reports/llm-task-024-trained-checks.json", {"status": "passed", "checks": results, "historicalFixturesRecomputedUnchanged": [descriptor(path) for path in fixtures], "pilotInputCorpusUnchanged": descriptor(CORPUS / "manifest.json"),
        "tokenizerManifest": descriptor(path / "manifest.json"), "recount": descriptor(count_path), "modelIntegration": descriptor(integration_path)})


def complete():
    tokenizer, manifest, path, comparison = selected()
    reports = AI / "evaluation/reports"
    junit = reports / "llm-task-024-trained-tests.xml"
    suite = ET.parse(junit).getroot()
    cases = list(suite.iter("testcase"))
    if not cases or any(list(suite.iter(tag)) for tag in ("failure", "error", "skipped")):
        raise ValueError("Required tokenizer regression failed or skipped")
    focused = [case for case in cases if case.get("classname", "").endswith("test_trained_gen2_tokenizer")]
    if len(focused) < 47:
        raise ValueError("Missing required corpus-trained tokenizer coverage")
    checks_path = reports / "llm-task-024-trained-checks.json"
    fast_path = reports / "llm-task-024-trained-cross-repository-quality.json"
    accepted = read(checks_path)
    fast = read(fast_path)
    if accepted["status"] != "passed" or fast["scorecard"]["summary"]["failed"] or accepted["tokenizerManifest"] != descriptor(path / "manifest.json"):
        raise ValueError("Required governance/cross-repository acceptance missing")
    count_path, counts = supporting_report("corpus/token-counts/gen2/v1")
    integration_path, integration = supporting_report("model/tokenizers/gen2/integration/v1")
    receipt_path = reports / "llm-task-024-completion.json"
    if receipt_path.exists():
        raise ValueError("Completion receipt already exists; use a new version")
    summary = (f"Completed owned tokenizer 0.1.0 from the exact admitted 42-document/1,475,271-byte training split. Both 16,384/32,768 candidates reached their targets and reproduce their merge sets. "
        f"Frozen-candidate validation comparison under the predeclared selection rule selects {tokenizer.vocab_size:,} entries; measured per-domain macro reduction for 32k versus 16k is {comparison['selection']['macroValidationReductionOf32768Versus16384']:.2%}. "
        f"The selected tokenizer counts {counts['trainingUniqueGen2Tokens']:,} unique train, {counts['validationUniqueGen2Tokens']:,} validation and {counts['testUniqueGen2Tokens']:,} test tokens. Validation never fits merges; test counting occurs only after selection. "
        "The immutable release/version record binds corpus, code, vocabulary, merges, special IDs, chat template, comparison and resource receipts. Literal markers, arbitrary bytes, SI/Unicode, long-code overhead, fertility, compression and codec throughput are measured. "
        f"A real {integration['model']['parameterCount']:,}-parameter random decoder consumes the selected tokenizer/template through the context compiler, with matching embeddings/logits and cached-prefix parity. "
        f"All {len(cases)} regressions ({len(focused)} new trained-tokenizer tests), governance/history checks and {fast['scorecard']['summary']['passed']} fast cross-repository checks passed. "
        "Old tokenizers, corpus artifacts and protected evaluation history remain unchanged. No model weights were trained, no checkpoint was loaded/saved, and no serving or external-retention approval was claimed. Task 025 architecture acceptance remains separate.")
    evidence_paths = [path / "manifest.json", r.VERSION_RECORD, AI / manifest["comparison"]["path"], count_path, integration_path,
                      AI / "docs/LLM_TRAINED_GEN2_TOKENIZER.v1.md", junit, checks_path, fast_path, ROOT / "backlog_history/llm-task-024-trained-before/manifest.json"]
    evidence = [item.relative_to(ROOT).as_posix() for item in evidence_paths] + [receipt_path.relative_to(ROOT).as_posix()]
    backlog_path = ROOT / "VOLTFORGE_LOCAL_LLM_BUILD_BACKLOG.json"
    backlog = read(backlog_path)
    task = next(item for item in backlog["tasks"] if item["id"] == "LLM-TASK-024")
    task.update(status="COMPLETED", completed_on="2026-09-09", implementation_state="corpus-trained-tokenizer-released-and-model-context-integrated", verification_status="passed-all-task024-acceptance", verification_result=summary, remaining_release_gates=[], blocked_by=[])
    task["target_paths"] += ["model/gen2_tokenizer_training/", "model/tokenizers/gen2/", "context_compiler/gen2_release.py", "tools/train_gen2_tokenizer.py", "tests/test_trained_gen2_tokenizer.py", "docs/LLM_TRAINED_GEN2_TOKENIZER.v1.md"]
    task["evidence_records"] += [item for item in evidence if item not in task["evidence_records"]]
    task["acceptance_limits"] = ["Offline tokenizer artifact and real neural input integration; no learned model, architecture sweep or trained context quality", "VFAI-FU-015 and deployment backup/restore acceptance remain open; no historical retention policy changed", "Actual model training exposures remain zero"]
    backlog["status_summary"] = dict(Counter(item["status"] for item in backlog["tasks"]))
    backlog["execution"]["next_task_id"] = "LLM-TASK-025"
    backlog["execution"]["current_user_scope"] = "Task024 completed; report the next task ID without starting task025 in this turn."
    backlog["current_checkpoint"] = {"completed_task_id": "LLM-TASK-024", "evidence": receipt_path.relative_to(ROOT).as_posix(), "summary": summary, "active_artifact_id": None, "neural_ready": False, "active_task_id": None, "next_ready_task_id": "LLM-TASK-025"}
    backlog["owned_gen2_tokenizer_release"] = {"version": "0.1.0", "vocabulary_size": tokenizer.vocab_size, "manifest": (path / "manifest.json").relative_to(ROOT).as_posix(), "comparison": (AI / manifest["comparison"]["path"]).relative_to(ROOT).as_posix(), "gen2_counts": count_path.relative_to(ROOT).as_posix(), "real_model_context_integration": integration_path.relative_to(ROOT).as_posix(), "serving_allowed": False, "training_run_approved": False}
    markdown_path = ROOT / "VOLTFORGE_LOCAL_LLM_BUILD_BACKLOG.md"
    markdown = markdown_path.read_text(encoding="utf-8")
    start = markdown.index("**41 active tasks:")
    end = markdown.index("**Previous checkpoint:", start)
    markdown = markdown[:start] + "**41 active tasks: 10 completed, 0 in progress, 31 TODO. No trained conversational model release is accepted.**\n\n**Completed: LLM-TASK-024 — owned corpus-trained tokenizer and real model/context version consumption. Next ready task: LLM-TASK-025.**\n\n" + summary + "\n\n[Release and completed plan](Voltforge_AI/docs/LLM_TRAINED_GEN2_TOKENIZER.v1.md) | [Acceptance evidence](Voltforge_AI/evaluation/reports/llm-task-024-completion.json)\n\n" + markdown[end:]
    start = markdown.index("#### LLM-TASK-024 —")
    end = markdown.index("#### LLM-TASK-025 —", start)
    markdown = markdown[:start] + (f"#### {task['id']} — {task['title']}\n\n**Status:** COMPLETED. **Priority:** {task['priority']}. **Depends on:** {', '.join(task['depends_on'])}.\n\nWork:\n\n"
        + "\n".join("- " + item for item in task["work"]) + "\n\nDefinition of done:\n\n" + "\n".join("- " + item for item in task["definition_of_done"])
        + "\n\n**Required evidence:** " + "; ".join(task["required_evidence"]) + ".\n\n**Verified completion:** " + summary
        + "\n\n**Remaining task024 work:** none under the recorded tokenizer/model-context criteria. Model architecture profiles and learned behavior remain separately gated.\n\n"
        "**Evidence:** [Trained release and plan](Voltforge_AI/docs/LLM_TRAINED_GEN2_TOKENIZER.v1.md), [completion receipt](Voltforge_AI/evaluation/reports/llm-task-024-completion.json). Earlier fixture evidence remains in the canonical JSON and immutable snapshots.\n\n") + markdown[end:]
    causes_path = ROOT / "AI_CHAT_ROOT_CAUSE_BACKLOG.json"
    causes = read(causes_path)
    causes["summary"]["implementation_this_revision"] = "Tasks 015 through 024 completed; owned 0.1.0 tokenizer fitted, selected, released, recounted and consumed by real decoder/context inputs. Task025 is next."
    causes["summary"]["acceptance_limit"] = summary + " Root-cause closure still requires the linked trained-model/runtime acceptance tasks."
    causes["evidence_files"] += [item for item in evidence if item not in causes["evidence_files"]]
    causes["owned_tokenizer_completion"] = {"task_id": "LLM-TASK-024", "status": "COMPLETED", "evidence": receipt_path.relative_to(ROOT).as_posix(), "model_weights_trained": False}
    analysis_path = ROOT / "AI_CHAT_ROOT_CAUSE_ANALYSIS.md"
    analysis = analysis_path.read_text(encoding="utf-8")
    start = analysis.index("## Completed: LLM-TASK-023")
    end = analysis.index("## Previous checkpoint:", start)
    analysis = analysis[:start] + "## Completed: LLM-TASK-024 owned tokenizer\n\n" + summary + "\n\n[Release evidence](Voltforge_AI/docs/LLM_TRAINED_GEN2_TOKENIZER.v1.md) | [Verification](Voltforge_AI/evaluation/reports/llm-task-024-completion.json)\n\n" + analysis[end:]
    receipt = {"schemaVersion": 1, "taskId": "LLM-TASK-024", "status": "planning-check-pending", "recordedAtUtc": datetime.now(timezone.utc).isoformat(), "summary": summary, "taskCompleted": True,
               "tokenizerManifest": descriptor(path / "manifest.json"), "comparison": descriptor(AI / manifest["comparison"]["path"]), "corpusRecount": descriptor(count_path), "modelContextIntegration": descriptor(integration_path),
               "tests": {"passed": len(cases), "newTrainedTokenizerTests": len(focused), "failed": 0, "skipped": 0, "junit": descriptor(junit)}, "checks": descriptor(checks_path), "crossRepositoryQuality": descriptor(fast_path),
               "modelWeightsTrained": False, "servingActivated": False, "externalRetentionAccepted": False, "remainingTask024Work": [], "nextTaskId": "LLM-TASK-025"}
    write(receipt_path, receipt)
    write(backlog_path, backlog)
    markdown_path.write_text(markdown, encoding="utf-8", newline="\n")
    write(causes_path, causes)
    analysis_path.write_text(analysis, encoding="utf-8", newline="\n")
    result = subprocess.run([sys.executable, "-B", "tools/verify_llm_backlogs.py"], cwd=AI, capture_output=True, text=True, encoding="utf-8")
    if result.returncode:
        raise ValueError("Planning check failed: " + result.stdout + result.stderr)
    receipt["planningCheck"] = json.loads(result.stdout)
    receipt["status"] = "completed-all-recorded-task024-tokenizer-acceptance"
    receipt["planningAndDocumentationBindings"] = [descriptor(item) for item in (backlog_path, markdown_path, causes_path, analysis_path, AI / "docs/VOLTForge_AI_RUNBOOK.md", AI / "docs/runbook-contract.v1.json", AI / "docs/LLM_TRAINED_GEN2_TOKENIZER.v1.md", AI / "docs/LLM_GEN2_TOKENIZER_PREPARATION.v1.md", AI / "docs/LLM_PILOT_INPUT_CORPUS.v1.md", AI / "tests/test_trained_gen2_tokenizer.py", Path(__file__))]
    write(receipt_path, receipt)
    print(json.dumps({"status": receipt["status"], "vocabularySize": tokenizer.vocab_size, "testsPassed": len(cases), "fastChecksPassed": fast["scorecard"]["summary"]["passed"], "nextTaskId": "LLM-TASK-025"}))


if __name__ == "__main__":
    if sys.argv[1:] == ["snapshot"]:
        snapshot()
    elif sys.argv[1:] == ["recount"]:
        recount()
    elif sys.argv[1:] == ["prepare-docs"]:
        prepare_docs()
    elif sys.argv[1:] == ["checks"]:
        checks()
    elif sys.argv[1:] == ["complete"]:
        complete()
    else:
        raise SystemExit("Use snapshot, recount, prepare-docs, checks or complete")
