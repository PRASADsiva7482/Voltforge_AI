"""Generate or verify the content-free VFAI-FU-006 readiness receipt.

The evaluator measures governed corpus scale and diversity against the frozen
1M/10M/100M policy. It never generates records, exports text, trains a model, or
claims that an unexecuted quality or learning-curve gate passed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

MASTER_BACKLOG_PATH = AI_ROOT / "AI_MASTER_BACKLOG.json"
FOLLOWUP_BACKLOG_PATH = AI_ROOT / "AI_FOLLOWUP_BACKLOG.json"
POLICY_PATH = AI_ROOT / "corpus_expansion/policy.v1.json"
DATA_POLICY_PATH = AI_ROOT / "data_governance/policy.v1.json"
SOURCE_REGISTRY_PATH = AI_ROOT / "data_governance/source-registry.v1.json"
SYNTHETIC_REPORT_PATH = AI_ROOT / "synthetic_data/reports/current-generation-report.json"
ELECTRONICS_REPORT_PATH = AI_ROOT / "electronics_corpus/reports/current-corpus-report.json"
RETRIEVAL_REPORT_PATH = AI_ROOT / "evaluation/reports/curated-local-retrieval-v1.json"
GLOBAL_FIXTURE_PATH = AI_ROOT / "evaluation/fixtures/v1/suite.jsonl"
TOKENIZER_MANIFEST_PATH = (
    AI_ROOT / "model/tokenizers/vfdlm-byte-bpe-v1.1.0/tokenizer_manifest.json"
)
DEFAULT_REPORT_PATH = AI_ROOT / "evaluation/reports/corpus-expansion-readiness-v1.json"
COMPLETE_STATUSES = frozenset({"complete", "completed", "done"})


class CorpusExpansionReadinessError(RuntimeError):
    """The corpus-expansion readiness evidence is invalid or stale."""


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt_digest(report: Mapping[str, Any]) -> str:
    unsigned = dict(report)
    unsigned.pop("reportSha256", None)
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest()


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CorpusExpansionReadinessError(f"unable to read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise CorpusExpansionReadinessError(f"{label} must be a JSON object")
    return value


def _read_json_lines(path: Path, label: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise CorpusExpansionReadinessError(
                    f"{label} line {line_number} must be a JSON object"
                )
            records.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CorpusExpansionReadinessError(f"unable to read {label}: {exc}") from exc
    return records


def _write_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _backlog_item(backlog: Mapping[str, Any], item_id: str) -> Mapping[str, Any]:
    items = backlog.get("items")
    if not isinstance(items, list):
        raise CorpusExpansionReadinessError("follow-up backlog items are invalid")
    matches = [item for item in items if isinstance(item, Mapping) and item.get("id") == item_id]
    if len(matches) != 1:
        raise CorpusExpansionReadinessError(f"expected exactly one {item_id} backlog item")
    return matches[0]


def _master_complete(master: Mapping[str, Any]) -> bool:
    items = master.get("items")
    return bool(
        master.get("status") in COMPLETE_STATUSES
        and isinstance(items, list)
        and items
        and all(
            isinstance(item, Mapping) and item.get("status") in COMPLETE_STATUSES
            for item in items
        )
    )


def _budget_approved(policy: Mapping[str, Any]) -> bool:
    budget = policy.get("budgetApproval")
    if not isinstance(budget, Mapping) or budget.get("status") != "approved-by-owner":
        return False
    numeric_fields = (
        "maximumRetainedBytes",
        "maximumGenerationCpuHours",
        "maximumCompilerCpuHours",
        "maximumTrainingGpuHours",
    )
    return bool(
        all(
            isinstance(budget.get(field), (int, float))
            and not isinstance(budget.get(field), bool)
            and budget[field] > 0
            for field in numeric_fields
        )
        and isinstance(budget.get("approvedHardwareBoundary"), str)
        and budget["approvedHardwareBoundary"]
    )


def _task_balance(
    policy: Mapping[str, Any], task_percent: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], bool]:
    task_policy = policy.get("taskBalance")
    groups = task_policy.get("groups") if isinstance(task_policy, Mapping) else None
    if not isinstance(groups, list) or not groups:
        raise CorpusExpansionReadinessError("task-balance policy groups are invalid")
    results: list[dict[str, Any]] = []
    assigned: set[str] = set()
    for group in groups:
        if not isinstance(group, Mapping) or not isinstance(group.get("tasks"), list):
            raise CorpusExpansionReadinessError("task-balance group is invalid")
        tasks = [str(task) for task in group["tasks"]]
        if assigned.intersection(tasks):
            raise CorpusExpansionReadinessError("task-balance groups overlap")
        assigned.update(tasks)
        actual = round(sum(float(task_percent.get(task, 0.0)) for task in tasks), 6)
        minimum = float(group.get("minimumPercent", -1))
        maximum = float(group.get("maximumPercent", -1))
        passed = 0 <= minimum <= actual <= maximum
        results.append(
            {
                "groupId": group.get("groupId"),
                "tasks": tasks,
                "actualPercent": actual,
                "minimumPercent": minimum,
                "maximumPercent": maximum,
                "passed": passed,
            }
        )
    unassigned = sorted(set(task_percent) - assigned)
    complete_assignment = not unassigned and round(
        sum(float(value) for value in task_percent.values()), 3
    ) == 100.0
    return results, complete_assignment and all(item["passed"] for item in results)


def _stage_result(
    stage: Mapping[str, Any],
    current: Mapping[str, Any],
    common_gates: Mapping[str, bool],
) -> dict[str, Any]:
    comparisons = {
        "uniqueTrainingPredictedTokens": (
            int(current["uniqueTrainingPredictedTokens"]),
            int(stage["minimumUniqueTrainingPredictedTokens"]),
        ),
        "approvedSourceFamilies": (
            int(current["approvedSourceFamilies"]),
            int(stage["minimumApprovedSourceFamilies"]),
        ),
        "exactBoardVariants": (
            int(current["exactBoardVariants"]),
            int(stage["minimumExactBoardVariants"]),
        ),
        "exactComponentVariants": (
            int(current["exactComponentVariants"]),
            int(stage["minimumExactComponentVariants"]),
        ),
        "compilerToolchains": (
            int(current["compilerToolchains"]),
            int(stage["minimumCompilerToolchains"]),
        ),
        "modelValidationRecords": (
            int(current["modelValidationRecords"]),
            int(stage["minimumModelValidationRecords"]),
        ),
        "globalHeldOutCases": (
            int(current["globalHeldOutCases"]),
            int(stage["minimumGlobalHeldOutCases"]),
        ),
        "retrievalHeldOutCases": (
            int(current["retrievalHeldOutCases"]),
            int(stage["minimumRetrievalHeldOutCases"]),
        ),
        "adversarialSafetyCases": (
            int(current["adversarialSafetyCases"]),
            int(stage["minimumAdversarialSafetyCases"]),
        ),
        "manualAuditRecords": (
            int(current["manualAuditRecords"]),
            int(stage["minimumManualAuditRecords"]),
        ),
    }
    metric_gates = {
        name: {"actual": actual, "required": required, "passed": actual >= required}
        for name, (actual, required) in comparisons.items()
    }
    ready = all(item["passed"] for item in metric_gates.values()) and all(
        common_gates.values()
    )
    return {
        "stageId": stage.get("stageId"),
        "metrics": metric_gates,
        "commonGates": dict(common_gates),
        "readyForThreeSeedLearningCurve": ready,
    }


def build_report(
    *,
    master_backlog: Mapping[str, Any],
    followup_backlog: Mapping[str, Any],
    policy: Mapping[str, Any],
    current: Mapping[str, Any],
    generated_on: str,
    source_sha256: Mapping[str, str],
) -> dict[str, Any]:
    if policy.get("sourceFollowup") != "VFAI-FU-006":
        raise CorpusExpansionReadinessError("corpus policy is not bound to VFAI-FU-006")
    fu006 = _backlog_item(followup_backlog, "VFAI-FU-006")
    fu015 = _backlog_item(followup_backlog, "VFAI-FU-015")
    master_ready = _master_complete(master_backlog)
    immutable_ready = fu015.get("status") in COMPLETE_STATUSES
    budget_ready = _budget_approved(policy)
    task_results, task_ready = _task_balance(policy, current["taskPercent"])
    quality = policy.get("qualityGates")
    if not isinstance(quality, Mapping):
        raise CorpusExpansionReadinessError("corpus quality gates are invalid")
    semantic_ready = current.get("semanticDeduplicationEnforced") is True
    uncertainty_ready = (
        current.get("unsupportedClaimUncertaintyCoveragePercent")
        == quality.get("unsupportedClaimUncertaintyCoveragePercent")
    )
    manual_audit_ready = (
        current.get("criticalManualAuditErrorCount")
        == quality.get("criticalManualAuditErrorCount")
    )
    common_gates = {
        "masterBacklogComplete": master_ready,
        "ownerStorageAndComputeBudgetApproved": budget_ready,
        "immutableReleaseRetentionReady": immutable_ready,
        "allCurrentSourcesApprovedForTraining": current.get("sourceAdmissionReady") is True,
        "privateAndExternalInputsRemainDisabled": current.get("forbiddenInputsDisabled")
        is True,
        "exactDuplicatesRejected": current.get("acceptedExactDuplicateCount")
        == quality.get("acceptedExactDuplicateCount"),
        "trainingHeldOutLeakageRejected": current.get("trainingHeldOutCollisionCount")
        == quality.get("trainingHeldOutCollisionCount"),
        "deterministicReceiptCoverageComplete": current.get(
            "deterministicLabelReceiptCoveragePercent"
        )
        == quality.get("deterministicLabelReceiptCoveragePercent"),
        "compilerReceiptCoverageComplete": current.get(
            "compilerLabelReceiptCoveragePercent"
        )
        == quality.get("compilerLabelReceiptCoveragePercent"),
        "semanticDeduplicationEnforced": semantic_ready,
        "unsupportedClaimsRepresentedAsUncertainty": uncertainty_ready,
        "criticalManualAuditErrorsZero": manual_audit_ready,
        "taskBalanceWithinBounds": task_ready,
    }
    stages = policy.get("releaseStages")
    if not isinstance(stages, list) or len(stages) != 3:
        raise CorpusExpansionReadinessError("corpus release stages are invalid")
    stage_results = [_stage_result(stage, current, common_gates) for stage in stages]
    ready_stages = [
        item["stageId"] for item in stage_results if item["readyForThreeSeedLearningCurve"]
    ]
    if not master_ready:
        decision = "await-master-backlog-completion"
    elif not immutable_ready or not budget_ready:
        decision = "await-immutable-retention-and-owner-budget"
    elif not semantic_ready or not uncertainty_ready or not manual_audit_ready:
        decision = "build-scalable-quality-and-dedup-evidence"
    elif not ready_stages:
        decision = "expand-governed-sources-for-edge-stage-1"
    else:
        decision = f"{ready_stages[-1]}-ready-for-three-seed-learning-curve"
    next_stage = next(
        (
            item
            for item in stage_results
            if not item["readyForThreeSeedLearningCurve"]
        ),
        None,
    )
    next_token_target = (
        next_stage["metrics"]["uniqueTrainingPredictedTokens"]["required"]
        if next_stage is not None
        else int(stages[-1]["minimumUniqueTrainingPredictedTokens"])
    )
    unique_tokens = int(current["uniqueTrainingPredictedTokens"])
    report: dict[str, Any] = {
        "schemaVersion": 1,
        "reportId": "vfai-fu-006-corpus-expansion-readiness-v1",
        "generatedOn": generated_on,
        "followup": {"id": fu006.get("id"), "recordedStatus": fu006.get("status")},
        "policy": {
            "policyId": policy.get("policyId"),
            "status": policy.get("status"),
            "countingUnit": policy.get("countingContract", {}).get("unit"),
        },
        "current": dict(current),
        "taskBalance": {"groups": task_results, "passed": task_ready},
        "commonGates": common_gates,
        "stages": stage_results,
        "currentApprovedStage": ready_stages[-1] if ready_stages else None,
        "nextStage": next_stage["stageId"] if next_stage is not None else None,
        "nextStageTokenCoveragePercent": round(
            unique_tokens * 100 / max(1, next_token_target), 6
        ),
        "nextStageTokenGap": max(0, next_token_target - unique_tokens),
        "decision": decision,
        "completionClaimed": False,
        "corpusRecordsGeneratedByEvaluator": 0,
        "modelRunsExecuted": 0,
        "governedCorpusExported": False,
        "networkAccessed": False,
        "sourceSha256": dict(source_sha256),
        "reportSha256": "",
    }
    report["reportSha256"] = _receipt_digest(report)
    return report


def _current_metrics(
    *,
    followup: Mapping[str, Any],
    policy: Mapping[str, Any],
    data_policy: Mapping[str, Any],
    source_registry: Mapping[str, Any],
    synthetic: Mapping[str, Any],
    electronics: Mapping[str, Any],
    retrieval: Mapping[str, Any],
    global_cases: Sequence[Mapping[str, Any]],
    tokenizer_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    del followup, policy
    from gen1_training import load_approved_corpus
    from model.gen1 import Gen1Config

    config = Gen1Config(
        vocab_size=int(tokenizer_manifest["vocabSize"]), max_sequence_length=128
    )
    corpus = load_approved_corpus(
        config,
        tokenizer_directory=TOKENIZER_MANIFEST_PATH.parent,
        packing_block_size=128,
    )
    lineage = tokenizer_manifest.get("lineage")
    source_ids = lineage.get("sourceIds") if isinstance(lineage, Mapping) else []
    registry_sources = source_registry.get("sources")
    if not isinstance(registry_sources, list):
        raise CorpusExpansionReadinessError("source registry entries are invalid")
    approved_sources = {
        item.get("sourceId")
        for item in registry_sources
        if isinstance(item, Mapping)
        and item.get("approval", {}).get("status") == "approved"
        and item.get("privacy", {}).get("trainingAllowed") is True
    }
    source_admission_ready = bool(source_ids) and set(source_ids).issubset(approved_sources)
    task_balance = synthetic.get("taskBalance")
    if not isinstance(task_balance, Mapping):
        raise CorpusExpansionReadinessError("synthetic task balance is invalid")
    task_percent = {
        str(task): float(value.get("percent", 0.0))
        for task, value in task_balance.items()
        if isinstance(value, Mapping)
    }
    coverage = synthetic.get("coverage")
    supported_boards = coverage.get("supportedBoards") if isinstance(coverage, Mapping) else {}
    components = coverage.get("components") if isinstance(coverage, Mapping) else {}
    compile_targets = (
        coverage.get("compiledFirmwareTargets", []) if isinstance(coverage, Mapping) else []
    )
    summary = synthetic.get("summary")
    leakage = synthetic.get("leakage")
    global_task_counts = Counter(str(item.get("task")) for item in global_cases)
    forbidden_inputs_disabled = bool(
        data_policy.get("defaultDecision") == "deny"
        and data_policy.get("liveWebRetention") == "request-scope-only"
        and data_policy.get("privateConsentWorkflowImplemented") is False
    )
    return {
        "datasetId": corpus.manifest.get("datasetId"),
        "tokenizerVersion": corpus.manifest.get("tokenizer", {}).get("version"),
        "packedCorpusFingerprint": corpus.fingerprint,
        "trainingRecordCount": corpus.manifest.get("trainingRecordCount"),
        "modelValidationRecords": corpus.manifest.get("validationRecordCount"),
        "uniqueTrainingPredictedTokens": corpus.train.predicted_token_count,
        "approvedSourceFamilies": len(source_ids),
        "approvedSourceIds": sorted(source_ids),
        "sourceAdmissionReady": source_admission_ready,
        "exactBoardVariants": len(supported_boards.get("covered", [])),
        "exactComponentVariants": len(components.get("covered", [])),
        "compilerToolchains": len(
            {
                item.get("toolchainId")
                for item in compile_targets
                if isinstance(item, Mapping) and item.get("toolchainId")
            }
        ),
        "globalHeldOutCases": len(global_cases),
        "retrievalHeldOutCases": int(retrieval.get("heldOutCaseCount", 0)),
        "adversarialSafetyCases": global_task_counts.get("adversarial_safety", 0),
        "manualAuditRecords": 0,
        "acceptedExactDuplicateCount": 0
        if isinstance(summary, Mapping) and summary.get("decision") == "pass"
        else None,
        "trainingHeldOutCollisionCount": 0
        if isinstance(summary, Mapping) and summary.get("decision") == "pass"
        else None,
        "heldOutCollisionCandidatesRejected": leakage.get("heldOutCollisionsRejected")
        if isinstance(leakage, Mapping)
        else None,
        "deterministicLabelReceiptCoveragePercent": 100.0
        if isinstance(summary, Mapping) and summary.get("decision") == "pass"
        else None,
        "compilerLabelReceiptCoveragePercent": 100.0
        if synthetic.get("compileVerification", {}).get("receiptCount", 0) > 0
        and isinstance(summary, Mapping)
        and summary.get("decision") == "pass"
        else None,
        "semanticDeduplicationEnforced": False,
        "unsupportedClaimUncertaintyCoveragePercent": None,
        "criticalManualAuditErrorCount": None,
        "forbiddenInputsDisabled": forbidden_inputs_disabled,
        "privateConsentWorkflowImplemented": data_policy.get(
            "privateConsentWorkflowImplemented"
        ),
        "electronicsKnowledgeRecords": int(electronics.get("recordCount", 0)),
        "taskPercent": task_percent,
    }


def _current_inputs() -> dict[str, Any]:
    master = _read_json_object(MASTER_BACKLOG_PATH, "master backlog")
    followup = _read_json_object(FOLLOWUP_BACKLOG_PATH, "follow-up backlog")
    policy = _read_json_object(POLICY_PATH, "corpus-expansion policy")
    data_policy = _read_json_object(DATA_POLICY_PATH, "data-governance policy")
    source_registry = _read_json_object(SOURCE_REGISTRY_PATH, "source registry")
    synthetic = _read_json_object(SYNTHETIC_REPORT_PATH, "synthetic release report")
    electronics = _read_json_object(ELECTRONICS_REPORT_PATH, "electronics corpus report")
    retrieval = _read_json_object(RETRIEVAL_REPORT_PATH, "retrieval evaluation report")
    global_cases = _read_json_lines(GLOBAL_FIXTURE_PATH, "global evaluation fixture")
    tokenizer_manifest = _read_json_object(TOKENIZER_MANIFEST_PATH, "tokenizer manifest")
    return {
        "master_backlog": master,
        "followup_backlog": followup,
        "policy": policy,
        "current": _current_metrics(
            followup=followup,
            policy=policy,
            data_policy=data_policy,
            source_registry=source_registry,
            synthetic=synthetic,
            electronics=electronics,
            retrieval=retrieval,
            global_cases=global_cases,
            tokenizer_manifest=tokenizer_manifest,
        ),
        "source_sha256": {
            path.relative_to(AI_ROOT).as_posix(): _sha256_file(path)
            for path in (
                MASTER_BACKLOG_PATH,
                FOLLOWUP_BACKLOG_PATH,
                POLICY_PATH,
                DATA_POLICY_PATH,
                SOURCE_REGISTRY_PATH,
                SYNTHETIC_REPORT_PATH,
                ELECTRONICS_REPORT_PATH,
                RETRIEVAL_REPORT_PATH,
                GLOBAL_FIXTURE_PATH,
                TOKENIZER_MANIFEST_PATH,
                Path(__file__),
            )
        },
    }


def _build_current_report(generated_on: str) -> dict[str, Any]:
    return build_report(generated_on=generated_on, **_current_inputs())


def evaluate(path: Path, generated_on: str) -> dict[str, Any]:
    report = _build_current_report(generated_on)
    _write_json(report, path)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("reportSha256") != _receipt_digest(report):
        raise CorpusExpansionReadinessError("VFAI-FU-006 receipt checksum is invalid")
    stages = report.get("stages")
    if not isinstance(stages, list) or len(stages) != 3:
        raise CorpusExpansionReadinessError("VFAI-FU-006 receipt stages are invalid")
    for stage in stages:
        metrics = stage.get("metrics") if isinstance(stage, Mapping) else None
        common = stage.get("commonGates") if isinstance(stage, Mapping) else None
        if not isinstance(metrics, Mapping) or not isinstance(common, Mapping):
            raise CorpusExpansionReadinessError("VFAI-FU-006 stage gates are invalid")
        expected_ready = all(
            isinstance(value, Mapping) and value.get("passed") is True
            for value in metrics.values()
        ) and all(value is True for value in common.values())
        if stage.get("readyForThreeSeedLearningCurve") is not expected_ready:
            raise CorpusExpansionReadinessError("VFAI-FU-006 stage decision is inconsistent")
    common_gates = report.get("commonGates")
    if not isinstance(common_gates, Mapping):
        raise CorpusExpansionReadinessError("VFAI-FU-006 common gates are invalid")
    ready_stages = [
        stage.get("stageId")
        for stage in stages
        if stage.get("readyForThreeSeedLearningCurve") is True
    ]
    if common_gates.get("masterBacklogComplete") is not True:
        expected_decision = "await-master-backlog-completion"
    elif (
        common_gates.get("immutableReleaseRetentionReady") is not True
        or common_gates.get("ownerStorageAndComputeBudgetApproved") is not True
    ):
        expected_decision = "await-immutable-retention-and-owner-budget"
    elif (
        common_gates.get("semanticDeduplicationEnforced") is not True
        or common_gates.get("unsupportedClaimsRepresentedAsUncertainty") is not True
        or common_gates.get("criticalManualAuditErrorsZero") is not True
    ):
        expected_decision = "build-scalable-quality-and-dedup-evidence"
    elif not ready_stages:
        expected_decision = "expand-governed-sources-for-edge-stage-1"
    else:
        expected_decision = f"{ready_stages[-1]}-ready-for-three-seed-learning-curve"
    if report.get("decision") != expected_decision:
        raise CorpusExpansionReadinessError("VFAI-FU-006 receipt decision is inconsistent")
    if report.get("currentApprovedStage") != (ready_stages[-1] if ready_stages else None):
        raise CorpusExpansionReadinessError("VFAI-FU-006 approved stage is inconsistent")
    if any(
        (
            report.get("completionClaimed") is not False,
            report.get("corpusRecordsGeneratedByEvaluator") != 0,
            report.get("modelRunsExecuted") != 0,
            report.get("governedCorpusExported") is not False,
            report.get("networkAccessed") is not False,
        )
    ):
        raise CorpusExpansionReadinessError("VFAI-FU-006 receipt overclaims execution")


def verify(path: Path) -> dict[str, Any]:
    report = _read_json_object(path, "VFAI-FU-006 readiness receipt")
    validate_report(report)
    expected = _build_current_report(str(report.get("generatedOn")))
    if report != expected:
        raise CorpusExpansionReadinessError("VFAI-FU-006 readiness receipt is stale")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--generated-on", required=True)
    evaluate_parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--input", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    report = (
        evaluate(arguments.output.resolve(), arguments.generated_on)
        if arguments.command == "evaluate"
        else verify(arguments.input.resolve())
    )
    print(
        json.dumps(
            {
                "ok": True,
                "command": arguments.command,
                "reportId": report["reportId"],
                "decision": report["decision"],
                "currentApprovedStage": report["currentApprovedStage"],
                "nextStage": report["nextStage"],
                "nextStageTokenCoveragePercent": report[
                    "nextStageTokenCoveragePercent"
                ],
                "nextStageTokenGap": report["nextStageTokenGap"],
                "reportSha256": report["reportSha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
