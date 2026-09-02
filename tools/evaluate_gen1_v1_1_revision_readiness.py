"""Generate or verify the content-free VFAI-FU-013 readiness receipt.

The evaluator checks authorization, lineage, and historical-artifact boundaries.
It never allocates a training run, parses training examples, loads model weights,
packages an artifact, or changes the signed model registry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from model.registry_manager import (  # noqa: E402
    RegistryManagerError,
    artifact_root_from_entry,
    verify_artifact_directory,
    verify_registry,
)
from tools.evaluate_corpus_expansion_readiness import (  # noqa: E402
    CorpusExpansionReadinessError,
    verify as verify_corpus_readiness,
)


MASTER_BACKLOG_PATH = AI_ROOT / "AI_MASTER_BACKLOG.json"
FOLLOWUP_BACKLOG_PATH = AI_ROOT / "AI_FOLLOWUP_BACKLOG.json"
POLICY_PATH = AI_ROOT / "gen1_training/tokenizer-v1.1-revision-policy.v1.json"
CORPUS_REPORT_PATH = AI_ROOT / "evaluation/reports/corpus-expansion-readiness-v1.json"
TOKENIZER_MANIFEST_PATH = (
    AI_ROOT / "model/tokenizers/vfdlm-byte-bpe-v1.1.0/tokenizer_manifest.json"
)
HISTORICAL_PLAN_PATH = AI_ROOT / "gen1_bootstrap/plan.v1.json"
REGISTRY_PATH = AI_ROOT / "model/registry/active_model.json"
HISTORICAL_ARTIFACT_MANIFEST_PATHS = (
    AI_ROOT / "model/registry/manifests/vfdlm-g1-edge-v0.1.0-bootstrap.json",
    AI_ROOT / "model/registry/manifests/vfdlm-g1-edge-v0.1.1-runtime.json",
    AI_ROOT / "model/registry/manifests/vfdlm-g1-edge-v0.1.2-optimized.json",
)
TRAINING_DATA_PATH = AI_ROOT / "gen1_training/data.py"
TRAINER_PATH = AI_ROOT / "gen1_training/trainer.py"
TRAIN_TOOL_PATH = AI_ROOT / "tools/train_gen1.py"
DEFAULT_REPORT_PATH = (
    AI_ROOT / "evaluation/reports/gen1-v1.1-revision-readiness-v1.json"
)
COMPLETE_STATUSES = frozenset({"complete", "completed", "done"})
STAGE_ORDER = {
    None: 0,
    "edge-stage-1-1m": 1,
    "edge-stage-2-10m": 2,
    "edge-stage-3-100m": 3,
}


class Gen1V11RevisionReadinessError(RuntimeError):
    """VFAI-FU-013 evidence is invalid, unsafe, or stale."""


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _declared_digest_valid(value: Mapping[str, Any], field: str) -> bool:
    declared = value.get(field)
    if not isinstance(declared, str) or len(declared) != 64:
        return False
    unsigned = dict(value)
    unsigned.pop(field, None)
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest() == declared


def _report_digest(report: Mapping[str, Any]) -> str:
    unsigned = dict(report)
    unsigned.pop("reportSha256", None)
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Gen1V11RevisionReadinessError(f"unable to read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise Gen1V11RevisionReadinessError(f"{label} must be a JSON object")
    return value


def _write_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _backlog_item(backlog: Mapping[str, Any], item_id: str) -> Mapping[str, Any]:
    items = backlog.get("items")
    matches = [
        item
        for item in items
        if isinstance(item, Mapping) and item.get("id") == item_id
    ] if isinstance(items, list) else []
    if len(matches) != 1:
        raise Gen1V11RevisionReadinessError(f"expected exactly one {item_id} item")
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


def _resolve_optional_path(value: Any, label: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise Gen1V11RevisionReadinessError(f"{label} must be null or a path")
    path = (AI_ROOT / value).resolve()
    try:
        path.relative_to(AI_ROOT.resolve())
    except ValueError as exc:
        raise Gen1V11RevisionReadinessError(f"{label} escapes the AI workspace") from exc
    return path if path.is_file() else None


def _corpus_evidence(report: Mapping[str, Any], required_stage: str) -> dict[str, Any]:
    stage = report.get("currentApprovedStage")
    receipt_valid = bool(
        report.get("reportId") == "vfai-fu-006-corpus-expansion-readiness-v1"
        and _declared_digest_valid(report, "reportSha256")
        and report.get("completionClaimed") is False
        and report.get("modelRunsExecuted") == 0
        and report.get("networkAccessed") is False
    )
    return {
        "receiptValid": receipt_valid,
        "reportSha256": report.get("reportSha256"),
        "decision": report.get("decision"),
        "currentApprovedStage": stage,
        "requiredStage": required_stage,
        "stageReady": bool(
            receipt_valid
            and stage in STAGE_ORDER
            and required_stage in STAGE_ORDER
            and STAGE_ORDER[stage] >= STAGE_ORDER[required_stage]
        ),
        "currentUniqueTrainingPredictedTokens": report.get(
            "current", {}
        ).get("uniqueTrainingPredictedTokens"),
        "immutableReleaseRetentionReady": report.get("commonGates", {}).get(
            "immutableReleaseRetentionReady"
        ) is True,
        "ownerStorageAndComputeBudgetApproved": report.get("commonGates", {}).get(
            "ownerStorageAndComputeBudgetApproved"
        ) is True,
    }


def _tokenizer_evidence(manifest: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    lineage = policy.get("lineageContract", {})
    files = manifest.get("files")
    file_checks = []
    if isinstance(files, Mapping):
        for descriptor in files.values():
            if not isinstance(descriptor, Mapping):
                file_checks.append(False)
                continue
            path = TOKENIZER_MANIFEST_PATH.parent / str(descriptor.get("path", ""))
            file_checks.append(
                path.is_file() and _sha256_file(path) == descriptor.get("sha256")
            )
    shards = manifest.get("lineage", {}).get("shards")
    shard_checks = []
    if isinstance(shards, list):
        for descriptor in shards:
            if not isinstance(descriptor, Mapping):
                shard_checks.append(False)
                continue
            path = AI_ROOT / str(descriptor.get("path", ""))
            shard_checks.append(
                path.is_file() and _sha256_file(path) == descriptor.get("sha256")
            )
    split = manifest.get("lineage", {}).get("split", {})
    ready = bool(
        _declared_digest_valid(manifest, "artifactSha256")
        and manifest.get("releaseStatus") == "approved"
        and manifest.get("tokenizerId") == lineage.get("tokenizerId")
        and manifest.get("version") == lineage.get("tokenizerVersion")
        and manifest.get("artifactSha256") == lineage.get("tokenizerArtifactSha256")
        and file_checks
        and all(file_checks)
        and shard_checks
        and all(shard_checks)
        and split.get("trainingRecordCount") == lineage.get("currentTrainingRecordCount")
        and split.get("evaluationRecordCount") == lineage.get("currentValidationRecordCount")
    )
    return {
        "ready": ready,
        "tokenizerId": manifest.get("tokenizerId"),
        "version": manifest.get("version"),
        "artifactSha256": manifest.get("artifactSha256"),
        "trainingRecordCount": split.get("trainingRecordCount"),
        "validationRecordCount": split.get("evaluationRecordCount"),
        "allTokenizerFilesMatch": bool(file_checks and all(file_checks)),
        "allCurrentShardBytesMatch": bool(shard_checks and all(shard_checks)),
        "currentShardPaths": [
            item.get("path") for item in shards or [] if isinstance(item, Mapping)
        ],
    }


def _historical_evidence(
    plan: Mapping[str, Any],
    registry: Mapping[str, Any],
    manifests: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    artifacts = registry.get("artifacts")
    registry_artifacts_safe = bool(
        isinstance(artifacts, list)
        and len(artifacts) == len(manifests) == 3
        and registry.get("activeArtifactId") is None
        and registry.get("previousActiveArtifactId") is None
        and registry.get("state") == "no-approved-artifact"
        and all(
            isinstance(item, Mapping)
            and item.get("releaseStatus") == "experimental"
            and item.get("activationEligible") is False
            for item in artifacts
        )
    )
    manifests_safe = bool(
        manifests
        and all(
            item.get("immutable") is True
            and item.get("tokenizer", {}).get("version") == "1.0.0"
            and item.get("evaluation", {}).get("releaseApproved") is False
            and item.get("releaseState", {}).get("releaseStatus") == "experimental"
            and item.get("releaseState", {}).get("activationEligible") is False
            and item.get("releaseState", {}).get("servingAllowed") is False
            for item in manifests
        )
    )
    plan_safe = bool(
        plan.get("runId") == "vfai014-gen1-bootstrap-v1"
        and plan.get("controls", {}).get("approvedUniqueTrainingPredictedTokens")
        == 71_377
        and plan.get("heldoutPolicy", {}).get("recordCount") == 16
        and plan.get("releaseBoundary", {}).get("releaseApproved") is False
    )
    return {
        "ready": registry_artifacts_safe and manifests_safe and plan_safe,
        "registryRevision": registry.get("revision"),
        "activeArtifactId": registry.get("activeArtifactId"),
        "artifactIds": [
            item.get("artifactId") for item in artifacts or [] if isinstance(item, Mapping)
        ],
        "allExistingArtifactsExperimentalAndIneligible": registry_artifacts_safe,
        "allExistingArtifactPackagesBindTokenizerV1": manifests_safe,
        "historicalPlanRunId": plan.get("runId"),
        "historicalUniqueTrainingPredictedTokens": plan.get("controls", {}).get(
            "approvedUniqueTrainingPredictedTokens"
        ),
        "historicalValidationRecordCount": plan.get("heldoutPolicy", {}).get(
            "recordCount"
        ),
    }


def _completion_input_evidence(policy: Mapping[str, Any]) -> dict[str, Any]:
    plan = policy.get("immutablePlanContract", {})
    results = policy.get("resultInputs", {})
    budget_path = _resolve_optional_path(
        plan.get("ownerComputeBudgetReceiptPath"), "owner compute budget receipt"
    )
    immutable_path = _resolve_optional_path(
        plan.get("immutableCorpusReleaseReceiptPath"), "immutable release receipt"
    )
    training_plan_path = _resolve_optional_path(
        plan.get("trainingPlanPath"), "immutable training plan"
    )
    run_paths = results.get("trainingRunReceiptPaths")
    if not isinstance(run_paths, list):
        raise Gen1V11RevisionReadinessError("training run receipt paths must be a list")
    resolved_runs = [
        _resolve_optional_path(value, "training run receipt") for value in run_paths
    ]
    evaluation_path = _resolve_optional_path(
        results.get("evaluationScorecardPath"), "evaluation scorecard"
    )
    rollback_path = _resolve_optional_path(
        results.get("rollbackAndCanaryReceiptPath"), "rollback/canary receipt"
    )
    return {
        "ownerComputeBudgetReceiptReady": budget_path is not None,
        "ownerComputeBudgetReceiptPath": (
            budget_path.relative_to(AI_ROOT).as_posix() if budget_path else None
        ),
        "immutableCorpusReleaseReceiptReady": immutable_path is not None,
        "immutableCorpusReleaseReceiptPath": (
            immutable_path.relative_to(AI_ROOT).as_posix() if immutable_path else None
        ),
        "immutableTrainingPlanReady": training_plan_path is not None,
        "immutableTrainingPlanPath": (
            training_plan_path.relative_to(AI_ROOT).as_posix()
            if training_plan_path else None
        ),
        "trainingRunReceiptCount": len([path for path in resolved_runs if path]),
        "expectedTrainingRunReceiptCount": len(run_paths),
        "allTrainingRunReceiptsReady": bool(
            run_paths and all(path is not None for path in resolved_runs)
        ),
        "evaluationScorecardReady": evaluation_path is not None,
        "rollbackAndCanaryReceiptReady": rollback_path is not None,
    }


def _expected_decision(gates: Mapping[str, Any]) -> str:
    if gates.get("masterBacklogComplete") is not True:
        return "await-master-backlog-completion"
    if gates.get("sourceEvidenceReady") is not True:
        return "repair-source-readiness-evidence"
    immutable = gates.get("immutableReleaseRetentionReady") is True
    budget = gates.get("ownerComputeBudgetApproved") is True
    if not immutable and not budget:
        return "await-immutable-retention-and-owner-budget"
    if not immutable:
        return "await-immutable-release-retention"
    if not budget:
        return "await-owner-compute-budget"
    if gates.get("governedCorpusStageReady") is not True:
        return "await-governed-corpus-stage"
    if gates.get("immutableTrainingPlanReady") is not True:
        return "create-immutable-tokenizer-v1.1-training-plan"
    if gates.get("trainingRunReceiptsReady") is not True:
        return "ready-to-run-approved-inactive-training-matrix"
    if gates.get("completeEvaluationScorecardReady") is not True:
        return "complete-inactive-artifact-evaluation-scorecard"
    return "inactive-results-ready-for-independent-release-review"


def build_report(
    *,
    master_backlog: Mapping[str, Any],
    followup_backlog: Mapping[str, Any],
    policy: Mapping[str, Any],
    corpus: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    historical: Mapping[str, Any],
    completion_inputs: Mapping[str, Any],
    generated_on: str,
    source_sha256: Mapping[str, str],
) -> dict[str, Any]:
    if policy.get("sourceFollowup") != "VFAI-FU-013":
        raise Gen1V11RevisionReadinessError("policy is not bound to VFAI-FU-013")
    fu013 = _backlog_item(followup_backlog, "VFAI-FU-013")
    fu015 = _backlog_item(followup_backlog, "VFAI-FU-015")
    source_ready = bool(
        corpus.get("receiptValid") is True
        and tokenizer.get("ready") is True
        and historical.get("ready") is True
    )
    immutable_ready = bool(
        fu015.get("status") in COMPLETE_STATUSES
        and corpus.get("immutableReleaseRetentionReady") is True
        and completion_inputs.get("immutableCorpusReleaseReceiptReady") is True
    )
    budget_ready = bool(
        corpus.get("ownerStorageAndComputeBudgetApproved") is True
        and completion_inputs.get("ownerComputeBudgetReceiptReady") is True
    )
    complete_scorecard = bool(
        completion_inputs.get("evaluationScorecardReady") is True
        and completion_inputs.get("rollbackAndCanaryReceiptReady") is True
    )
    gates = {
        "masterBacklogComplete": _master_complete(master_backlog),
        "sourceEvidenceReady": source_ready,
        "immutableReleaseRetentionReady": immutable_ready,
        "ownerComputeBudgetApproved": budget_ready,
        "governedCorpusStageReady": corpus.get("stageReady") is True,
        "immutableTrainingPlanReady": completion_inputs.get(
            "immutableTrainingPlanReady"
        ) is True,
        "trainingRunReceiptsReady": completion_inputs.get(
            "allTrainingRunReceiptsReady"
        ) is True,
        "completeEvaluationScorecardReady": complete_scorecard,
    }
    report: dict[str, Any] = {
        "schemaVersion": 1,
        "reportId": "vfai-fu-013-gen1-v1.1-revision-readiness-v1",
        "generatedOn": generated_on,
        "followup": {"id": fu013.get("id"), "recordedStatus": fu013.get("status")},
        "immutableReleaseFollowup": {
            "id": fu015.get("id"),
            "recordedStatus": fu015.get("status"),
        },
        "policy": {
            "policyId": policy.get("policyId"),
            "status": policy.get("status"),
            "requiredCorpusStage": policy.get("minimumStartConditions", {}).get(
                "requiredCorpusStage"
            ),
            "minimumIndependentSeeds": policy.get("minimumStartConditions", {}).get(
                "minimumIndependentSeeds"
            ),
        },
        "corpus": dict(corpus),
        "approvedTrainingTokenizer": dict(tokenizer),
        "historicalArtifacts": dict(historical),
        "completionInputs": dict(completion_inputs),
        "gates": gates,
        "decision": _expected_decision(gates),
        "completionClaimed": False,
        "trainingExamplesParsedByEvaluator": False,
        "modelWeightsLoadedByEvaluator": False,
        "trainingRunsAllocatedByEvaluator": 0,
        "optimizerStepsExecutedByEvaluator": 0,
        "candidateArtifactsCreatedByEvaluator": 0,
        "historicalArtifactsModifiedByEvaluator": False,
        "registryModifiedByEvaluator": False,
        "networkAccessed": False,
        "releaseOrActivationApproved": False,
        "sourceSha256": dict(source_sha256),
        "reportSha256": "",
    }
    report["reportSha256"] = _report_digest(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("reportSha256") != _report_digest(report):
        raise Gen1V11RevisionReadinessError("VFAI-FU-013 receipt checksum is invalid")
    gates = report.get("gates")
    if not isinstance(gates, Mapping) or report.get("decision") != _expected_decision(gates):
        raise Gen1V11RevisionReadinessError("VFAI-FU-013 receipt decision is inconsistent")
    if any(
        (
            report.get("completionClaimed") is not False,
            report.get("trainingExamplesParsedByEvaluator") is not False,
            report.get("modelWeightsLoadedByEvaluator") is not False,
            report.get("trainingRunsAllocatedByEvaluator") != 0,
            report.get("optimizerStepsExecutedByEvaluator") != 0,
            report.get("candidateArtifactsCreatedByEvaluator") != 0,
            report.get("historicalArtifactsModifiedByEvaluator") is not False,
            report.get("registryModifiedByEvaluator") is not False,
            report.get("networkAccessed") is not False,
            report.get("releaseOrActivationApproved") is not False,
        )
    ):
        raise Gen1V11RevisionReadinessError("VFAI-FU-013 receipt overclaims mutation")


def _current_inputs() -> dict[str, Any]:
    master = _read_json(MASTER_BACKLOG_PATH, "master backlog")
    followup = _read_json(FOLLOWUP_BACKLOG_PATH, "follow-up backlog")
    policy = _read_json(POLICY_PATH, "VFAI-FU-013 readiness policy")
    try:
        corpus_report = verify_corpus_readiness(CORPUS_REPORT_PATH)
    except CorpusExpansionReadinessError as exc:
        raise Gen1V11RevisionReadinessError(
            f"corpus readiness receipt is invalid or stale: {exc}"
        ) from exc
    tokenizer_manifest = _read_json(TOKENIZER_MANIFEST_PATH, "tokenizer manifest")
    historical_plan = _read_json(HISTORICAL_PLAN_PATH, "historical bootstrap plan")
    try:
        registry = verify_registry(REGISTRY_PATH)
        verified_artifacts = [
            verify_artifact_directory(artifact_root_from_entry(REGISTRY_PATH, entry))
            for entry in registry["artifacts"]
        ]
    except RegistryManagerError as exc:
        raise Gen1V11RevisionReadinessError(
            f"historical registry or package verification failed: {exc}"
        ) from exc
    historical_manifests = [artifact.manifest for artifact in verified_artifacts]
    required_stage = policy.get("minimumStartConditions", {}).get(
        "requiredCorpusStage"
    )
    if required_stage not in STAGE_ORDER:
        raise Gen1V11RevisionReadinessError("required corpus stage is invalid")
    source_paths = (
        MASTER_BACKLOG_PATH,
        FOLLOWUP_BACKLOG_PATH,
        POLICY_PATH,
        CORPUS_REPORT_PATH,
        TOKENIZER_MANIFEST_PATH,
        HISTORICAL_PLAN_PATH,
        REGISTRY_PATH,
        *HISTORICAL_ARTIFACT_MANIFEST_PATHS,
        TRAINING_DATA_PATH,
        TRAINER_PATH,
        TRAIN_TOOL_PATH,
        Path(__file__),
    )
    return {
        "master_backlog": master,
        "followup_backlog": followup,
        "policy": policy,
        "corpus": _corpus_evidence(corpus_report, required_stage),
        "tokenizer": _tokenizer_evidence(tokenizer_manifest, policy),
        "historical": _historical_evidence(
            historical_plan, registry, historical_manifests
        ),
        "completion_inputs": _completion_input_evidence(policy),
        "source_sha256": {
            path.relative_to(AI_ROOT).as_posix(): _sha256_file(path)
            for path in source_paths
        },
    }


def _build_current_report(generated_on: str) -> dict[str, Any]:
    return build_report(generated_on=generated_on, **_current_inputs())


def evaluate(path: Path, generated_on: str) -> dict[str, Any]:
    report = _build_current_report(generated_on)
    _write_json(report, path)
    return report


def verify(path: Path) -> dict[str, Any]:
    report = _read_json(path, "VFAI-FU-013 readiness receipt")
    validate_report(report)
    if report != _build_current_report(str(report.get("generatedOn"))):
        raise Gen1V11RevisionReadinessError("VFAI-FU-013 readiness receipt is stale")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--generated-on", required=True)
    evaluate_parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--input", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = (
        evaluate(args.output.resolve(), args.generated_on)
        if args.command == "evaluate"
        else verify(args.input.resolve())
    )
    print(
        json.dumps(
            {
                "ok": True,
                "command": args.command,
                "reportId": report["reportId"],
                "decision": report["decision"],
                "tokenizerVersion": report["approvedTrainingTokenizer"]["version"],
                "currentUniqueTrainingPredictedTokens": report["corpus"].get(
                    "currentUniqueTrainingPredictedTokens"
                ),
                "trainingRunsAllocatedByEvaluator": report[
                    "trainingRunsAllocatedByEvaluator"
                ],
                "reportSha256": report["reportSha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
