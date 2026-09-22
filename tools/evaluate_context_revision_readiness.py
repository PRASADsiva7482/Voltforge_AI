"""Generate or verify the VFAI-FU-008 context-revision readiness receipt.

The evaluator reads content-free manifests and receipts only. It does not load
model weights, training records, prompts, or project content; allocate a model;
run training; access the network; or approve a release or registry activation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

MASTER_BACKLOG_PATH = AI_ROOT / "AI_MASTER_BACKLOG.json"
FOLLOWUP_BACKLOG_PATH = AI_ROOT / "AI_FOLLOWUP_BACKLOG.json"
POLICY_PATH = AI_ROOT / "gen1_training/context-revision-policy.v1.json"
CORPUS_REPORT_PATH = AI_ROOT / "evaluation/reports/corpus-expansion-readiness-v1.json"
HISTORICAL_CONTEXT_POLICY_PATH = AI_ROOT / "context_compiler/policy.v1.json"
HISTORICAL_CONTEXT_REPORT_PATH = (
    AI_ROOT / "evaluation/reports/project-context-compiler-v1.json"
)
HISTORICAL_CONTEXT_SNAPSHOT_PATH = (
    AI_ROOT / "evaluation/snapshots/project-context-compiler-v1.json"
)
TRAINING_TOKENIZER_MANIFEST_PATH = (
    AI_ROOT / "model/tokenizers/vfdlm-byte-bpe-v1.1.0/tokenizer_manifest.json"
)
REGISTRY_PATH = AI_ROOT / "model/registry/active_model.json"
BASELINE_CONFIG_PATH = AI_ROOT / "model/gen1/configs/gen1-selected-v1.json"
DEFAULT_REPORT_PATH = AI_ROOT / "evaluation/reports/context-revision-readiness-v1.json"
COMPLETE_STATUSES = frozenset({"complete", "completed", "done"})


class ContextRevisionReadinessError(RuntimeError):
    """The context-revision readiness evidence is invalid, stale, or unsafe."""


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _declared_digest_valid(value: Mapping[str, Any], field: str) -> bool:
    declared = value.get(field)
    unsigned = dict(value)
    unsigned.pop(field, None)
    return isinstance(declared, str) and declared == hashlib.sha256(
        _canonical_json(unsigned)
    ).hexdigest()


def _receipt_digest(report: Mapping[str, Any]) -> str:
    unsigned = dict(report)
    unsigned.pop("reportSha256", None)
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest()


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContextRevisionReadinessError(f"unable to read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContextRevisionReadinessError(f"{label} must be a JSON object")
    return value


def _write_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _resolve_workspace_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ContextRevisionReadinessError(f"{label} path is missing")
    path = (AI_ROOT / value).resolve()
    try:
        path.relative_to(AI_ROOT.resolve())
    except ValueError as exc:
        raise ContextRevisionReadinessError(f"{label} path escapes the AI workspace") from exc
    return path


def _backlog_item(backlog: Mapping[str, Any], item_id: str) -> Mapping[str, Any]:
    items = backlog.get("items")
    if not isinstance(items, list):
        raise ContextRevisionReadinessError("follow-up backlog items are invalid")
    matches = [
        item
        for item in items
        if isinstance(item, Mapping) and item.get("id") == item_id
    ]
    if len(matches) != 1:
        raise ContextRevisionReadinessError(f"expected exactly one {item_id} item")
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


def _corpus_evidence(
    report: Mapping[str, Any], policy: Mapping[str, Any]
) -> dict[str, Any]:
    stages = report.get("stages")
    ready_stage_ids = [
        str(stage.get("stageId"))
        for stage in stages or []
        if isinstance(stage, Mapping)
        and stage.get("readyForThreeSeedLearningCurve") is True
    ]
    dependencies = policy.get("dependencies")
    if not isinstance(dependencies, Mapping):
        raise ContextRevisionReadinessError("context revision dependencies are invalid")
    required = {
        "pilot": dependencies.get("minimumPilotCorpusStage"),
        "selection": dependencies.get("minimumSelectionCorpusStage"),
        "release": dependencies.get("minimumReleaseCorpusStage"),
    }
    receipt_valid = (
        report.get("reportId") == "vfai-fu-006-corpus-expansion-readiness-v1"
        and _declared_digest_valid(report, "reportSha256")
        and report.get("completionClaimed") is False
        and report.get("networkAccessed") is False
        and report.get("governedCorpusExported") is False
    )
    return {
        "reportId": report.get("reportId"),
        "reportSha256": report.get("reportSha256"),
        "receiptValid": receipt_valid,
        "decision": report.get("decision"),
        "currentApprovedStage": report.get("currentApprovedStage"),
        "readyStageIds": ready_stage_ids,
        "currentUniqueTrainingPredictedTokens": report.get("current", {}).get(
            "uniqueTrainingPredictedTokens"
        ),
        "requiredStages": required,
        "pilotStageReady": required["pilot"] in ready_stage_ids,
        "selectionStageReady": required["selection"] in ready_stage_ids,
        "releaseStageReady": required["release"] in ready_stage_ids,
    }


def _historical_context_evidence(
    context_policy: Mapping[str, Any],
    report: Mapping[str, Any],
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    policy_valid = _declared_digest_valid(context_policy, "policySha256")
    report_valid = _declared_digest_valid(report, "reportSha256")
    snapshot_valid = _declared_digest_valid(snapshot, "snapshotSha256")
    binding_valid = bool(
        report.get("policyId") == context_policy.get("policyId")
        and report.get("policySha256") == context_policy.get("policySha256")
        and report.get("snapshotSha256") == snapshot.get("snapshotSha256")
        and snapshot.get("policyId") == context_policy.get("policyId")
        and snapshot.get("policySha256") == context_policy.get("policySha256")
        and report.get("rawPromptStored") is False
        and report.get("rawProjectContextStored") is False
        and snapshot.get("rawPromptStored") is False
        and snapshot.get("rawProjectContextStored") is False
        and report.get("neuralServingApproved") is False
    )
    return {
        "policyId": context_policy.get("policyId"),
        "policySha256": context_policy.get("policySha256"),
        "compilerVersion": context_policy.get("compilerVersion"),
        "tokenizerVersion": context_policy.get("tokenizer", {}).get("version"),
        "minimumSupportedContextWindowTokens": context_policy.get("budget", {}).get(
            "minimumSupportedContextWindowTokens"
        ),
        "snapshotSha256": snapshot.get("snapshotSha256"),
        "snapshotContextWindowTokens": snapshot.get("contextWindowTokens"),
        "snapshotReservedOutputTokens": snapshot.get("reservedOutputTokens"),
        "snapshotPromptTokens": snapshot.get("promptTokens"),
        "artifactContextWindows": report.get("artifactContextWindows"),
        "policyChecksumValid": policy_valid,
        "reportChecksumValid": report_valid,
        "snapshotChecksumValid": snapshot_valid,
        "bindingsValid": binding_valid,
        "ready": policy_valid and report_valid and snapshot_valid and binding_valid,
        "mayBeRewrittenInPlace": False,
    }


def _tokenizer_evidence(manifest: Mapping[str, Any]) -> dict[str, Any]:
    checksum_valid = _declared_digest_valid(manifest, "artifactSha256")
    ready = bool(
        checksum_valid
        and manifest.get("tokenizerId") == "vfdlm-byte-bpe"
        and manifest.get("version") == "1.1.0"
        and manifest.get("releaseStatus") == "approved"
        and manifest.get("algorithm") == "voltforge-byte-bpe"
        and manifest.get("vocabSize") == 3072
    )
    return {
        "tokenizerId": manifest.get("tokenizerId"),
        "version": manifest.get("version"),
        "releaseStatus": manifest.get("releaseStatus"),
        "vocabSize": manifest.get("vocabSize"),
        "artifactSha256": manifest.get("artifactSha256"),
        "checksumValid": checksum_valid,
        "ready": ready,
    }


def _artifact_evidence(
    registry: Mapping[str, Any],
    baseline: Mapping[str, Any],
    minimum_context: int,
) -> dict[str, Any]:
    artifacts = registry.get("artifacts")
    contexts = {
        str(item.get("artifactId")): item.get("contextLength")
        for item in artifacts or []
        if isinstance(item, Mapping)
    }
    all_contexts_numeric = bool(contexts) and all(
        isinstance(value, int) and not isinstance(value, bool) for value in contexts.values()
    )
    baseline_valid = _declared_digest_valid(baseline, "manifestSha256")
    baseline_context = baseline.get("modelConfig", {}).get("maxSequenceLength")
    all_inactive = bool(
        isinstance(artifacts, list)
        and artifacts
        and all(
            isinstance(item, Mapping)
            and item.get("activationEligible") is False
            and item.get("releaseStatus") == "experimental"
            for item in artifacts
        )
        and registry.get("activeArtifactId") is None
    )
    all_below = all_contexts_numeric and all(
        int(value) < minimum_context for value in contexts.values()
    )
    return {
        "registryRevision": registry.get("revision"),
        "registrySha256": registry.get("registrySha256"),
        "activeArtifactId": registry.get("activeArtifactId"),
        "artifactContextWindows": contexts,
        "allExistingArtifactsBelowMinimum": all_below,
        "allExistingArtifactsInactive": all_inactive,
        "baselineCandidateId": baseline.get("candidateId"),
        "baselineContextLength": baseline_context,
        "baselineConfigChecksumValid": baseline_valid,
        "newArtifactRequired": all_below and all_inactive and baseline_context == 128,
        "ready": all_below and all_inactive and baseline_valid and baseline_context == 128,
    }


def _optional_json(path_value: Any, label: str) -> tuple[str | None, dict[str, Any] | None]:
    if path_value is None:
        return None, None
    path = _resolve_workspace_path(path_value, label)
    if not path.is_file():
        return str(path_value), None
    return str(path_value), _read_json_object(path, label)


def _migration_evidence(
    policy: Mapping[str, Any],
    historical: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
) -> dict[str, Any]:
    migration = policy.get("compatibilityMigration")
    if not isinstance(migration, Mapping):
        raise ContextRevisionReadinessError("compatibility migration policy is invalid")
    approved_version = tokenizer.get("version")
    historical_version = historical.get("tokenizerVersion")
    policy_path, candidate_policy = _optional_json(
        migration.get("candidateCompilerPolicyPath"), "candidate compiler policy"
    )
    policy_checksum_valid = bool(
        candidate_policy is not None
        and _declared_digest_valid(candidate_policy, "policySha256")
    )
    candidate_policy_ready = bool(
        policy_checksum_valid
        and candidate_policy.get("policyId") != historical.get("policyId")
        and candidate_policy.get("tokenizer", {}).get("tokenizerId")
        == tokenizer.get("tokenizerId")
        and candidate_policy.get("tokenizer", {}).get("version") == approved_version
        and candidate_policy.get("privacy", {}).get("rawPromptLogged") is False
        and candidate_policy.get("privacy", {}).get("rawProjectContextLogged") is False
    )
    snapshot_path, candidate_snapshot = _optional_json(
        migration.get("candidateCompilerSnapshotPath"), "candidate compiler snapshot"
    )
    snapshot_checksum_valid = bool(
        candidate_snapshot is not None
        and _declared_digest_valid(candidate_snapshot, "snapshotSha256")
    )
    snapshot_ready = bool(
        snapshot_checksum_valid
        and candidate_policy_ready
        and candidate_snapshot.get("policyId") == candidate_policy.get("policyId")
        and candidate_snapshot.get("policySha256") == candidate_policy.get("policySha256")
        and candidate_snapshot.get("tokenizerId") == tokenizer.get("tokenizerId")
        and candidate_snapshot.get("tokenizerVersion") == approved_version
        and candidate_snapshot.get("rawPromptStored") is False
        and candidate_snapshot.get("rawProjectContextStored") is False
        and isinstance(candidate_snapshot.get("promptTokens"), int)
        and candidate_snapshot.get("promptTokens") > 0
    )
    return {
        "historicalTokenizerVersion": historical_version,
        "approvedTrainingTokenizerVersion": approved_version,
        "historicalBindingMatchesApprovedTrainingTokenizer": historical_version
        == approved_version,
        "migrationRequired": historical_version != approved_version,
        "candidatePolicy": {
            "path": policy_path,
            "assigned": policy_path is not None,
            "exists": candidate_policy is not None,
            "checksumValid": policy_checksum_valid,
            "ready": candidate_policy_ready,
            "policyId": candidate_policy.get("policyId") if candidate_policy else None,
        },
        "candidateSnapshot": {
            "path": snapshot_path,
            "assigned": snapshot_path is not None,
            "exists": candidate_snapshot is not None,
            "checksumValid": snapshot_checksum_valid,
            "ready": snapshot_ready,
            "snapshotSha256": candidate_snapshot.get("snapshotSha256")
            if candidate_snapshot
            else None,
        },
        "ready": candidate_policy_ready and snapshot_ready,
    }


def _owner_budget_evidence(policy: Mapping[str, Any]) -> dict[str, Any]:
    contract = policy.get("ownerComputeBudget")
    if not isinstance(contract, Mapping):
        raise ContextRevisionReadinessError("owner compute budget policy is invalid")
    path_value, receipt = _optional_json(
        contract.get("approvalReceiptPath"), "owner compute budget receipt"
    )
    checksum_valid = bool(
        receipt is not None and _declared_digest_valid(receipt, "receiptSha256")
    )
    finite_fields = (
        "maximumStorageBytes",
        "maximumTrainingDeviceHours",
        "maximumEvaluationDeviceHours",
    )
    finite_budget = bool(
        receipt is not None
        and all(
            isinstance(receipt.get(field), (int, float))
            and not isinstance(receipt.get(field), bool)
            and math.isfinite(float(receipt[field]))
            and float(receipt[field]) > 0
            for field in finite_fields
        )
    )
    ready = bool(
        checksum_valid
        and receipt.get("receiptKind") == "vfai-fu-008-owner-compute-budget-v1"
        and receipt.get("policyId") == policy.get("policyId")
        and receipt.get("ownerApproved") is True
        and receipt.get("hardwareBoundaryApproved") is True
        and finite_budget
        and receipt.get("containsSecret") is False
    )
    return {
        "path": path_value,
        "assigned": path_value is not None,
        "exists": receipt is not None,
        "checksumValid": checksum_valid,
        "finiteBudgetRecorded": finite_budget,
        "ready": ready,
    }


def _candidate_assets_evidence(
    policy: Mapping[str, Any], tokenizer: Mapping[str, Any]
) -> dict[str, Any]:
    candidates = policy.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ContextRevisionReadinessError("context candidates are invalid")
    expected_ids = {f"context-{value}" for value in (1024, 2048, 4096, 8192)}
    if {item.get("candidateId") for item in candidates if isinstance(item, Mapping)} != expected_ids:
        raise ContextRevisionReadinessError("context candidate matrix is incomplete")
    evidence = []
    raw_sets: set[str] = set()
    split_ids: set[str] = set()
    for candidate in candidates:
        candidate_id = str(candidate.get("candidateId"))
        context_length = candidate.get("contextLength")
        manifest_path, manifest = _optional_json(
            candidate.get("candidateManifestPath"), f"{candidate_id} candidate manifest"
        )
        packing_path, packing = _optional_json(
            candidate.get("packingManifestPath"), f"{candidate_id} packing manifest"
        )
        manifest_valid = bool(
            manifest is not None
            and _declared_digest_valid(manifest, "manifestSha256")
            and manifest.get("candidateId") == candidate_id
            and manifest.get("contextLength") == context_length
            and manifest.get("tokenizerArtifactSha256") == tokenizer.get("artifactSha256")
            and manifest.get("fromScratch") is True
            and manifest.get("randomInitializationOwnedByProject") is True
            and manifest.get("seedIds") == policy.get("experimentProtocol", {}).get(
                "seedIds"
            )
            and manifest.get("pretrainedWeightsUsed") is False
            and manifest.get("releaseOrActivationApproved") is False
        )
        packing_valid = bool(
            packing is not None
            and _declared_digest_valid(packing, "manifestSha256")
            and packing.get("candidateId") == candidate_id
            and packing.get("contextLength") == context_length
            and packing.get("tokenizerArtifactSha256") == tokenizer.get("artifactSha256")
            and packing.get("approvedSourcesOnly") is True
            and packing.get("heldOutCollisionCount") == 0
            and packing.get("longPositionOccupancyMeasured") is True
            and packing.get("rawRecordSetSha256")
            and packing.get("splitId")
        )
        if packing_valid:
            raw_sets.add(str(packing.get("rawRecordSetSha256")))
            split_ids.add(str(packing.get("splitId")))
        evidence.append(
            {
                "candidateId": candidate_id,
                "contextLength": context_length,
                "candidateManifestPath": manifest_path,
                "candidateManifestAssigned": manifest_path is not None,
                "candidateManifestValid": manifest_valid,
                "packingManifestPath": packing_path,
                "packingManifestAssigned": packing_path is not None,
                "packingManifestValid": packing_valid,
                "ready": manifest_valid and packing_valid,
            }
        )
    all_individually_ready = all(item["ready"] for item in evidence)
    identical_exposure = all_individually_ready and len(raw_sets) == 1 and len(split_ids) == 1
    return {
        "candidates": evidence,
        "allCandidateManifestsReady": all(
            item["candidateManifestValid"] for item in evidence
        ),
        "allPackingManifestsReady": all(item["packingManifestValid"] for item in evidence),
        "identicalRawRecordAndSplitExposure": identical_exposure,
        "ready": all_individually_ready and identical_exposure,
    }


def _result_evidence(policy: Mapping[str, Any]) -> dict[str, Any]:
    protocol = policy.get("experimentProtocol")
    if not isinstance(protocol, Mapping):
        raise ContextRevisionReadinessError("context experiment protocol is invalid")
    path_value, report = _optional_json(
        protocol.get("candidateResultReportPath"), "context candidate result report"
    )
    checksum_valid = bool(
        report is not None and _declared_digest_valid(report, "reportSha256")
    )
    candidates = report.get("candidates") if report else None
    result_ids = {
        item.get("candidateId")
        for item in candidates or []
        if isinstance(item, Mapping)
    }
    expected_ids = {f"context-{value}" for value in (1024, 2048, 4096, 8192)}
    candidate_gates_pass = bool(
        isinstance(candidates, list)
        and result_ids == expected_ids
        and all(
            isinstance(item, Mapping)
            and item.get("independentSeedCount", 0)
            >= int(protocol.get("minimumIndependentSeeds", 3))
            and isinstance(item.get("acceptanceGates"), Mapping)
            and item.get("acceptanceGates")
            and all(value is True for value in item["acceptanceGates"].values())
            for item in candidates
        )
    )
    selected_context = report.get("selectedContextLength") if report else None
    ready = bool(
        checksum_valid
        and report.get("reportKind") == "vfai-fu-008-context-candidate-results-v1"
        and report.get("policyId") == policy.get("policyId")
        and report.get("corpusStage")
        == policy.get("dependencies", {}).get("minimumReleaseCorpusStage")
        and candidate_gates_pass
        and isinstance(selected_context, int)
        and selected_context >= policy.get("promotion", {}).get(
            "minimumSelectableContextLength", 4096
        )
        and report.get("networkAccessed") is False
        and report.get("governedDataExported") is False
        and report.get("releaseOrActivationApproved") is False
    )
    return {
        "path": path_value,
        "assigned": path_value is not None,
        "exists": report is not None,
        "checksumValid": checksum_valid,
        "candidateMatrixComplete": result_ids == expected_ids,
        "allCandidateAcceptanceGatesPassed": candidate_gates_pass,
        "selectedContextLength": selected_context,
        "ready": ready,
    }


def build_report(
    *,
    master_backlog: Mapping[str, Any],
    followup_backlog: Mapping[str, Any],
    policy: Mapping[str, Any],
    corpus: Mapping[str, Any],
    historical_context: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    migration: Mapping[str, Any],
    owner_budget: Mapping[str, Any],
    candidate_assets: Mapping[str, Any],
    results: Mapping[str, Any],
    generated_on: str,
    source_sha256: Mapping[str, str],
) -> dict[str, Any]:
    if policy.get("sourceFollowup") != "VFAI-FU-008":
        raise ContextRevisionReadinessError("policy is not bound to VFAI-FU-008")
    fu008 = _backlog_item(followup_backlog, "VFAI-FU-008")
    master_ready = _master_complete(master_backlog)
    source_evidence_ready = bool(
        corpus.get("receiptValid") is True
        and historical_context.get("ready") is True
        and tokenizer.get("ready") is True
        and artifacts.get("ready") is True
    )
    gates = {
        "masterBacklogComplete": master_ready,
        "sourceEvidenceReady": source_evidence_ready,
        "pilotCorpusStageReady": corpus.get("pilotStageReady") is True,
        "selectionCorpusStageReady": corpus.get("selectionStageReady") is True,
        "releaseCorpusStageReady": corpus.get("releaseStageReady") is True,
        "ownerComputeBudgetApproved": owner_budget.get("ready") is True,
        "tokenizerBoundCompilerRevisionReady": migration.get("ready") is True,
        "allCandidateAndPackingManifestsReady": candidate_assets.get("ready") is True,
        "candidateResultsReady": results.get("ready") is True,
    }
    if not master_ready:
        decision = "await-master-backlog-completion"
    elif not source_evidence_ready:
        decision = "repair-source-readiness-evidence"
    elif gates["pilotCorpusStageReady"] is not True:
        decision = "await-governed-corpus-stage"
    elif gates["ownerComputeBudgetApproved"] is not True:
        decision = "await-owner-compute-budget"
    elif gates["tokenizerBoundCompilerRevisionReady"] is not True:
        decision = "implement-tokenizer-bound-context-compiler-revision"
    elif gates["allCandidateAndPackingManifestsReady"] is not True:
        decision = "implement-long-context-candidate-assets"
    elif gates["releaseCorpusStageReady"] is not True:
        decision = "await-release-corpus-stage"
    elif gates["candidateResultsReady"] is not True:
        decision = "ready-to-run-owned-context-candidate-matrix"
    else:
        decision = "context-results-ready-for-independent-release-review"
    report: dict[str, Any] = {
        "schemaVersion": 1,
        "reportId": "vfai-fu-008-context-revision-readiness-v1",
        "generatedOn": generated_on,
        "followup": {"id": fu008.get("id"), "recordedStatus": fu008.get("status")},
        "policy": {
            "policyId": policy.get("policyId"),
            "status": policy.get("status"),
            "candidateContextLengths": [
                item.get("contextLength") for item in policy.get("candidates", [])
            ],
            "minimumIndependentSeeds": policy.get("experimentProtocol", {}).get(
                "minimumIndependentSeeds"
            ),
        },
        "corpus": dict(corpus),
        "historicalContext": dict(historical_context),
        "approvedTrainingTokenizer": dict(tokenizer),
        "existingArtifacts": dict(artifacts),
        "compatibilityMigration": dict(migration),
        "ownerComputeBudget": dict(owner_budget),
        "candidateAssets": dict(candidate_assets),
        "candidateResults": dict(results),
        "gates": gates,
        "decision": decision,
        "completionClaimed": False,
        "modelWeightsLoadedByEvaluator": False,
        "governedCorpusLoadedByEvaluator": False,
        "trainingRunsExecutedByEvaluator": 0,
        "candidateArtifactsCreatedByEvaluator": 0,
        "rawPromptOrProjectContentStored": False,
        "networkAccessed": False,
        "releaseOrActivationApproved": False,
        "sourceSha256": dict(source_sha256),
        "reportSha256": "",
    }
    report["reportSha256"] = _receipt_digest(report)
    return report


def _current_inputs() -> dict[str, Any]:
    master = _read_json_object(MASTER_BACKLOG_PATH, "master backlog")
    followup = _read_json_object(FOLLOWUP_BACKLOG_PATH, "follow-up backlog")
    policy = _read_json_object(POLICY_PATH, "context revision policy")
    corpus_report = _read_json_object(CORPUS_REPORT_PATH, "corpus readiness report")
    context_policy = _read_json_object(
        HISTORICAL_CONTEXT_POLICY_PATH, "historical context policy"
    )
    context_report = _read_json_object(
        HISTORICAL_CONTEXT_REPORT_PATH, "historical context report"
    )
    context_snapshot = _read_json_object(
        HISTORICAL_CONTEXT_SNAPSHOT_PATH, "historical context snapshot"
    )
    tokenizer_manifest = _read_json_object(
        TRAINING_TOKENIZER_MANIFEST_PATH, "approved training tokenizer manifest"
    )
    registry = _read_json_object(REGISTRY_PATH, "model registry")
    baseline = _read_json_object(BASELINE_CONFIG_PATH, "baseline model config")
    corpus = _corpus_evidence(corpus_report, policy)
    historical = _historical_context_evidence(
        context_policy, context_report, context_snapshot
    )
    tokenizer = _tokenizer_evidence(tokenizer_manifest)
    minimum_context = int(historical.get("minimumSupportedContextWindowTokens", 0))
    artifacts = _artifact_evidence(registry, baseline, minimum_context)
    source_paths = (
        MASTER_BACKLOG_PATH,
        FOLLOWUP_BACKLOG_PATH,
        POLICY_PATH,
        CORPUS_REPORT_PATH,
        HISTORICAL_CONTEXT_POLICY_PATH,
        HISTORICAL_CONTEXT_REPORT_PATH,
        HISTORICAL_CONTEXT_SNAPSHOT_PATH,
        TRAINING_TOKENIZER_MANIFEST_PATH,
        REGISTRY_PATH,
        BASELINE_CONFIG_PATH,
        Path(__file__),
    )
    return {
        "master_backlog": master,
        "followup_backlog": followup,
        "policy": policy,
        "corpus": corpus,
        "historical_context": historical,
        "tokenizer": tokenizer,
        "artifacts": artifacts,
        "migration": _migration_evidence(policy, historical, tokenizer),
        "owner_budget": _owner_budget_evidence(policy),
        "candidate_assets": _candidate_assets_evidence(policy, tokenizer),
        "results": _result_evidence(policy),
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


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("reportSha256") != _receipt_digest(report):
        raise ContextRevisionReadinessError("VFAI-FU-008 receipt checksum is invalid")
    gates = report.get("gates")
    if not isinstance(gates, Mapping):
        raise ContextRevisionReadinessError("VFAI-FU-008 readiness gates are invalid")
    if gates.get("masterBacklogComplete") is not True:
        expected = "await-master-backlog-completion"
    elif gates.get("sourceEvidenceReady") is not True:
        expected = "repair-source-readiness-evidence"
    elif gates.get("pilotCorpusStageReady") is not True:
        expected = "await-governed-corpus-stage"
    elif gates.get("ownerComputeBudgetApproved") is not True:
        expected = "await-owner-compute-budget"
    elif gates.get("tokenizerBoundCompilerRevisionReady") is not True:
        expected = "implement-tokenizer-bound-context-compiler-revision"
    elif gates.get("allCandidateAndPackingManifestsReady") is not True:
        expected = "implement-long-context-candidate-assets"
    elif gates.get("releaseCorpusStageReady") is not True:
        expected = "await-release-corpus-stage"
    elif gates.get("candidateResultsReady") is not True:
        expected = "ready-to-run-owned-context-candidate-matrix"
    else:
        expected = "context-results-ready-for-independent-release-review"
    if report.get("decision") != expected:
        raise ContextRevisionReadinessError("VFAI-FU-008 receipt decision is inconsistent")
    if any(
        (
            report.get("completionClaimed") is not False,
            report.get("modelWeightsLoadedByEvaluator") is not False,
            report.get("governedCorpusLoadedByEvaluator") is not False,
            report.get("trainingRunsExecutedByEvaluator") != 0,
            report.get("candidateArtifactsCreatedByEvaluator") != 0,
            report.get("rawPromptOrProjectContentStored") is not False,
            report.get("networkAccessed") is not False,
            report.get("releaseOrActivationApproved") is not False,
        )
    ):
        raise ContextRevisionReadinessError("VFAI-FU-008 receipt overclaims execution")


def verify(path: Path) -> dict[str, Any]:
    report = _read_json_object(path, "VFAI-FU-008 readiness receipt")
    validate_report(report)
    expected = _build_current_report(str(report.get("generatedOn")))
    if report != expected:
        raise ContextRevisionReadinessError("VFAI-FU-008 readiness receipt is stale")
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
                "currentApprovedCorpusStage": report["corpus"]["currentApprovedStage"],
                "currentUniqueTrainingPredictedTokens": report["corpus"][
                    "currentUniqueTrainingPredictedTokens"
                ],
                "historicalCompilerTokenizerVersion": report["historicalContext"][
                    "tokenizerVersion"
                ],
                "approvedTrainingTokenizerVersion": report[
                    "approvedTrainingTokenizer"
                ]["version"],
                "reportSha256": report["reportSha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
