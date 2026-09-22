"""Fail-closed VFAI-015 scale, revise, or stop analysis."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping
import uuid

from gen1_bootstrap import check_bootstrap_scorecard
from gen1_sweep import check_scorecard


AI_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PATH = AI_ROOT / "gen1_decision" / "policy.v1.json"
DEFAULT_REPORT_PATH = AI_ROOT / "model" / "gen1" / "decisions" / "gen1-scale-decision-v1.json"


class ScaleDecisionContractError(RuntimeError):
    """Raised when decision policy or bound evidence is inconsistent."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ScaleDecisionContractError(f"unable to read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ScaleDecisionContractError(f"{label} must be a JSON object")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _resolve_repo_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ScaleDecisionContractError(f"{label} path is missing")
    path = (AI_ROOT / value).resolve()
    try:
        path.relative_to(AI_ROOT.resolve())
    except ValueError as exc:
        raise ScaleDecisionContractError(f"{label} path escapes the AI workspace") from exc
    return path


def _portable_path(path: Path) -> str:
    return path.resolve().relative_to(AI_ROOT.resolve()).as_posix()


def _verify_self_hash(value: Mapping[str, Any], key: str, label: str) -> None:
    if value.get(key) != _fingerprint({name: item for name, item in value.items() if name != key}):
        raise ScaleDecisionContractError(f"{label} self-checksum changed")


def load_scale_policy(path: str | Path = DEFAULT_POLICY_PATH) -> dict[str, Any]:
    policy = _read_json(Path(path), "Gen1 scale-decision policy")
    expected = {
        "schemaVersion",
        "policyId",
        "purpose",
        "evidence",
        "profiles",
        "releaseGates",
        "largerProfileGates",
        "hypothesisRules",
        "decisionOptions",
        "fallbackPolicy",
    }
    if set(policy) != expected or policy.get("schemaVersion") != 1:
        raise ScaleDecisionContractError("scale-decision policy keys/schema are invalid")
    evidence_contract = policy["evidence"]
    if set(evidence_contract) != {
        "architectureSweep",
        "bootstrapScorecard",
        "bestBootstrapConfig",
        "globalMetricPolicy",
        "activeRegistry",
    }:
        raise ScaleDecisionContractError("scale-decision evidence contract is invalid")

    resolved: dict[str, dict[str, Any]] = {}
    for key in (
        "architectureSweep",
        "bootstrapScorecard",
        "bestBootstrapConfig",
        "globalMetricPolicy",
    ):
        contract = evidence_contract[key]
        evidence_path = _resolve_repo_path(contract.get("path"), key)
        if _sha256_file(evidence_path) != contract.get("fileSha256"):
            raise ScaleDecisionContractError(f"{key} file checksum changed")
        resolved[key] = _read_json(evidence_path, key)
    check_scorecard()
    check_bootstrap_scorecard()
    architecture = resolved["architectureSweep"]
    bootstrap = resolved["bootstrapScorecard"]
    best = resolved["bestBootstrapConfig"]
    if architecture.get("reportSha256") != evidence_contract["architectureSweep"].get(
        "reportSha256"
    ):
        raise ScaleDecisionContractError("architecture sweep report checksum changed")
    if bootstrap.get("reportSha256") != evidence_contract["bootstrapScorecard"].get(
        "reportSha256"
    ):
        raise ScaleDecisionContractError("bootstrap report checksum changed")
    _verify_self_hash(best, "manifestSha256", "best bootstrap config")
    if best.get("manifestSha256") != evidence_contract["bestBootstrapConfig"].get(
        "manifestSha256"
    ):
        raise ScaleDecisionContractError("best bootstrap selection checksum changed")
    if (
        best.get("releaseApproved") is not False
        or bootstrap.get("releaseBoundary", {}).get("releaseApproved") is not False
        or best.get("step") != bootstrap.get("selection", {}).get("selectedStep")
    ):
        raise ScaleDecisionContractError("bootstrap release/selection boundary changed")

    metric_policy = resolved["globalMetricPolicy"]
    metrics = metric_policy.get("metrics")
    if (
        not isinstance(metrics, list)
        or len(metrics) != policy["releaseGates"]["requiredGlobalMetricCount"]
        or len({item.get("id") for item in metrics}) != len(metrics)
    ):
        raise ScaleDecisionContractError("global metric count/IDs changed")

    profiles = policy["profiles"]
    if (
        not isinstance(profiles, list)
        or [item.get("name") for item in profiles] != ["edge", "core", "server"]
        or [item.get("minimumApprovedTrainingTokens") for item in profiles]
        != [100_000_000, 500_000_000, 1_200_000_000]
    ):
        raise ScaleDecisionContractError("deployment profile data targets changed")
    ranges = [(8_000_000, 25_000_000), (25_000_000, 60_000_000), (60_000_000, 150_000_000)]
    for profile, expected_range in zip(profiles, ranges, strict=True):
        actual_range = profile.get("parameterRange", {})
        if (actual_range.get("minimum"), actual_range.get("maximum")) != expected_range:
            raise ScaleDecisionContractError("deployment profile parameter range changed")
    release = policy["releaseGates"]
    if (
        release.get("minimumDataCoveragePercent") != 100.0
        or release.get("requiredGlobalMetricsPassing") != len(metrics)
        or release.get("maximumCriticalCaseFailures") != 0
        or not all(
            release.get(key) is True
            for key in (
                "requireApprovedGenerationRuntime",
                "requireApprovedArtifactActivationPath",
                "requireReleaseApprovedCheckpoint",
            )
        )
    ):
        raise ScaleDecisionContractError("global release-gate contract changed")
    larger = policy["largerProfileGates"]
    if (
        larger.get("minimumIndependentSeeds", 0) < 3
        or larger.get("confidenceLevel") != 0.95
        or larger.get("minimumCriticalDomainCompositeImprovementPoints", 0) <= 0
        or larger.get("minimumBootstrapProxyLossImprovementPercent") != 1.0
        or not all(
            larger.get(key) is True
            for key in (
                "requireConfidenceLowerBoundAboveZero",
                "requireNoCriticalMetricRegression",
                "requireSmallerProfileAtDataTarget",
                "requireSmallerProfileOptimizationPlateau",
                "requireMeasuredTargetHardware",
            )
        )
    ):
        raise ScaleDecisionContractError("larger-profile gate contract changed")
    if policy["decisionOptions"] != [
        "edge-neural-release",
        "core-neural-release",
        "server-neural-release",
        "no-neural-release",
    ]:
        raise ScaleDecisionContractError("decision option contract changed")
    fallback = policy["fallbackPolicy"]
    if (
        fallback.get("selectedWhenAnyReleaseGateFails") != "no-neural-release"
        or fallback.get("activeRegistryMustRemainNull") is not True
        or fallback.get("deterministicToolsRemainAuthoritative") is not True
    ):
        raise ScaleDecisionContractError("fallback safety contract changed")
    registry_path = _resolve_repo_path(
        evidence_contract["activeRegistry"].get("path"), "active registry"
    )
    if not registry_path.is_file():
        raise ScaleDecisionContractError("active registry is missing")
    policy["policyFingerprint"] = _fingerprint(policy)
    policy["resolvedEvidence"] = resolved
    return policy


def evaluate_larger_profile_eligibility(
    observed: Mapping[str, Any], gates: Mapping[str, Any]
) -> dict[str, Any]:
    expected = {
        "dataCoveragePercent",
        "independentSeeds",
        "criticalDomainCompositeImprovementPoints",
        "confidenceLowerBoundImprovementPoints",
        "bootstrapProxyLossImprovementPercent",
        "criticalMetricRegressions",
        "smallerProfileAtDataTarget",
        "smallerProfileOptimizationPlateau",
        "targetHardwareMeasured",
        "candidateGlobalMetricsPassing",
        "candidateCriticalCaseFailures",
        "generationRuntimeApproved",
    }
    if set(observed) != expected:
        raise ScaleDecisionContractError("larger-profile observation keys are invalid")
    failures = []
    checks = [
        (observed["dataCoveragePercent"] >= 100.0, "INSUFFICIENT_APPROVED_DATA"),
        (
            observed["independentSeeds"] >= gates["minimumIndependentSeeds"],
            "INSUFFICIENT_INDEPENDENT_SEEDS",
        ),
        (
            observed["criticalDomainCompositeImprovementPoints"]
            >= gates["minimumCriticalDomainCompositeImprovementPoints"],
            "CRITICAL_DOMAIN_GAIN_NOT_MATERIAL",
        ),
        (
            observed["confidenceLowerBoundImprovementPoints"] > 0.0,
            "GAIN_NOT_ABOVE_NOISE",
        ),
        (
            observed["bootstrapProxyLossImprovementPercent"]
            >= gates["minimumBootstrapProxyLossImprovementPercent"],
            "BOOTSTRAP_PROXY_GAIN_NOT_MATERIAL",
        ),
        (observed["criticalMetricRegressions"] == 0, "CRITICAL_METRIC_REGRESSION"),
        (observed["smallerProfileAtDataTarget"] is True, "SMALLER_PROFILE_DATA_NOT_READY"),
        (
            observed["smallerProfileOptimizationPlateau"] is True,
            "SMALLER_PROFILE_NOT_AT_OPTIMIZATION_CEILING",
        ),
        (observed["targetHardwareMeasured"] is True, "TARGET_HARDWARE_UNMEASURED"),
        (
            observed["candidateGlobalMetricsPassing"] == 13,
            "GLOBAL_METRICS_NOT_ALL_PASSING",
        ),
        (
            observed["candidateCriticalCaseFailures"] == 0,
            "CRITICAL_CASE_FAILURE",
        ),
        (observed["generationRuntimeApproved"] is True, "GENERATION_RUNTIME_UNAPPROVED"),
    ]
    failures.extend(code for passed, code in checks if not passed)
    return {
        "eligible": not failures,
        "failedGates": failures,
        "observed": dict(observed),
    }


def _percent_change(initial: float, final: float) -> float:
    return (initial - final) * 100.0 / initial


def _profile(policy: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return next(item for item in policy["profiles"] if item["name"] == name)


def _derive_decision(policy: Mapping[str, Any]) -> dict[str, Any]:
    evidence = policy["resolvedEvidence"]
    sweep = evidence["architectureSweep"]
    bootstrap = evidence["bootstrapScorecard"]
    best_config = evidence["bestBootstrapConfig"]
    approved_tokens = int(sweep["trainingDataReadiness"]["approvedTrainingTokens"])
    profile_readiness = {}
    for name in ("edge", "core", "server"):
        profile = _profile(policy, name)
        target = int(profile["minimumApprovedTrainingTokens"])
        profile_readiness[name] = {
            "approvedTrainingTokens": approved_tokens,
            "minimumApprovedTrainingTokens": target,
            "coveragePercent": approved_tokens * 100.0 / target,
            "tokensMissing": target - approved_tokens,
            "dataReady": approved_tokens >= target,
        }

    scorecard = bootstrap["checkpointScorecard"]
    first = scorecard[0]
    penultimate = scorecard[-2]
    selected = next(
        item for item in scorecard if item["step"] == bootstrap["selection"]["selectedStep"]
    )
    full_losses = [item["fullValidation"]["nextTokenLoss"] for item in scorecard]
    output_losses = [item["outputHeldout"]["nextTokenLoss"] for item in scorecard]
    monotonic_full = all(left > right for left, right in zip(full_losses, full_losses[1:]))
    monotonic_output = all(left > right for left, right in zip(output_losses, output_losses[1:]))
    task_improvements = {
        task: _percent_change(
            bootstrap["initialHeldout"]["outputHeldout"]["taskMetrics"][task][
                "nextTokenLoss"
            ],
            selected["outputHeldout"]["taskMetrics"][task]["nextTokenLoss"],
        )
        for task in selected["outputHeldout"]["taskMetrics"]
    }

    architecture_results = {item["candidateId"]: item for item in sweep["scorecard"]}
    edge_result = architecture_results["edge-wide-gqa"]
    core_result = architecture_results["core-balanced"]
    core_cost = {
        "parameterRatioVsSelectedEdge": core_result["parameterCount"]
        / edge_result["parameterCount"],
        "trainingWallTimeRatioVsSelectedEdge": core_result["training"]["wallSeconds"]
        / edge_result["training"]["wallSeconds"],
        "peakMemoryRatioVsSelectedEdge": core_result["training"]["peakMemory"][
            "peakWorkingSetMiB"
        ]
        / edge_result["training"]["peakMemory"]["peakWorkingSetMiB"],
        "checkpointBytesRatioVsSelectedEdge": core_result["artifact"]["checkpointBytes"]
        / edge_result["artifact"]["checkpointBytes"],
        "decodeThroughputRatioVsSelectedEdge": core_result["inference"][
            "cachedDecodeTokensPerSecond"
        ]["median"]
        / edge_result["inference"]["cachedDecodeTokensPerSecond"]["median"],
    }
    neural_gates = bootstrap["globalHeldoutScorecard"]["gates"]
    neural_metrics_passing = sum(
        item["bestNeuralCheckpoint"]["thresholdMet"] for item in neural_gates
    )
    runtime_approved = False
    activation_path_approved = False
    checkpoint_release_approved = bool(best_config["releaseApproved"])
    common_release_failures = []
    if neural_metrics_passing != policy["releaseGates"]["requiredGlobalMetricsPassing"]:
        common_release_failures.append("GLOBAL_METRICS_NOT_ALL_PASSING")
    if not runtime_approved:
        common_release_failures.append("GENERATION_RUNTIME_UNAPPROVED")
    if not activation_path_approved:
        common_release_failures.append("ARTIFACT_ACTIVATION_PATH_UNAPPROVED")
    if not checkpoint_release_approved:
        common_release_failures.append("CHECKPOINT_NOT_RELEASE_APPROVED")

    core_larger_observed = {
        "dataCoveragePercent": profile_readiness["core"]["coveragePercent"],
        "independentSeeds": 1,
        "criticalDomainCompositeImprovementPoints": 0.0,
        "confidenceLowerBoundImprovementPoints": 0.0,
        "bootstrapProxyLossImprovementPercent": sweep["selection"][
            "coreLossImprovementPercent"
        ],
        "criticalMetricRegressions": 0,
        "smallerProfileAtDataTarget": profile_readiness["edge"]["dataReady"],
        "smallerProfileOptimizationPlateau": False,
        "targetHardwareMeasured": False,
        "candidateGlobalMetricsPassing": 0,
        "candidateCriticalCaseFailures": 0,
        "generationRuntimeApproved": False,
    }
    core_eligibility = evaluate_larger_profile_eligibility(
        core_larger_observed, policy["largerProfileGates"]
    )
    server_observed = {
        "dataCoveragePercent": profile_readiness["server"]["coveragePercent"],
        "independentSeeds": 0,
        "criticalDomainCompositeImprovementPoints": 0.0,
        "confidenceLowerBoundImprovementPoints": 0.0,
        "bootstrapProxyLossImprovementPercent": 0.0,
        "criticalMetricRegressions": 0,
        "smallerProfileAtDataTarget": profile_readiness["core"]["dataReady"],
        "smallerProfileOptimizationPlateau": False,
        "targetHardwareMeasured": False,
        "candidateGlobalMetricsPassing": 0,
        "candidateCriticalCaseFailures": 0,
        "generationRuntimeApproved": False,
    }
    server_eligibility = evaluate_larger_profile_eligibility(
        server_observed, policy["largerProfileGates"]
    )

    return {
        "decision": {
            "selectedOption": "no-neural-release",
            "modelDisposition": "retain-edge-bootstrap-as-experimental-inactive",
            "scaleDecision": "do-not-scale-parameters",
            "revisionPriority": "approved-data-and-generation-evaluation-before-more-capacity",
            "activeRuntimeDecision": "keep-no-approved-neural-artifact",
            "parameterIncreaseApproved": False,
            "additionalRepeatedDataTrainingApproved": False,
            "reason": "Every neural release option fails mandatory release gates, and both larger profiles fail data, replication, critical-domain, runtime, plateau, and hardware evidence requirements.",
            "nextGate": policy["fallbackPolicy"]["nextGate"],
        },
        "profileReadiness": profile_readiness,
        "hypothesisAssessment": {
            "dataLimited": {
                "verdict": "supported-primary-bottleneck",
                "confidence": "high",
                "evidence": {
                    "edgeDataCoveragePercent": profile_readiness["edge"]["coveragePercent"],
                    "trainingExposureMultiple": bootstrap["run"]["exposureMultiple"],
                    "fullHeldoutLossMonotonicallyImproved": monotonic_full,
                    "outputHeldoutLossMonotonicallyImproved": monotonic_output,
                    "lastIntervalFullLossImprovementPercent": _percent_change(
                        penultimate["fullValidation"]["nextTokenLoss"],
                        selected["fullValidation"]["nextTokenLoss"],
                    ),
                    "lastIntervalOutputLossImprovementPercent": _percent_change(
                        penultimate["outputHeldout"]["nextTokenLoss"],
                        selected["outputHeldout"]["nextTokenLoss"],
                    ),
                    "minimumTaskLossImprovementPercent": min(task_improvements.values()),
                    "maximumTaskLossImprovementPercent": max(task_improvements.values()),
                },
                "action": "Expand diverse, governed, VoltForge-owned records and independent held-out coverage before repeating training or increasing parameters.",
            },
            "optimizationLimited": {
                "verdict": "not-established-secondary-hypothesis",
                "confidence": "insufficient-evidence",
                "evidence": {
                    "independentBootstrapSeeds": 1,
                    "selectedCheckpointWasLastStep": selected["step"] == scorecard[-1]["step"],
                    "lossPlateauObserved": False,
                    "approvedUniqueDataExhaustedBeforeScheduleComparison": True,
                },
                "action": "Do not tune on repeated records. Re-test schedules with at least three seeds only after materially more unique approved data exists.",
            },
            "architectureLimited": {
                "verdict": "not-supported",
                "confidence": "moderate",
                "evidence": {
                    "selectedArchitecture": sweep["selection"]["selectedCandidateId"],
                    "coreProxyLossImprovementPercent": sweep["selection"][
                        "coreLossImprovementPercent"
                    ],
                    "materialProxyThresholdPercent": sweep["selectionPolicy"][
                        "coreMinimumMaterialLossImprovementPercent"
                    ],
                    "coreMaterialThresholdMet": sweep["selection"][
                        "coreMaterialThresholdMet"
                    ],
                    "coreCost": core_cost,
                },
                "action": "Keep the measured edge architecture as the experimental baseline; do not infer an architecture ceiling from an undertrained one-seed sweep.",
            },
            "capacityLimited": {
                "verdict": "not-supported-and-scaling-would-worsen-data-readiness",
                "confidence": "high-for-current-data",
                "evidence": {
                    "edgeTokensPerParameter": approved_tokens / edge_result["parameterCount"],
                    "coreTokensPerParameter": approved_tokens / core_result["parameterCount"],
                    "coreEligibility": core_eligibility,
                    "serverCandidateMeasured": False,
                    "serverEligibility": server_eligibility,
                },
                "action": "Do not construct or train a larger profile until edge reaches its data target, plateaus under replicated optimization, and larger-profile critical gains clear confidence and hardware gates.",
            },
        },
        "optionDecisionLog": [
            {
                "option": "edge-neural-release",
                "selected": False,
                "disposition": "retain-experimental-inactive",
                "failedGates": ["INSUFFICIENT_APPROVED_DATA", *common_release_failures],
                "evidence": {
                    "parameterCount": best_config["parameterCount"],
                    "dataCoveragePercent": profile_readiness["edge"]["coveragePercent"],
                    "fullTop1TokenAccuracy": selected["fullValidation"][
                        "top1TokenAccuracy"
                    ],
                    "outputTop1TokenAccuracy": selected["outputHeldout"][
                        "top1TokenAccuracy"
                    ],
                    "globalMetricsPassing": neural_metrics_passing,
                    "globalMetricsRequired": len(neural_gates),
                },
            },
            {
                "option": "core-neural-release",
                "selected": False,
                "disposition": "reject-scale",
                "failedGates": core_eligibility["failedGates"],
                "evidence": {
                    "parameterCount": core_result["parameterCount"],
                    "dataCoveragePercent": profile_readiness["core"]["coveragePercent"],
                    "proxyLossImprovementPercent": sweep["selection"][
                        "coreLossImprovementPercent"
                    ],
                    "criticalDomainGenerationEvidenceAvailable": False,
                    "independentSeeds": 1,
                    "cost": core_cost,
                },
            },
            {
                "option": "server-neural-release",
                "selected": False,
                "disposition": "reject-unmeasured-scale",
                "failedGates": server_eligibility["failedGates"],
                "evidence": {
                    "candidateConstructed": False,
                    "dataCoveragePercent": profile_readiness["server"]["coveragePercent"],
                    "coreCeilingDemonstrated": False,
                    "targetHardwareMeasured": False,
                },
            },
            {
                "option": "no-neural-release",
                "selected": True,
                "disposition": "selected-safe-state",
                "failedGates": [],
                "evidence": {
                    "checkpointRetainedForResearch": True,
                    "checkpointReleaseApproved": False,
                    "activeNeuralArtifactRequired": False,
                    "deterministicToolsRemainAuthoritative": True,
                },
            },
        ],
        "criticalGainBeyondNoise": {
            "currentStatus": "not-demonstrated",
            "reason": "The core comparison has one seed and teacher-forced proxy loss only; it has no checkpoint generation results, confidence interval, or critical-domain composite score.",
            "futureMinimumEvidence": {
                "independentSeeds": policy["largerProfileGates"][
                    "minimumIndependentSeeds"
                ],
                "confidenceLevel": policy["largerProfileGates"]["confidenceLevel"],
                "criticalDomainCompositeImprovementPoints": policy[
                    "largerProfileGates"
                ]["minimumCriticalDomainCompositeImprovementPoints"],
                "confidenceLowerBoundMustExceedPoints": 0.0,
                "criticalMetricRegressionsAllowed": 0,
                "allGlobalMetricsMustPass": True,
            },
        },
        "revisionSequence": [
            {
                "order": 1,
                "action": "Preserve the step-256 edge checkpoint and scorecards as an immutable experimental baseline; do not activate it.",
                "exitEvidence": "VFAI-016 packages an experimental inactive artifact with checksums while activeArtifactId remains null.",
            },
            {
                "order": 2,
                "action": "Expand governed project-owned data and independent held-out coverage, prioritizing current critical failures, task diversity, exact variants, and compiler/simulation/retrieval evidence.",
                "exitEvidence": "Edge has at least 100M approved quality-controlled tokens with leakage-free, independently governed evaluation coverage.",
            },
            {
                "order": 3,
                "action": "Build the local experimental inference, context, and output-gate path without making the checkpoint active or borrowing deterministic credit.",
                "exitEvidence": "The raw checkpoint can execute all 26 frozen cases offline and receives its own 13-metric scorecard.",
            },
            {
                "order": 4,
                "action": "Retrain the edge profile with at least three deterministic seeds and controlled schedule/data ablations; diagnose a real plateau before capacity changes.",
                "exitEvidence": "Replicated edge learning curves and confidence intervals separate data and optimization effects.",
            },
            {
                "order": 5,
                "action": "Reconsider core only if edge is data-ready and plateaued and core improves critical domain composite score beyond noise without any critical regression on measured target hardware.",
                "exitEvidence": "Every larger-profile gate passes; otherwise no parameter increase is authorized.",
            },
        ],
        "safeOperatingBoundary": {
            "activeArtifactIdMustRemainNullAtDecision": True,
            "deterministicToolsRemainAuthoritative": policy["fallbackPolicy"][
                "deterministicToolsRemainAuthoritative"
            ],
            "approvedLocalRetrievalMayContinue": policy["fallbackPolicy"][
                "approvedLocalRetrievalMayContinue"
            ],
            "internetRetrievalRole": "optional-untrusted-evidence-with-citations-never-a-model",
            "experimentalCheckpointMayServeUsers": False,
            "experimentalCheckpointMayAutoApplyChanges": False,
            "VFAI016PackagingPermission": "experimental-inactive-only",
            "VFAI016ActivationPermission": False,
        },
        "evidenceSummary": {
            "approvedTrainingTokens": approved_tokens,
            "architectureSweepReportSha256": sweep["reportSha256"],
            "bootstrapReportSha256": bootstrap["reportSha256"],
            "bestBootstrapManifestSha256": best_config["manifestSha256"],
            "selectedBootstrapStep": selected["step"],
            "bootstrapFirstEvaluatedStep": first["step"],
            "bootstrapGlobalMetricsPassing": neural_metrics_passing,
            "bootstrapGlobalMetricCount": len(neural_gates),
        },
    }


def make_scale_decision(
    *,
    policy_path: str | Path = DEFAULT_POLICY_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    policy = load_scale_policy(policy_path)
    registry_contract = policy["evidence"]["activeRegistry"]
    registry_path = _resolve_repo_path(registry_contract["path"], "active registry")
    registry = _read_json(registry_path, "active registry")
    if (
        registry.get("state") != registry_contract["requiredState"]
        or registry.get("activeArtifactId") is not None
        or registry.get("releaseStatus") != "none"
    ):
        raise ScaleDecisionContractError(
            "active registry is not in the required no-approved-artifact state"
        )
    derived = _derive_decision(policy)
    report = {
        "schemaVersion": 1,
        "reportId": "vfai015-gen1-scale-decision-v1",
        "generatedAtUtc": _utc_now(),
        "policyId": policy["policyId"],
        "policyPath": _portable_path(Path(policy_path)),
        "policyFileSha256": _sha256_file(Path(policy_path)),
        "policyFingerprint": policy["policyFingerprint"],
        "registryObservation": {
            "path": _portable_path(registry_path),
            "fileSha256AtDecision": _sha256_file(registry_path),
            "state": registry["state"],
            "activeArtifactId": registry["activeArtifactId"],
            "releaseStatus": registry["releaseStatus"],
            "futureVFAI016MutationExpected": True,
        },
        **derived,
    }
    report["reportSha256"] = _fingerprint(report)
    _atomic_json(Path(report_path), report)
    return report


def check_scale_decision(
    *,
    policy_path: str | Path = DEFAULT_POLICY_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    policy = load_scale_policy(policy_path)
    report = _read_json(Path(report_path), "Gen1 scale decision")
    if report.get("reportSha256") != _fingerprint(
        {key: value for key, value in report.items() if key != "reportSha256"}
    ):
        raise ScaleDecisionContractError("scale-decision report checksum mismatch")
    if (
        report.get("policyFingerprint") != policy["policyFingerprint"]
        or report.get("policyFileSha256") != _sha256_file(Path(policy_path))
    ):
        raise ScaleDecisionContractError("scale-decision policy binding changed")
    expected = _derive_decision(policy)
    for key, value in expected.items():
        if report.get(key) != value:
            raise ScaleDecisionContractError(f"scale-decision derivation changed: {key}")
    registry = report.get("registryObservation")
    if (
        not isinstance(registry, Mapping)
        or registry.get("state") != policy["evidence"]["activeRegistry"]["requiredState"]
        or registry.get("activeArtifactId") is not None
        or registry.get("releaseStatus") != "none"
        or not isinstance(registry.get("fileSha256AtDecision"), str)
        or len(registry["fileSha256AtDecision"]) != 64
    ):
        raise ScaleDecisionContractError("recorded active-registry boundary changed")
    if (
        report["decision"]["selectedOption"] != "no-neural-release"
        or report["decision"]["parameterIncreaseApproved"] is not False
        or report["safeOperatingBoundary"]["experimentalCheckpointMayServeUsers"]
        is not False
    ):
        raise ScaleDecisionContractError("safe no-release decision changed")
    return {
        "decision": "pass",
        "reportId": report["reportId"],
        "selectedOption": report["decision"]["selectedOption"],
        "primaryBottleneck": "data-limited",
        "parameterIncreaseApproved": False,
        "activeArtifactAtDecision": None,
        "nextGate": report["decision"]["nextGate"],
    }
