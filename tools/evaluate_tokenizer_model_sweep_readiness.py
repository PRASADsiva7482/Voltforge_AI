"""Generate or verify the VFAI-FU-005 joint-sweep readiness receipt.

The evaluator validates immutable tokenizer and corpus lineage, but it does not
train a tokenizer or model. Its report is content-free and cannot complete the
follow-up while the governed corpus and alternative-tokenizer gates remain open.
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

MASTER_BACKLOG_PATH = AI_ROOT / "AI_MASTER_BACKLOG.json"
FOLLOWUP_BACKLOG_PATH = AI_ROOT / "AI_FOLLOWUP_BACKLOG.json"
POLICY_PATH = AI_ROOT / "gen1_sweep/joint-tokenizer-model-policy.v1.json"
DEFAULT_REPORT_PATH = (
    AI_ROOT / "evaluation/reports/tokenizer-model-sweep-readiness-v1.json"
)
COMPLETE_STATUSES = frozenset({"complete", "completed", "done"})


class SweepReadinessError(RuntimeError):
    """The VFAI-FU-005 readiness evidence is invalid or stale."""


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
        raise SweepReadinessError(f"unable to read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise SweepReadinessError(f"{label} must be a JSON object")
    return value


def _write_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _backlog_item(backlog: Mapping[str, Any], item_id: str) -> Mapping[str, Any]:
    items = backlog.get("items")
    if not isinstance(items, list):
        raise SweepReadinessError("follow-up backlog items are invalid")
    matches = [item for item in items if isinstance(item, Mapping) and item.get("id") == item_id]
    if len(matches) != 1:
        raise SweepReadinessError(f"expected exactly one {item_id} backlog item")
    return matches[0]


def _resolve_workspace_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SweepReadinessError(f"{label} path is missing")
    path = (AI_ROOT / value).resolve()
    try:
        path.relative_to(AI_ROOT.resolve())
    except ValueError as exc:
        raise SweepReadinessError(f"{label} path escapes the AI workspace") from exc
    return path


def _candidate_evidence(
    candidate: Mapping[str, Any], policy: Mapping[str, Any]
) -> dict[str, Any]:
    candidate_id = candidate.get("candidateId")
    vocab_size = candidate.get("vocabSize")
    role = candidate.get("role")
    manifest_value = candidate.get("manifestPath")
    base = {
        "candidateId": candidate_id,
        "vocabSize": vocab_size,
        "role": role,
        "declaredState": candidate.get("state"),
        "manifestPath": manifest_value,
    }
    if manifest_value is None:
        return {
            **base,
            "approved": False,
            "reason": "MANIFEST_NOT_ASSIGNED",
            "artifactSha256": None,
            "manifestFileSha256": None,
        }
    manifest_path = _resolve_workspace_path(manifest_value, str(candidate_id))
    if not manifest_path.is_file():
        return {
            **base,
            "approved": False,
            "reason": "MANIFEST_NOT_FOUND",
            "artifactSha256": None,
            "manifestFileSha256": None,
        }
    manifest = _read_json_object(manifest_path, f"{candidate_id} tokenizer manifest")
    artifact_digest = manifest.get("artifactSha256")
    unsigned_manifest = dict(manifest)
    unsigned_manifest.pop("artifactSha256", None)
    artifact_checksum_valid = artifact_digest == hashlib.sha256(
        _canonical_json(unsigned_manifest)
    ).hexdigest()
    files = manifest.get("files")
    config_descriptor = files.get("config") if isinstance(files, Mapping) else None
    config_path_value = (
        config_descriptor.get("path") if isinstance(config_descriptor, Mapping) else None
    )
    config_path = (
        manifest_path.parent / config_path_value
        if isinstance(config_path_value, str) and config_path_value
        else None
    )
    config_checksum_valid = bool(
        config_path is not None
        and config_path.is_file()
        and config_descriptor.get("sha256") == _sha256_file(config_path)
    )
    config = (
        _read_json_object(config_path, f"{candidate_id} tokenizer config")
        if config_checksum_valid and config_path is not None
        else {}
    )
    fixed_contract = policy.get("fixedTokenizerContract")
    fixed_special_ids = (
        fixed_contract.get("specialTokenIds")
        if isinstance(fixed_contract, Mapping)
        else None
    )
    training = config.get("training")
    from_scratch = bool(
        isinstance(training, Mapping)
        and training.get("fromScratch") is True
        and training.get("pretrainedVocabularyLoaded") is False
        and training.get("pretrainedMergesLoaded") is False
    )
    gates = {
        "releaseApproved": manifest.get("releaseStatus") == "approved",
        "declaredVocabularyMatches": manifest.get("vocabSize") == vocab_size,
        "algorithmMatches": manifest.get("algorithm")
        == fixed_contract.get("algorithm")
        if isinstance(fixed_contract, Mapping)
        else False,
        "artifactChecksumValid": artifact_checksum_valid,
        "configChecksumValid": config_checksum_valid,
        "fixedSpecialTokenIdsMatch": config.get("specialTokenIds") == fixed_special_ids,
        "trainedFromScratch": from_scratch,
    }
    approved = all(gates.values())
    return {
        **base,
        "approved": approved,
        "reason": None if approved else "TOKENIZER_ADMISSION_GATE_FAILED",
        "artifactSha256": artifact_digest,
        "manifestFileSha256": _sha256_file(manifest_path),
        "gates": gates,
    }


def _policy_controls_complete(policy: Mapping[str, Any]) -> bool:
    admission = policy.get("candidateAdmission")
    exposure = policy.get("fairExposure")
    model_sweep = policy.get("modelSweep")
    selection = policy.get("selection")
    forbidden = policy.get("forbiddenInputsAndServices")
    return bool(
        isinstance(admission, Mapping)
        and admission.get("sameApprovedNormalizedRecordIds") is True
        and admission.get("sameTrainingAndValidationSplitHashes") is True
        and admission.get("sameRawUtf8TrainingAndValidationBytes") is True
        and isinstance(exposure, Mapping)
        and exposure.get("tokenBudgetEquivalenceAllowed") is False
        and exposure.get("sameRecordOrderAndSeedSchedule") is True
        and exposure.get("sameRawContentExposurePerSeed") is True
        and len(exposure.get("requiredAccounting", [])) >= 10
        and isinstance(model_sweep, Mapping)
        and int(model_sweep.get("minimumSeeds", 0)) >= 3
        and model_sweep.get("releaseActivationAllowed") is False
        and isinstance(selection, Mapping)
        and selection.get("method") == "measured-pareto-frontier"
        and selection.get("requiresMaterialImprovementOverBaseline") is True
        and selection.get("automaticPromotionAllowed") is False
        and isinstance(forbidden, list)
        and len(forbidden) >= 7
    )


def build_report(
    *,
    master_backlog: Mapping[str, Any],
    followup_backlog: Mapping[str, Any],
    policy: Mapping[str, Any],
    corpus_manifest: Mapping[str, Any],
    corpus_fingerprint: str,
    candidates: Sequence[Mapping[str, Any]],
    generated_on: str,
    source_sha256: Mapping[str, str],
) -> dict[str, Any]:
    if policy.get("sourceFollowup") != "VFAI-FU-005":
        raise SweepReadinessError("joint-sweep policy is not bound to VFAI-FU-005")
    fu005 = _backlog_item(followup_backlog, "VFAI-FU-005")
    fu006 = _backlog_item(followup_backlog, "VFAI-FU-006")
    master_items = master_backlog.get("items")
    master_complete = bool(
        master_backlog.get("status") in COMPLETE_STATUSES
        and isinstance(master_items, list)
        and master_items
        and all(
            isinstance(item, Mapping) and item.get("status") in COMPLETE_STATUSES
            for item in master_items
        )
    )
    data_dependency = policy.get("dataDependency")
    if not isinstance(data_dependency, Mapping):
        raise SweepReadinessError("joint-sweep data dependency is invalid")
    edge_minimum = int(data_dependency.get("minimumUniqueTrainingPredictedTokens", 0))
    training = corpus_manifest.get("training")
    if not isinstance(training, Mapping):
        raise SweepReadinessError("packed corpus training descriptor is invalid")
    unique_predictions = int(training.get("predictedTokenCount", 0))
    corpus_scale_ready = edge_minimum > 0 and unique_predictions >= edge_minimum
    data_governance_ready = fu006.get("status") in COMPLETE_STATUSES
    candidate_minimum = int(
        policy.get("candidateAdmission", {}).get("requiredApprovedCandidateCount", 0)
    )
    approved_candidate_count = sum(item.get("approved") is True for item in candidates)
    candidate_gate_ready = (
        candidate_minimum > 1 and approved_candidate_count >= candidate_minimum
    )
    controls_complete = _policy_controls_complete(policy)
    execution_allowed = all(
        (
            master_complete,
            corpus_scale_ready,
            data_governance_ready,
            candidate_gate_ready,
            controls_complete,
        )
    )
    if not master_complete:
        decision = "await-master-backlog-completion"
    elif not corpus_scale_ready or not data_governance_ready:
        decision = "await-governed-corpus-expansion"
    elif not candidate_gate_ready:
        decision = "ready-to-train-tokenizer-candidates"
    elif not controls_complete:
        decision = "repair-joint-sweep-controls"
    else:
        decision = "ready-for-joint-controlled-sweep"
    coverage_percent = round(unique_predictions * 100 / max(1, edge_minimum), 6)
    report: dict[str, Any] = {
        "schemaVersion": 1,
        "reportId": "vfai-fu-005-tokenizer-model-sweep-readiness-v1",
        "generatedOn": generated_on,
        "followup": {
            "id": fu005.get("id"),
            "recordedStatus": fu005.get("status"),
        },
        "policy": {
            "policyId": policy.get("policyId"),
            "status": policy.get("status"),
            "requiredApprovedCandidateCount": candidate_minimum,
            "minimumUniqueTrainingPredictedTokens": edge_minimum,
        },
        "corpus": {
            "datasetId": corpus_manifest.get("datasetId"),
            "approvalStatus": corpus_manifest.get("approvalStatus"),
            "trainingRecordCount": corpus_manifest.get("trainingRecordCount"),
            "validationRecordCount": corpus_manifest.get("validationRecordCount"),
            "uniqueTrainingPredictedTokens": unique_predictions,
            "edgeMinimumUniqueTrainingPredictedTokens": edge_minimum,
            "edgeCoveragePercent": coverage_percent,
            "edgeTokenGap": max(0, edge_minimum - unique_predictions),
            "trainingCorpusSha256": corpus_manifest.get("trainingCorpusSha256"),
            "validationCorpusSha256": corpus_manifest.get("validationCorpusSha256"),
            "packedCorpusFingerprint": corpus_fingerprint,
            "tokenizer": corpus_manifest.get("tokenizer"),
        },
        "candidateTokenizers": [dict(item) for item in candidates],
        "gates": {
            "masterBacklogComplete": master_complete,
            "governedCorpusAtEdgeMinimum": corpus_scale_ready,
            "corpusGovernanceAndDiversityAccepted": data_governance_ready,
            "approvedTokenizerCandidates": approved_candidate_count,
            "approvedTokenizerCandidateGate": candidate_gate_ready,
            "fairExposureAndSelectionControlsDefined": controls_complete,
            "jointSweepExecutionAllowed": execution_allowed,
        },
        "decision": decision,
        "completionClaimed": False,
        "tokenizerCandidatesTrainedByEvaluator": False,
        "modelSweepExecuted": False,
        "governedCorpusExported": False,
        "networkAccessed": False,
        "unmeasured": [
            "alternative tokenizer quality and compression",
            "joint tokenizer and decoder quality-cost frontier",
            "three-seed training reproducibility",
            "training throughput and peak memory",
            "artifact size and runtime latency",
        ],
        "sourceSha256": dict(source_sha256),
        "reportSha256": "",
    }
    report["reportSha256"] = _receipt_digest(report)
    return report


def _current_inputs() -> dict[str, Any]:
    master = _read_json_object(MASTER_BACKLOG_PATH, "master backlog")
    followup = _read_json_object(FOLLOWUP_BACKLOG_PATH, "follow-up backlog")
    policy = _read_json_object(POLICY_PATH, "joint-sweep policy")
    declared_candidates = policy.get("vocabularyCandidates")
    if not isinstance(declared_candidates, list) or not declared_candidates:
        raise SweepReadinessError("joint-sweep vocabulary candidates are invalid")
    candidates = [
        _candidate_evidence(candidate, policy)
        for candidate in declared_candidates
        if isinstance(candidate, Mapping)
    ]
    if len(candidates) != len(declared_candidates):
        raise SweepReadinessError("joint-sweep vocabulary candidate entry is invalid")
    baselines = [item for item in candidates if item.get("role") == "approved-baseline"]
    if len(baselines) != 1 or not isinstance(baselines[0].get("manifestPath"), str):
        raise SweepReadinessError("joint-sweep policy requires one approved baseline")
    baseline_manifest_path = _resolve_workspace_path(
        baselines[0]["manifestPath"], "approved baseline"
    )
    baseline_manifest = _read_json_object(
        baseline_manifest_path, "approved baseline tokenizer manifest"
    )

    # Imported lazily so receipt inspection and unit tests stay content-free and
    # do not initialize the native training runtime.
    from gen1_training import load_approved_corpus
    from model.gen1 import Gen1Config

    config = Gen1Config(
        vocab_size=int(baseline_manifest["vocabSize"]), max_sequence_length=128
    )
    corpus = load_approved_corpus(
        config,
        tokenizer_directory=baseline_manifest_path.parent,
        packing_block_size=128,
    )
    return {
        "master_backlog": master,
        "followup_backlog": followup,
        "policy": policy,
        "corpus_manifest": corpus.manifest,
        "corpus_fingerprint": corpus.fingerprint,
        "candidates": candidates,
        "source_sha256": {
            "AI_MASTER_BACKLOG.json": _sha256_file(MASTER_BACKLOG_PATH),
            "AI_FOLLOWUP_BACKLOG.json": _sha256_file(FOLLOWUP_BACKLOG_PATH),
            "gen1_sweep/joint-tokenizer-model-policy.v1.json": _sha256_file(
                POLICY_PATH
            ),
            "tools/evaluate_tokenizer_model_sweep_readiness.py": _sha256_file(
                Path(__file__)
            ),
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
        raise SweepReadinessError("VFAI-FU-005 readiness receipt checksum is invalid")
    gates = report.get("gates")
    if not isinstance(gates, Mapping):
        raise SweepReadinessError("VFAI-FU-005 readiness gates are invalid")
    execution_allowed = all(
        (
            gates.get("masterBacklogComplete") is True,
            gates.get("governedCorpusAtEdgeMinimum") is True,
            gates.get("corpusGovernanceAndDiversityAccepted") is True,
            gates.get("approvedTokenizerCandidateGate") is True,
            gates.get("fairExposureAndSelectionControlsDefined") is True,
        )
    )
    if gates.get("jointSweepExecutionAllowed") is not execution_allowed:
        raise SweepReadinessError("VFAI-FU-005 execution gate contradicts its inputs")
    if gates.get("masterBacklogComplete") is not True:
        expected_decision = "await-master-backlog-completion"
    elif (
        gates.get("governedCorpusAtEdgeMinimum") is not True
        or gates.get("corpusGovernanceAndDiversityAccepted") is not True
    ):
        expected_decision = "await-governed-corpus-expansion"
    elif gates.get("approvedTokenizerCandidateGate") is not True:
        expected_decision = "ready-to-train-tokenizer-candidates"
    elif gates.get("fairExposureAndSelectionControlsDefined") is not True:
        expected_decision = "repair-joint-sweep-controls"
    else:
        expected_decision = "ready-for-joint-controlled-sweep"
    if report.get("decision") != expected_decision:
        raise SweepReadinessError("VFAI-FU-005 receipt decision contradicts its gates")
    if any(
        (
            report.get("completionClaimed") is not False,
            report.get("tokenizerCandidatesTrainedByEvaluator") is not False,
            report.get("modelSweepExecuted") is not False,
            report.get("governedCorpusExported") is not False,
            report.get("networkAccessed") is not False,
        )
    ):
        raise SweepReadinessError("VFAI-FU-005 readiness receipt overclaims execution")


def verify(path: Path) -> dict[str, Any]:
    report = _read_json_object(path, "VFAI-FU-005 readiness receipt")
    validate_report(report)
    expected = _build_current_report(str(report.get("generatedOn")))
    if report != expected:
        raise SweepReadinessError("VFAI-FU-005 readiness receipt is stale")
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
                "uniqueTrainingPredictedTokens": report["corpus"][
                    "uniqueTrainingPredictedTokens"
                ],
                "edgeCoveragePercent": report["corpus"]["edgeCoveragePercent"],
                "approvedTokenizerCandidates": report["gates"][
                    "approvedTokenizerCandidates"
                ],
                "reportSha256": report["reportSha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
