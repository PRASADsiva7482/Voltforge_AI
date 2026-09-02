"""Generate or verify the VFAI-FU-011 component-coverage readiness receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


AI_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = AI_ROOT.parent
UI_ROOT = WORKSPACE_ROOT / "Voltforge_UI"
BACKEND_ROOT = WORKSPACE_ROOT / "Voltforge_BL"
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from hardware_coverage import get_component_coverage  # noqa: E402
from tools.verify_hardware_coverage import verify as verify_ui_coverage  # noqa: E402


MASTER_BACKLOG_PATH = AI_ROOT / "AI_MASTER_BACKLOG.json"
FOLLOWUP_BACKLOG_PATH = AI_ROOT / "AI_FOLLOWUP_BACKLOG.json"
POLICY_PATH = AI_ROOT / "hardware_coverage/component-policy.v1.json"
CORPUS_CATALOG_PATH = AI_ROOT / "electronics_corpus/catalog.v1.json"
COMPONENT_SERVICE_PATH = AI_ROOT / "hardware_coverage/component_service.py"
VERIFIER_PATH = AI_ROOT / "tools/verify_hardware_coverage.py"
AI_ROUTES_PATH = AI_ROOT / "api/routes.py"
UI_CATALOG_PATH = UI_ROOT / "src/features/canvas/componentCatalog.ts"
UI_TYPES_PATH = UI_ROOT / "src/types/domain.ts"
UI_SERVICES_PATH = UI_ROOT / "src/api/services.ts"
UI_PANEL_PATH = UI_ROOT / "src/features/editor/ComponentPanel.tsx"
UI_PRESENTATION_PATH = UI_ROOT / "src/features/ai/aiHardwareCoverage.ts"
BACKEND_CONTROLLER_PATH = (
    BACKEND_ROOT / "src/main/java/in/voltforge/api/ai/controller/AiController.java"
)
BACKEND_SERVICE_PATH = (
    BACKEND_ROOT / "src/main/java/in/voltforge/api/ai/service/AiService.java"
)
BACKEND_IMPL_PATH = (
    BACKEND_ROOT / "src/main/java/in/voltforge/api/ai/service/impl/AiServiceImpl.java"
)
DEFAULT_REPORT_PATH = AI_ROOT / "evaluation/reports/component-coverage-readiness-v1.json"
COMPLETE_STATUSES = frozenset({"complete", "completed", "done"})


class ComponentCoverageReadinessError(RuntimeError):
    """VFAI-FU-011 evidence is invalid or stale."""


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report_digest(report: Mapping[str, Any]) -> str:
    unsigned = dict(report)
    unsigned.pop("reportSha256", None)
    return hashlib.sha256(_canonical(unsigned)).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ComponentCoverageReadinessError(f"unable to read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ComponentCoverageReadinessError(f"{label} must be a JSON object")
    return value


def _write_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _backlog_item(backlog: Mapping[str, Any]) -> Mapping[str, Any]:
    items = backlog.get("items")
    matches = [
        item
        for item in items
        if isinstance(item, Mapping) and item.get("id") == "VFAI-FU-011"
    ] if isinstance(items, list) else []
    if len(matches) != 1:
        raise ComponentCoverageReadinessError("expected exactly one VFAI-FU-011 item")
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


def _workspace_file(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = (WORKSPACE_ROOT / value).resolve()
    try:
        path.relative_to(WORKSPACE_ROOT.resolve())
    except ValueError as exc:
        raise ComponentCoverageReadinessError("component evidence path escapes workspace") from exc
    return path if path.is_file() else None


def _integration_evidence() -> dict[str, Any]:
    sources = {
        "aiRoute": AI_ROUTES_PATH.read_text(encoding="utf-8"),
        "backendController": BACKEND_CONTROLLER_PATH.read_text(encoding="utf-8"),
        "backendService": BACKEND_SERVICE_PATH.read_text(encoding="utf-8"),
        "backendImplementation": BACKEND_IMPL_PATH.read_text(encoding="utf-8"),
        "uiTypes": UI_TYPES_PATH.read_text(encoding="utf-8"),
        "uiServices": UI_SERVICES_PATH.read_text(encoding="utf-8"),
        "uiPanel": UI_PANEL_PATH.read_text(encoding="utf-8"),
        "uiPresentation": UI_PRESENTATION_PATH.read_text(encoding="utf-8"),
    }
    checks = {
        "aiReadOnlyEndpoint": all(
            marker in sources["aiRoute"]
            for marker in ('@router.get("/component-coverage")', "get_component_coverage")
        ),
        "authenticatedBackendProxy": all(
            marker in (sources["backendController"] + sources["backendImplementation"])
            for marker in ("/component-coverage", "getComponentCoverage")
        ) and "getComponentCoverage" in sources["backendService"],
        "typedUiContract": all(
            marker in sources["uiTypes"]
            for marker in ("AiComponentCoverageResponse", "simulation-only")
        ) and "getComponentCoverage" in sources["uiServices"],
        "componentPanelDisplaysCoverage": all(
            marker in sources["uiPanel"]
            for marker in (
                "componentCoverageQuery",
                "componentCoverageByType",
                "aiComponentCoverageStatusLabel",
            )
        ),
        "simulationAndVariantLabelsDistinct": all(
            marker in sources["uiPresentation"]
            for marker in ("Variant required", "Simulation model", "AI exact")
        ),
    }
    return {"checks": checks, "ready": all(checks.values())}


def _assigned_inputs(policy: Mapping[str, Any], component_types: set[str]) -> dict[str, Any]:
    exact = policy.get("exactPromotionInputs", {})
    approved = exact.get("approvedComponentTypes")
    approved_valid = bool(
        isinstance(approved, list)
        and approved
        and len(approved) == len(set(approved))
        and set(approved) <= component_types
    )
    return {
        "ownerPriorityReceiptPath": exact.get("ownerPriorityReceiptPath"),
        "ownerPriorityReceiptAssigned": _workspace_file(
            exact.get("ownerPriorityReceiptPath")
        ) is not None,
        "exactCandidateSelectionPath": exact.get("exactCandidateSelectionPath"),
        "exactCandidateSelectionAssigned": _workspace_file(
            exact.get("exactCandidateSelectionPath")
        ) is not None,
        "approvedComponentTypes": approved if isinstance(approved, list) else [],
        "approvedComponentTypesValid": approved_valid,
        "immutableCorpusReleaseReady": exact.get("immutableCorpusReleaseReady") is True,
        "requiredFollowupBeforeNewCorpusRelease": exact.get(
            "requiredFollowupBeforeNewCorpusRelease"
        ),
    }


def _expected_decision(gates: Mapping[str, Any]) -> str:
    if gates.get("masterBacklogComplete") is not True:
        return "await-master-backlog-completion"
    if gates.get("uiComponentParityVerified") is not True:
        return "repair-ui-component-parity"
    if gates.get("genericLabelsFailClosed") is not True:
        return "repair-generic-component-boundary"
    if gates.get("endToEndCoverageIntegrationReady") is not True:
        return "complete-component-coverage-integration"
    if gates.get("ownerPriorityReceiptAssigned") is not True:
        return "await-owner-prioritized-exact-components"
    if gates.get("immutableCorpusReleaseReady") is not True:
        return "await-immutable-component-corpus-release"
    if gates.get("exactCandidateSelectionAssigned") is not True:
        return "await-evidenced-exact-candidate-selection"
    if gates.get("approvedComponentTypesValid") is not True:
        return "await-approved-exact-component-types"
    return "ready-for-exact-component-corpus-release"


def build_report(
    *,
    master_backlog: Mapping[str, Any],
    followup_backlog: Mapping[str, Any],
    policy: Mapping[str, Any],
    coverage: Mapping[str, Any],
    parity: Mapping[str, Any],
    integration: Mapping[str, Any],
    inputs: Mapping[str, Any],
    generated_on: str,
    source_sha256: Mapping[str, str],
) -> dict[str, Any]:
    if policy.get("sourceFollowup") != "VFAI-FU-011":
        raise ComponentCoverageReadinessError("policy is not bound to VFAI-FU-011")
    fu011 = _backlog_item(followup_backlog)
    summary = coverage.get("summary", {})
    generic_fail_closed = bool(
        coverage.get("entryCount") == 60
        and summary.get("verified") == 0
        and summary.get("variantRequired") == 51
        and summary.get("simulationOnly") == 9
        and coverage.get("genericLabelsMaySelectCandidate") is False
        and all(
            entry.get("aiElectricalClaimsAllowed") is False
            and entry.get("physicalIdentitySelected") is False
            for entry in coverage.get("entries", [])
        )
    )
    gates = {
        "masterBacklogComplete": _master_complete(master_backlog),
        "uiComponentParityVerified": parity.get("componentEntryCount") == 60,
        "genericLabelsFailClosed": generic_fail_closed,
        "endToEndCoverageIntegrationReady": integration.get("ready") is True,
        "ownerPriorityReceiptAssigned": inputs.get("ownerPriorityReceiptAssigned") is True,
        "immutableCorpusReleaseReady": inputs.get("immutableCorpusReleaseReady") is True,
        "exactCandidateSelectionAssigned": inputs.get("exactCandidateSelectionAssigned") is True,
        "approvedComponentTypesValid": inputs.get("approvedComponentTypesValid") is True,
    }
    report: dict[str, Any] = {
        "schemaVersion": 1,
        "reportId": "vfai-fu-011-component-coverage-readiness-v1",
        "generatedOn": generated_on,
        "followup": {"id": fu011.get("id"), "recordedStatus": fu011.get("status")},
        "policy": {"policyId": policy.get("policyId"), "status": policy.get("status")},
        "coverage": {
            "reportId": coverage.get("reportId"),
            "reportSha256": coverage.get("reportSha256"),
            "entryCount": coverage.get("entryCount"),
            "summary": dict(summary) if isinstance(summary, Mapping) else {},
            "genericLabelsMaySelectCandidate": coverage.get(
                "genericLabelsMaySelectCandidate"
            ),
        },
        "parity": dict(parity),
        "integration": dict(integration),
        "assignedInputs": dict(inputs),
        "gates": gates,
        "decision": _expected_decision(gates),
        "completionClaimed": False,
        "exactComponentsPromotedByEvaluator": 0,
        "corpusFilesModifiedByEvaluator": False,
        "genericVariantSelectedByEvaluator": False,
        "networkAccessed": False,
        "liveStateMutated": False,
        "sourceSha256": dict(source_sha256),
        "reportSha256": "",
    }
    report["reportSha256"] = _report_digest(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("reportSha256") != _report_digest(report):
        raise ComponentCoverageReadinessError("VFAI-FU-011 receipt checksum is invalid")
    gates = report.get("gates")
    if not isinstance(gates, Mapping) or report.get("decision") != _expected_decision(gates):
        raise ComponentCoverageReadinessError("VFAI-FU-011 receipt decision is inconsistent")
    if any(
        (
            report.get("completionClaimed") is not False,
            report.get("exactComponentsPromotedByEvaluator") != 0,
            report.get("corpusFilesModifiedByEvaluator") is not False,
            report.get("genericVariantSelectedByEvaluator") is not False,
            report.get("networkAccessed") is not False,
            report.get("liveStateMutated") is not False,
        )
    ):
        raise ComponentCoverageReadinessError("VFAI-FU-011 receipt overclaims promotion")


def _current_inputs() -> dict[str, Any]:
    master = _read_json(MASTER_BACKLOG_PATH, "master backlog")
    followup = _read_json(FOLLOWUP_BACKLOG_PATH, "follow-up backlog")
    policy = _read_json(POLICY_PATH, "component coverage policy")
    coverage = get_component_coverage()
    parity = verify_ui_coverage(ui_component_catalog=UI_CATALOG_PATH)
    component_types = {entry["componentType"] for entry in coverage["entries"]}
    source_paths = (
        MASTER_BACKLOG_PATH,
        FOLLOWUP_BACKLOG_PATH,
        POLICY_PATH,
        CORPUS_CATALOG_PATH,
        COMPONENT_SERVICE_PATH,
        VERIFIER_PATH,
        AI_ROUTES_PATH,
        UI_CATALOG_PATH,
        UI_TYPES_PATH,
        UI_SERVICES_PATH,
        UI_PANEL_PATH,
        UI_PRESENTATION_PATH,
        BACKEND_CONTROLLER_PATH,
        BACKEND_SERVICE_PATH,
        BACKEND_IMPL_PATH,
        Path(__file__),
    )
    return {
        "master_backlog": master,
        "followup_backlog": followup,
        "policy": policy,
        "coverage": coverage,
        "parity": parity,
        "integration": _integration_evidence(),
        "inputs": _assigned_inputs(policy, component_types),
        "source_sha256": {
            path.relative_to(WORKSPACE_ROOT).as_posix(): _sha256_file(path)
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
    report = _read_json(path, "VFAI-FU-011 readiness receipt")
    validate_report(report)
    if report != _build_current_report(str(report.get("generatedOn"))):
        raise ComponentCoverageReadinessError("VFAI-FU-011 readiness receipt is stale")
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
                "componentEntryCount": report["coverage"]["entryCount"],
                "verifiedComponentCount": report["coverage"]["summary"]["verified"],
                "reportSha256": report["reportSha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
