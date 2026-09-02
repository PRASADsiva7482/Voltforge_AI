"""Generate or verify the VFAI-FU-012 board-geometry readiness receipt."""

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
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from hardware_coverage import get_hardware_coverage  # noqa: E402
from tools.verify_hardware_coverage import verify as verify_ui_coverage  # noqa: E402


MASTER_BACKLOG_PATH = AI_ROOT / "AI_MASTER_BACKLOG.json"
FOLLOWUP_BACKLOG_PATH = AI_ROOT / "AI_FOLLOWUP_BACKLOG.json"
POLICY_PATH = AI_ROOT / "hardware_coverage/board-geometry-policy.v1.json"
HARDWARE_SERVICE_PATH = AI_ROOT / "hardware_coverage/service.py"
VERIFIER_PATH = AI_ROOT / "tools/verify_hardware_coverage.py"
UI_BOARD_CATALOG_PATH = UI_ROOT / "src/features/canvas/boardCatalog.ts"
UI_COMPONENT_CATALOG_PATH = UI_ROOT / "src/features/canvas/componentCatalog.ts"
UI_FIXTURE_PATH = UI_ROOT / "src/features/canvas/boardGeometryFixtures.v1.json"
UI_GEOMETRY_PATH = UI_ROOT / "src/features/canvas/boardGeometry.ts"
UI_PIN_REGISTRY_PATH = UI_ROOT / "src/features/canvas/pinRegistry.ts"
UI_COMPONENT_FACTORY_PATH = UI_ROOT / "src/features/canvas/componentFactory.ts"
UI_SETTINGS_PATH = UI_ROOT / "src/features/editor/ProjectSettingsModal.tsx"
UI_STYLES_PATH = UI_ROOT / "src/styles/editor.css"
UI_TEST_PATH = UI_ROOT / "scripts/run-board-geometry-tests.mjs"
UI_PACKAGE_PATH = UI_ROOT / "package.json"
DEFAULT_REPORT_PATH = AI_ROOT / "evaluation/reports/board-geometry-readiness-v1.json"
COMPLETE_STATUSES = frozenset({"complete", "completed", "done"})


class BoardGeometryReadinessError(RuntimeError):
    """VFAI-FU-012 evidence is invalid or stale."""


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
        raise BoardGeometryReadinessError(f"unable to read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise BoardGeometryReadinessError(f"{label} must be a JSON object")
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
        if isinstance(item, Mapping) and item.get("id") == "VFAI-FU-012"
    ] if isinstance(items, list) else []
    if len(matches) != 1:
        raise BoardGeometryReadinessError("expected exactly one VFAI-FU-012 item")
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
        raise BoardGeometryReadinessError("geometry evidence path escapes workspace") from exc
    return path if path.is_file() else None


def _integration_evidence() -> dict[str, Any]:
    catalog = UI_BOARD_CATALOG_PATH.read_text(encoding="utf-8")
    registry = UI_PIN_REGISTRY_PATH.read_text(encoding="utf-8")
    component_factory = UI_COMPONENT_FACTORY_PATH.read_text(encoding="utf-8")
    settings = UI_SETTINGS_PATH.read_text(encoding="utf-8")
    styles = UI_STYLES_PATH.read_text(encoding="utf-8")
    ui_test = UI_TEST_PATH.read_text(encoding="utf-8")
    package = UI_PACKAGE_PATH.read_text(encoding="utf-8")
    checks = {
        "boardTypeSelectsPinoutFixture": all(
            marker in (catalog + registry)
            for marker in (
                "createVerifiedBoardPins(boardType)",
                "createBoardPins(boardItem.footprint, boardItem.type)",
            )
        ),
        "electricalAndGeometryStatusesSeparate": all(
            marker in settings
            for marker in (
                "aiCoverageStatusLabel",
                "boardGeometryStatusLabel",
                "reported independently",
            )
        ),
        "artworkLimitationsPresented": all(
            marker in (settings + styles)
            for marker in ("boardGeometryTooltip", "is-pinout-verified", "is-shared-footprint")
        ),
        "legacySavedPinsFailSafe": all(
            marker in component_factory
            for marker in ("hasLegacyPin", "return existingPins")
        ),
        "deterministicGeometryTestRegistered": all(
            marker in (ui_test + package)
            for marker in ("saved-project compatibility", "test:board-geometry")
        ),
    }
    return {"checks": checks, "ready": all(checks.values())}


def _expected_decision(gates: Mapping[str, Any]) -> str:
    if gates.get("masterBacklogComplete") is not True:
        return "await-master-backlog-completion"
    if gates.get("exactElectricalBoardsCovered") is not True:
        return "repair-exact-board-fixture-coverage"
    if gates.get("manufacturerPinoutFixturesValid") is not True:
        return "repair-manufacturer-pinout-fixtures"
    if gates.get("artworkLimitationsFailClosed") is not True:
        return "repair-mechanical-artwork-boundary"
    if gates.get("integrationAndCompatibilityReady") is not True:
        return "complete-geometry-integration"
    if gates.get("browserGeometryReviewAssigned") is not True:
        return "await-browser-geometry-review"
    return "ready-to-complete-geometry-reconciliation"


def build_report(
    *,
    master_backlog: Mapping[str, Any],
    followup_backlog: Mapping[str, Any],
    policy: Mapping[str, Any],
    coverage: Mapping[str, Any],
    parity: Mapping[str, Any],
    integration: Mapping[str, Any],
    browser_review_assigned: bool,
    generated_on: str,
    source_sha256: Mapping[str, str],
) -> dict[str, Any]:
    if policy.get("sourceFollowup") != "VFAI-FU-012":
        raise BoardGeometryReadinessError("policy is not bound to VFAI-FU-012")
    fu012 = _backlog_item(followup_backlog)
    geometry = parity.get("boardGeometry")
    verified_count = coverage.get("summary", {}).get("verified")
    gates = {
        "masterBacklogComplete": _master_complete(master_backlog),
        "exactElectricalBoardsCovered": verified_count == 8,
        "manufacturerPinoutFixturesValid": bool(
            isinstance(geometry, Mapping)
            and geometry.get("fixtureCount") == 8
            and geometry.get("manufacturerVerifiedPinoutCount") == 8
        ),
        "artworkLimitationsFailClosed": bool(
            isinstance(geometry, Mapping)
            and geometry.get("mechanicallyVerifiedArtworkCount") == 0
            and geometry.get("documentedArtworkLimitationCount") == 8
            and policy.get("mechanicalBoundary", {}).get(
                "generatedArtworkMechanicallyVerified"
            ) is False
        ),
        "integrationAndCompatibilityReady": integration.get("ready") is True,
        "browserGeometryReviewAssigned": browser_review_assigned,
    }
    report: dict[str, Any] = {
        "schemaVersion": 1,
        "reportId": "vfai-fu-012-board-geometry-readiness-v1",
        "generatedOn": generated_on,
        "followup": {"id": fu012.get("id"), "recordedStatus": fu012.get("status")},
        "policy": {"policyId": policy.get("policyId"), "status": policy.get("status")},
        "electricalCoverage": {
            "reportId": coverage.get("reportId"),
            "reportSha256": coverage.get("reportSha256"),
            "verifiedBoardCount": verified_count,
        },
        "geometry": dict(geometry) if isinstance(geometry, Mapping) else {},
        "integration": dict(integration),
        "gates": gates,
        "decision": _expected_decision(gates),
        "completionClaimed": False,
        "savedProjectsRewrittenByEvaluator": False,
        "simulatorNetSemanticsModifiedByEvaluator": False,
        "aiElectricalCoverageModifiedByEvaluator": False,
        "corpusFilesModifiedByEvaluator": False,
        "networkAccessed": False,
        "liveStateMutated": False,
        "sourceSha256": dict(source_sha256),
        "reportSha256": "",
    }
    report["reportSha256"] = _report_digest(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("reportSha256") != _report_digest(report):
        raise BoardGeometryReadinessError("VFAI-FU-012 receipt checksum is invalid")
    gates = report.get("gates")
    if not isinstance(gates, Mapping) or report.get("decision") != _expected_decision(gates):
        raise BoardGeometryReadinessError("VFAI-FU-012 receipt decision is inconsistent")
    if any(
        (
            report.get("completionClaimed") is not False,
            report.get("savedProjectsRewrittenByEvaluator") is not False,
            report.get("simulatorNetSemanticsModifiedByEvaluator") is not False,
            report.get("aiElectricalCoverageModifiedByEvaluator") is not False,
            report.get("corpusFilesModifiedByEvaluator") is not False,
            report.get("networkAccessed") is not False,
            report.get("liveStateMutated") is not False,
        )
    ):
        raise BoardGeometryReadinessError("VFAI-FU-012 receipt overclaims mutation")


def _current_inputs() -> dict[str, Any]:
    master = _read_json(MASTER_BACKLOG_PATH, "master backlog")
    followup = _read_json(FOLLOWUP_BACKLOG_PATH, "follow-up backlog")
    policy = _read_json(POLICY_PATH, "board geometry policy")
    coverage = get_hardware_coverage()
    parity = verify_ui_coverage(
        UI_BOARD_CATALOG_PATH,
        UI_COMPONENT_CATALOG_PATH,
        UI_FIXTURE_PATH,
    )
    browser_receipt = policy.get("completionInputs", {}).get(
        "browserGeometryReviewReceiptPath"
    )
    source_paths = (
        MASTER_BACKLOG_PATH,
        FOLLOWUP_BACKLOG_PATH,
        POLICY_PATH,
        HARDWARE_SERVICE_PATH,
        VERIFIER_PATH,
        UI_BOARD_CATALOG_PATH,
        UI_COMPONENT_CATALOG_PATH,
        UI_FIXTURE_PATH,
        UI_GEOMETRY_PATH,
        UI_PIN_REGISTRY_PATH,
        UI_COMPONENT_FACTORY_PATH,
        UI_SETTINGS_PATH,
        UI_STYLES_PATH,
        UI_TEST_PATH,
        UI_PACKAGE_PATH,
        Path(__file__),
    )
    return {
        "master_backlog": master,
        "followup_backlog": followup,
        "policy": policy,
        "coverage": coverage,
        "parity": parity,
        "integration": _integration_evidence(),
        "browser_review_assigned": _workspace_file(browser_receipt) is not None,
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
    report = _read_json(path, "VFAI-FU-012 readiness receipt")
    validate_report(report)
    if report != _build_current_report(str(report.get("generatedOn"))):
        raise BoardGeometryReadinessError("VFAI-FU-012 readiness receipt is stale")
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
                "verifiedBoardCount": report["electricalCoverage"]["verifiedBoardCount"],
                "geometryFixtureCount": report["geometry"].get("fixtureCount"),
                "reportSha256": report["reportSha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
