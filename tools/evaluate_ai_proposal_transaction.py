"""Run a content-free static receipt for the VFAI-029 proposal transaction boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


AI_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = AI_ROOT.parent
UI_ROOT = WORKSPACE_ROOT / "Voltforge_UI"
BL_ROOT = WORKSPACE_ROOT / "Voltforge_BL"
REPORT_PATH = AI_ROOT / "evaluation/reports/ai-proposal-transaction-v1.json"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt_digest(report: Mapping[str, Any]) -> str:
    unsigned = dict(report)
    unsigned.pop("reportSha256", None)
    encoded = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_report() -> dict[str, Any]:
    proposal = _read(UI_ROOT / "src/features/ai/aiProposal.ts")
    transaction = _read(UI_ROOT / "src/features/ai/aiProposalTransaction.ts")
    review = _read(UI_ROOT / "src/features/ai/AiProposalReview.tsx")
    chat = _read(UI_ROOT / "src/features/ai/AiChatPanel.tsx")
    validator = _read(UI_ROOT / "src/features/editor/AiValidatorPanel.tsx")
    canvas_store = _read(UI_ROOT / "src/store/canvasStore.ts")
    runtime_delta = _read(UI_ROOT / "src/features/simulator/runtimeDelta.ts")
    domain = _read(UI_ROOT / "src/types/domain.ts")
    project_store = _read(UI_ROOT / "src/store/projectStore.ts")
    editor = _read(UI_ROOT / "src/features/editor/CircuitEditorPage.tsx")
    settings = _read(UI_ROOT / "src/features/editor/ProjectSettingsModal.tsx")
    package_json = json.loads(_read(UI_ROOT / "package.json"))
    ui_tests = _read(UI_ROOT / "scripts/run-ai-proposal-tests.mjs")
    update_request = _read(BL_ROOT / "src/main/java/in/voltforge/api/project/dto/UpdateProjectRequest.java")
    project_service = _read(BL_ROOT / "src/main/java/in/voltforge/api/project/service/impl/ProjectServiceImpl.java")
    errors = _read(BL_ROOT / "src/main/java/in/voltforge/api/common/exception/GlobalExceptionHandler.java")
    backend_test = _read(BL_ROOT / "src/test/java/in/voltforge/api/project/service/impl/ProjectServiceRevisionTest.java")

    checks = {
        "typedDiffsCarrySourceRevisions": all(
            marker in proposal
            for marker in (
                "AiProposalKind",
                "sourceEditorRevision",
                "sourceProjectRevision",
                "kind: 'wire'",
                "kind: 'addition'",
                "kind: 'value-change'",
                "kind: 'removal'",
                "kind: 'code-fix'",
            )
        ),
        "editorRevisionIgnoresSolverNoise": "SIMULATION_RUNTIME_PROPERTY_KEYS" in proposal
        and "modelRevision" in proposal
        and "export const SIMULATION_RUNTIME_PROPERTY_KEYS" in runtime_delta,
        "previewRequiresExplicitSelection": "Review proposed changes" in review
        and 'type="checkbox"' in review
        and "Apply selected" in review
        and "selectedIds" in review,
        "chatUsesAtomicPlanner": "planAiProposalChanges" in chat
        and "commitCanvasSnapshot(nextNodes, nextWires)" in chat
        and "setAppliedTransaction" in chat
        and "onUndo" in chat,
        "validatorUsesSameAtomicPlanner": "planAiProposalChanges" in validator
        and "commitCanvasSnapshot(plan.nextNodes, plan.nextWires)" in validator
        and "setAppliedTransaction" in validator
        and "onUndo={undoAppliedProposal}" in validator,
        "plannerRejectsBeforeStoreMutation": "invalid item rejects the whole plan" in transaction
        and "errors.push" in transaction
        and "nextNodes" in transaction
        and "nextWires" in transaction,
        "canvasCommitIsOneUndoBoundary": "commitCanvasSnapshot: (nodes: CanvasNode[], wires: Wire[])" in canvas_store
        and "current.pushHistory()" in canvas_store
        and "nodesById: buildNodesMap(nodes)" in canvas_store,
        "undoRequiresUnchangedRevision": "afterEditorRevision" in chat
        and "currentEditorRevision !== appliedTransaction.afterEditorRevision" in chat
        and "useCanvasStore.getState().undo()" in chat
        and "currentEditorRevision !== appliedTransaction.afterEditorRevision" in validator,
        "generatedCodeHasNoDirectMutationButton": "onApplyCode" not in chat
        and "onApplyCode" not in editor,
        "persistedSavesUseOptimisticRevision": "expectedRevision?: string" in domain
        and "expectedRevision: currentProject.updatedAt" in editor
        and "expectedRevision: currentProject!.updatedAt" in settings
        and "setProjectUpdatedAt" in project_store,
        "backendRejectsStaleRevision": "expectedRevision" in update_request
        and "PROJECT_REVISION_STALE" in project_service
        and "ConflictException" in project_service
        and "handleConflict" in errors,
        "revisionRegressionCoveragePresent": "rejectsStaleEditorRevisionBeforeMutatingOrSaving" in backend_test
        and "acceptsMatchingEditorRevisionAndReturnsTheSavedProject" in backend_test
        and "test:ai-proposal" in package_json["scripts"]["test"]
        and "planAiProposalChanges" in ui_tests,
        "noThirdPartyGenerationPathAdded": not any(
            marker in (proposal + transaction + chat + validator).lower()
            for marker in ("api.openai.com", "anthropic", "google.generativeai", "third-party model")
        ),
    }
    report = {
        "schemaVersion": 1,
        "reportId": "vfai029-ai-proposal-transaction-v1",
        "generatedOn": "2026-08-31",
        "contractVersion": "1.0.0",
        "checkCount": len(checks),
        "passedCheckCount": sum(bool(value) for value in checks.values()),
        "checks": checks,
        "networkAccessed": False,
        "rawPromptStored": False,
        "rawProjectContextStored": False,
        "rawModelOutputStored": False,
        "thirdPartyGenerationProvider": False,
        "evaluatorSha256": _sha(Path(__file__).resolve()),
        "reportSha256": "",
    }
    report["reportSha256"] = _receipt_digest(report)
    return report


def evaluate(path: Path = REPORT_PATH) -> dict[str, Any]:
    report = build_report()
    if not all(report["checks"].values()):
        failed = [key for key, value in report["checks"].items() if not value]
        raise SystemExit(f"VFAI-029 checks failed: {', '.join(failed)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = evaluate()
    print(json.dumps({"passed": result["passedCheckCount"], "total": result["checkCount"], "reportSha256": result["reportSha256"]}))
