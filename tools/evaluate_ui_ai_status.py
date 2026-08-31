"""Run a content-free static receipt for the VFAI-028 UI AI boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


AI_ROOT = Path(__file__).resolve().parents[1]
UI_ROOT = AI_ROOT.parent / "Voltforge_UI"
REPORT_PATH = AI_ROOT / "evaluation/reports/ui-ai-status-v1.json"


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
    panel = _read(UI_ROOT / "src/features/ai/AiChatPanel.tsx")
    presentation = _read(UI_ROOT / "src/features/ai/aiPresentation.ts")
    css = _read(UI_ROOT / "src/styles/components.css")
    package_json = json.loads(_read(UI_ROOT / "package.json"))
    test_script = _read(UI_ROOT / "scripts/run-ai-presentation-tests.mjs")
    domain = _read(UI_ROOT / "src/types/domain.ts")
    services = _read(UI_ROOT / "src/api/services.ts")

    checks = {
        "explicitLocalRunStates": all(
            marker in presentation
            for marker in (
                "'connecting'",
                "'checking'",
                "'streaming'",
                "'complete'",
                "'partial'",
                "'cancelled'",
                "'offline'",
                "'error'",
            )
        ),
        "localModelAndFallbackAreNamed": "VoltForge local model" in presentation
        and "Deterministic engineering tools" in presentation
        and "Unavailable local AI" in presentation,
        "sourceTrailCoversEvidenceClasses": all(
            marker in presentation
            for marker in (
                "local-model",
                "deterministic-tools",
                "project-context",
                "local-docs",
                "internet-evidence",
            )
        )
        and "collectAiSources(msg.sourceMetadata)" in panel,
        "uncertaintyIsParsedAndVisible": "eventType === 'uncertainty'" in panel
        and "Evidence uncertainty" in panel
        and "missingEvidence" in panel,
        "cancellationUsesAbortSignal": "new AbortController()" in panel
        and "abortRef.current.abort()" in panel
        and "Cancel VoltForge AI response" in panel
        and "AbortSignal" in services,
        "projectSwitchClearsConversationScope": "previousProjectIdRef" in panel
        and "setSessionId(undefined)" in panel
        and "setMessages([])" in panel
        and "abortRef.current?.abort()" in panel,
        "safeErrorsDoNotRenderUpstreamText": "safeAiErrorMessage" in panel
        and "event.message" not in panel
        and "event.error" not in panel
        and "console.error('Stream error:'" not in panel,
        "canonicalSsePayloadRemainsSupported": "canonicalPayload" in panel
        and "const event = { ...envelope, ...canonicalPayload }" in panel,
        "confidenceAndCitationsAreAccessible": "Low confidence" in panel
        and 'aria-label="Evidence and citations"' in panel
        and "citation.snippet" in panel
        and "noopener noreferrer" in panel,
        "statusLiveRegionIsAccessible": 'role="status"' in panel
        and 'aria-live="polite"' in panel
        and "vf-ai-chat__status" in css,
        "memoryUnavailableAndControlsAreExplicit": "memoryLoadState" in panel
        and "Project memory is unavailable" in panel
        and "setMemoryEnabled" in panel
        and "clearMemory" in panel
        and "aria-labelledby=\"vf-ai-memory-heading\"" in panel,
        "reducedMotionIsHandled": "prefers-reduced-motion: reduce" in css
        and "vf-ai-chat__status-indicator" in css,
        "uiRegressionScriptIsInDefaultSuite": "test:ai-presentation" in package_json["scripts"]["test"]
        and package_json["scripts"].get("test:ai-presentation") == "node scripts/run-ai-presentation-tests.mjs"
        and "classifyAiTerminal" in test_script,
        "contractTypesRemainVersioned": "export type AiSseEvent" in domain
        and "contractVersion: '1.0.0'" in domain
        and "AiInternetRetrievalStatus" in domain,
        "noThirdPartyGenerationLabel": not any(
            marker in (panel + presentation).lower()
            for marker in ("api.openai.com", "anthropic", "google.generativeai", "third-party model")
        ),
    }
    report = {
        "schemaVersion": 1,
        "reportId": "vfai028-ui-ai-status-v1",
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
        raise SystemExit(f"VFAI-028 checks failed: {', '.join(failed)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = evaluate()
    print(json.dumps({"passed": result["passedCheckCount"], "total": result["checkCount"], "reportSha256": result["reportSha256"]}))
