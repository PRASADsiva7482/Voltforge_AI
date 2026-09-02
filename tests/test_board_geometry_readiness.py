from __future__ import annotations

from copy import deepcopy
import json

import pytest

from tools.evaluate_board_geometry_readiness import (
    BoardGeometryReadinessError,
    POLICY_PATH,
    build_report,
    validate_report,
)


def _master() -> dict[str, object]:
    return {"status": "complete", "items": [{"id": "VFAI-035", "status": "done"}]}


def _followups() -> dict[str, object]:
    return {"items": [{"id": "VFAI-FU-012", "status": "accepted_for_later"}]}


def _coverage() -> dict[str, object]:
    return {
        "reportId": "vfai-fu-001-ui-hardware-coverage",
        "reportSha256": "coverage",
        "summary": {"verified": 8, "variantRequired": 22, "unsupported": 19},
    }


def _parity(*, fixtures: int = 8, mechanical: int = 0) -> dict[str, object]:
    return {
        "boardGeometry": {
            "fixtureCount": fixtures,
            "manufacturerVerifiedPinoutCount": fixtures,
            "mechanicallyVerifiedArtworkCount": mechanical,
            "documentedArtworkLimitationCount": fixtures,
            "pinCount": 328,
        }
    }


def _report(*, browser: bool = False, fixtures: int = 8) -> dict[str, object]:
    return build_report(
        master_backlog=_master(),
        followup_backlog=_followups(),
        policy=json.loads(POLICY_PATH.read_text(encoding="utf-8")),
        coverage=_coverage(),
        parity=_parity(fixtures=fixtures),
        integration={"ready": True},
        browser_review_assigned=browser,
        generated_on="2026-09-01",
        source_sha256={"policy": "source"},
    )


def test_browser_review_is_first_remaining_gate() -> None:
    report = _report()

    assert report["decision"] == "await-browser-geometry-review"
    assert report["savedProjectsRewrittenByEvaluator"] is False
    assert report["simulatorNetSemanticsModifiedByEvaluator"] is False
    validate_report(report)


def test_fixture_drift_fails_before_browser_gate() -> None:
    assert _report(fixtures=7)["decision"] == "repair-manufacturer-pinout-fixtures"
    assert _report(browser=True)["decision"] == "ready-to-complete-geometry-reconciliation"


def test_tampered_receipt_fails_closed() -> None:
    report = _report()
    tampered = deepcopy(report)
    tampered["completionClaimed"] = True

    with pytest.raises(BoardGeometryReadinessError, match="checksum"):
        validate_report(tampered)
