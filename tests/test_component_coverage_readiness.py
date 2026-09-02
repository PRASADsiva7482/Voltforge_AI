from __future__ import annotations

from copy import deepcopy
import json

import pytest

from tools.evaluate_component_coverage_readiness import (
    ComponentCoverageReadinessError,
    POLICY_PATH,
    build_report,
    validate_report,
)


def _master() -> dict[str, object]:
    return {"status": "complete", "items": [{"id": "VFAI-035", "status": "done"}]}


def _followups() -> dict[str, object]:
    return {"items": [{"id": "VFAI-FU-011", "status": "accepted_for_later"}]}


def _coverage(*, fail_closed: bool = True) -> dict[str, object]:
    entries = [
        {
            "aiElectricalClaimsAllowed": False if fail_closed else True,
            "physicalIdentitySelected": False,
        }
        for _ in range(60)
    ]
    return {
        "reportId": "vfai-fu-011-ui-component-coverage",
        "reportSha256": "coverage",
        "entryCount": 60,
        "summary": {
            "verified": 0,
            "variantRequired": 51,
            "simulationOnly": 9,
            "unsupported": 0,
        },
        "genericLabelsMaySelectCandidate": False,
        "entries": entries,
    }


def _report(
    *,
    parity: bool = True,
    integration: bool = True,
    owner: bool = False,
    immutable: bool = False,
    selection: bool = False,
    approved: bool = False,
) -> dict[str, object]:
    return build_report(
        master_backlog=_master(),
        followup_backlog=_followups(),
        policy=json.loads(POLICY_PATH.read_text(encoding="utf-8")),
        coverage=_coverage(),
        parity={"componentEntryCount": 60 if parity else 59},
        integration={"ready": integration},
        inputs={
            "ownerPriorityReceiptAssigned": owner,
            "immutableCorpusReleaseReady": immutable,
            "exactCandidateSelectionAssigned": selection,
            "approvedComponentTypesValid": approved,
        },
        generated_on="2026-09-01",
        source_sha256={"policy": "source"},
    )


def test_owner_priorities_are_first_remaining_external_gate() -> None:
    report = _report()

    assert report["decision"] == "await-owner-prioritized-exact-components"
    assert report["exactComponentsPromotedByEvaluator"] == 0
    assert report["corpusFilesModifiedByEvaluator"] is False
    validate_report(report)


def test_exact_promotion_gates_advance_without_claiming_completion() -> None:
    assert _report(owner=True)["decision"] == "await-immutable-component-corpus-release"
    assert (
        _report(owner=True, immutable=True)["decision"]
        == "await-evidenced-exact-candidate-selection"
    )
    assert (
        _report(owner=True, immutable=True, selection=True)["decision"]
        == "await-approved-exact-component-types"
    )
    report = _report(owner=True, immutable=True, selection=True, approved=True)
    assert report["decision"] == "ready-for-exact-component-corpus-release"
    assert report["completionClaimed"] is False
    validate_report(report)


def test_tampered_receipt_fails_closed() -> None:
    report = _report()
    tampered = deepcopy(report)
    tampered["completionClaimed"] = True

    with pytest.raises(ComponentCoverageReadinessError, match="checksum"):
        validate_report(tampered)
