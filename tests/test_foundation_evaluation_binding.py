"""The additive threshold binding preserves design and cannot approve a model."""
from copy import deepcopy

import pytest

from tools import verify_foundation_evaluation_binding as binding


def test_current_binding_verifies_exact_design_and_suite_bytes():
    result = binding.verify()
    assert result["status"] == "passed"
    assert result["thresholdState"] == "frozen-independent-suite"
    assert result["historicalDesignFilesPreserved"] == 4
    assert result["suite"]["counts"]["acceptance"] == 500
    assert result["modelReleaseApproved"] is result["activationAllowed"] is False


@pytest.mark.parametrize("change", ["approval", "hash", "missing-input", "suite", "state", "rewrite"])
def test_binding_changes_cannot_borrow_the_original_acceptance(change):
    value = deepcopy(binding.read(binding.PATH))
    if change == "approval":
        value["activationAllowed"] = True
    elif change == "hash":
        value["inputs"][0]["sha256"] = "0" * 64
    elif change == "missing-input":
        value["inputs"].pop()
    elif change == "suite":
        value["suiteSha256"] = "0" * 64
    elif change == "state":
        value["thresholdState"] = "model-approved"
    else:
        value["historicalDesignRewritten"] = True
    with pytest.raises(ValueError):
        binding.verify(value)
