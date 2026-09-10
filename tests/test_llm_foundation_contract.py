"""Adversarial design mutations must not acquire Gen2 foundation acceptance."""
from copy import deepcopy
import sys

import pytest

from tools import verify_llm_foundation as foundation


def documents():
    return [foundation.read(foundation.AI / path) for path in foundation.FILES]


def test_design_is_complete_and_does_not_authorize_a_model_release():
    native_already_imported = "torch" in sys.modules
    report = foundation.verify()
    assert report["status"] == "passed"
    assert report["mappedFollowups"] == 15
    assert report["modelReleaseApproved"] is False
    assert report["nativeModelImported"] is native_already_imported
    assert report["nextTask"] == "LLM-TASK-018"


@pytest.mark.parametrize("field", ["importedPretrainedWeightsAllowed", "thirdPartyTokenizerAllowed",
                                  "hostedGenerationAllowed", "liveChatTrainingAllowed", "workstationGpuIsArchitectureLimit"])
def test_forbidden_direction_cannot_be_enabled(field):
    contract, policy, mapping = documents()
    contract["decisions"][field] = True
    with pytest.raises(ValueError, match="Forbidden foundation decision"):
        foundation.verify(contract, policy, mapping)


@pytest.mark.parametrize("field", ["trainingAllowedByThisContract", "activationAllowedByThisContract", "designValidationIsReleaseEvidence"])
def test_a_design_cannot_grant_downstream_acceptance(field):
    contract, policy, mapping = documents()
    contract["acceptance"][field] = True
    with pytest.raises(ValueError, match="Design cannot approve"):
        foundation.verify(contract, policy, mapping)


@pytest.mark.parametrize("stage_index", [0, 1])
def test_base_or_instruction_acceptance_cannot_skip_assistant_gates(stage_index):
    contract, policy, mapping = documents()
    policy["stages"][stage_index]["chatEligible"] = True
    with pytest.raises(ValueError, match="Only integrated assistant"):
        foundation.verify(contract, policy, mapping)


def test_instruction_stage_cannot_use_an_unrelated_or_pretrained_parent():
    contract, policy, mapping = documents()
    policy["stages"][1]["parentKind"] = "arbitrary-checkpoint"
    with pytest.raises(ValueError, match="Invalid release parent"):
        foundation.verify(contract, policy, mapping)


def test_existing_artifact_cannot_be_renamed_as_the_new_family():
    contract, policy, mapping = documents()
    contract["identity"]["baseExample"] = "vfdlm-g1-edge-v0.1.0"
    with pytest.raises(ValueError, match="Canonical Gen2 identity"):
        foundation.verify(contract, policy, mapping)


def test_missing_independent_threshold_policy_cannot_be_treated_as_a_pass():
    contract, policy, mapping = documents()
    policy["thresholdContract"]["missingOrUnfrozenDecision"] = "pass"
    with pytest.raises(ValueError, match="Missing acceptance evidence"):
        foundation.verify(contract, policy, mapping)


def test_release_policy_cannot_reopen_accepted_history():
    contract, policy, mapping = documents()
    policy["lifecycle"]["transitions"]["accepted"] = ["candidate"]
    with pytest.raises(ValueError, match="Release transition drift"):
        foundation.verify(contract, policy, mapping)


@pytest.mark.parametrize("change", ["missing-module", "unknown-task", "duplicate-module"])
def test_module_boundaries_have_complete_real_task_owners(change):
    contract, policy, mapping = documents()
    if change == "missing-module":
        contract["modules"].pop()
    elif change == "unknown-task":
        contract["modules"][0]["ownerTasks"] = ["LLM-TASK-999"]
    else:
        contract["modules"][-1] = deepcopy(contract["modules"][0])
    with pytest.raises(ValueError, match="module boundary|implementation task"):
        foundation.verify(contract, policy, mapping)


@pytest.mark.parametrize("change", ["prefix", "revision", "route"])
def test_current_route_and_project_revision_contract_cannot_be_dropped(change):
    contract, policy, mapping = documents()
    if change == "prefix":
        contract["api"]["springPrefix"] = "/new-ai"
    elif change == "revision":
        contract["api"]["revisionFields"] = []
    else:
        contract["api"]["pythonRoutes"].append(["POST", "/not-an-existing-route"])
    with pytest.raises(ValueError, match="prefix drift|Revision binding|route declaration"):
        foundation.verify(contract, policy, mapping)


@pytest.mark.parametrize("change", ["omit", "status", "scope", "rewrite", "hash"])
def test_old_followups_and_policies_cannot_be_silently_reclassified(change):
    contract, policy, mapping = documents()
    if change == "omit":
        mapping["followups"].pop()
    elif change == "status":
        mapping["followups"][-1]["observedStatus"] = "done"
    elif change == "scope":
        mapping["policyOverrides"][0]["appliesTo"] = "all-models"
    elif change == "rewrite":
        mapping["policyOverrides"][0]["changesHistoricalPolicy"] = True
    else:
        mapping["protectedFiles"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="mapping|follow-up status|Policy replacement|Historical file changed"):
        foundation.verify(contract, policy, mapping, preserve_baseline=True)


def test_task_preservation_proof_is_separate_from_later_design_validation(monkeypatch):
    original = foundation.digest
    def changed_registry(path):
        return "0" * 64 if path == foundation.AI / "model/registry/active_model.json" else original(path)
    monkeypatch.setattr(foundation, "digest", changed_registry)
    assert foundation.verify()["preservationChecked"] is False
    with pytest.raises(ValueError, match="Historical file changed"):
        foundation.verify(preserve_baseline=True)


def test_duplicate_json_keys_are_rejected_instead_of_last_value_winning():
    with pytest.raises(ValueError, match="Duplicate contract key"):
        foundation.unique_object([("activationAllowed", False), ("activationAllowed", True)])
