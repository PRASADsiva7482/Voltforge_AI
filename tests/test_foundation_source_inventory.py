"""Source rights, held-out isolation and unique-token accounting cannot be bypassed."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import pytest

from data_governance.foundation import inventory as inv


@pytest.fixture(scope="module")
def frozen_copy(tmp_path_factory):
    root = tmp_path_factory.mktemp("source-inventory")
    for name in inv.LOCKED_SOURCES:
        shutil.copyfile(inv.ROOT / name, root / name)
    original = inv.ROOT
    try:
        inv.ROOT = root
        with patch("socket.socket.connect", side_effect=AssertionError("Inventory must stay offline")):
            inv.freeze()
    finally:
        inv.ROOT = original
    return root


@pytest.fixture
def frozen(frozen_copy, monkeypatch):
    monkeypatch.setattr(inv, "ROOT", frozen_copy)
    return inv.read(frozen_copy / "inventory.v1.json")


def owned_source(tmp_path):
    path = tmp_path / "source.py"
    path.write_text("# synthetic owned source fixture\n", encoding="utf-8")
    return {"sourceId": "vf-test-source", "kind": "project-authored-synthetic-source", "acquisitionMethod": "test fixture",
            "origin": {"path": "source.py", "owner": "fixture", "evidence": "unit-test authored fixture"}, "revision": "1.0.0",
            "checksum": inv.file_hash(path), "license": {"status": "approved", "identifier": "test-only", "evidence": "fixture"},
            "privacy": {"containsPrivateUserData": False, "trainingAllowed": True},
            "allowedUses": ["training"], "prohibitedUses": [], "approval": {"status": "approved", "approvedUses": ["training"]},
            "retention": {"duration": "fixture only"}, "deletionProcedure": "Delete the fixture in the test temporary directory."}


def test_existing_source_permission_does_not_grant_all_uses(tmp_path):
    result = inv.source_decisions(owned_source(tmp_path), root=tmp_path)
    assert result["training"]["allowed"]
    assert not result["runtime-retrieval"]["allowed"]
    assert not result["redistribution"]["allowed"]


@pytest.mark.parametrize("change", ["license", "privacy", "revision", "checksum", "approval", "origin", "acquisition", "retention", "deletion", "changed-bytes"])
def test_incomplete_or_stale_permissions_fail_closed(tmp_path, change):
    source = owned_source(tmp_path)
    if change == "license":
        source["license"]["status"] = "publicly-available"
    elif change == "privacy":
        source["privacy"]["containsPrivateUserData"] = True
    elif change == "revision":
        source["revision"] = "unknown"
    elif change == "checksum":
        source["checksum"] = None
    elif change == "approval":
        source["approval"]["status"] = "pending"
    elif change == "origin":
        source["origin"]["path"] = None
    elif change == "acquisition":
        source.pop("acquisitionMethod")
    elif change == "retention":
        source.pop("retention")
    elif change == "deletion":
        source.pop("deletionProcedure")
    else:
        (tmp_path / "source.py").write_text("changed source", encoding="utf-8")
    assert not inv.source_decisions(source, root=tmp_path)["training"]["allowed"]


@pytest.mark.parametrize("kind", ["user-chat", "private-user-project", "teacher-model-output", "evaluation-material", "secret", "live-web-retrieval", "legacy-origin-unverified"])
def test_excluded_kinds_cannot_be_admitted_by_an_approval_label(tmp_path, kind):
    source = owned_source(tmp_path)
    source["kind"] = kind
    assert not inv.source_decisions(source, root=tmp_path)["training"]["allowed"]


@pytest.mark.parametrize("path", ["../outside.txt", "C:/outside.txt", "//server/share", "source.txt:stream"])
def test_paths_cannot_escape_or_use_alternate_streams(tmp_path, path):
    with pytest.raises(ValueError, match="path"):
        inv.safe_path(path, tmp_path)


def test_owned_generator_does_not_unquarantine_its_legacy_output(frozen):
    assert frozen["sourceCounts"]["approvedForExistingTrainingUse"] == 4
    assert frozen["unitCounts"] == {"quarantined": 12, "approved-existing-shard": 4, "approved-upstream-facts": 8}
    legacy = [row for row in frozen["units"] if row["path"].startswith("model/artifacts/") or row["path"] == "dataset.txt"]
    assert len(legacy) == 12
    assert sum(row["recordCount"] for row in legacy) == 17814
    assert all(not row["countedInExistingPool"] for row in legacy)
    assert frozen["gen2TrainingAllowed"] is False


def test_current_proxy_counts_and_retained_copies_are_separate(frozen):
    pool = frozen["existingPool"]
    assert pool["inputRecords"] == pool["currentApprovedShardRecords"] == 250
    assert pool["uniqueEligibleRecords"] == 227
    assert pool["excluded"] == {"historical-held-out": 23}
    assert pool["proxyTokens"] == 112418
    assert len(frozen["retainedCopies"]) == 8
    assert not any(row["countedAsAdditionalUniqueData"] for row in frozen["retainedCopies"])
    assert not any(row["historicalBytesSubstituted"] for row in frozen["tokenizerHistory"])
    assert frozen["tokenizerHistory"][0]["unresolvedHistoricalShardIds"]


def record(identity, text):
    return {"recordId": identity, "task": "fixture", "input": {"text": text}, "output": {"answer": "fixture answer"}, "metadata": {"provenance": identity}}


class ByteCounter:
    def encode(self, text):
        return list(text.encode())


def test_renamed_ids_provenance_and_retained_copies_cannot_add_unique_tokens():
    first, second = record("one", "owned synthetic fixture"), record("two", "owned synthetic fixture")
    result = inv.summarize_pool([first, second], set(), ByteCounter(), foundation_checker=lambda _: None)
    assert result["uniqueEligibleRecords"] == 1
    assert result["proxyTokens"] == len(inv.payload_text(first).encode())
    assert result["excluded"] == {"exact-payload-duplicate": 1}


def test_renamed_historical_heldout_payload_remains_excluded():
    previous = record("historical-heldout", "same protected payload")
    current = record("renamed-training", "same protected payload")
    result = inv.summarize_pool([current], {"historical-heldout"}, ByteCounter(), heldout_records=[previous], foundation_checker=lambda _: None)
    assert result["inputRecords"] == 1
    assert result["proxyTokens"] == 0
    assert result["excluded"] == {"historical-held-out": 1}


def test_task018_prompt_relabelling_is_excluded_from_the_pool():
    from evaluation.foundation.suite import load_split
    case = load_split("acceptance")[0]
    result = inv.summarize_pool([record("apparently-new", case["scenarioText"])], set(), ByteCounter())
    assert result["proxyTokens"] == 0
    assert result["excluded"] == {"foundation-evaluation-leakage": 1}


def test_leakage_checker_failure_never_becomes_an_empty_collision_list():
    def broken(_):
        raise ValueError("broken frozen suite")
    with pytest.raises(ValueError, match="broken frozen suite"):
        inv.summarize_pool([record("one", "example")], set(), ByteCounter(), foundation_checker=broken)


def test_repetition_changes_exposures_but_never_unique_tokens():
    policy = inv.read(inv.ROOT / "policy.v1.json")
    result = inv.token_gap(100, policy)
    assert result[0]["additionalUniqueProxyTokensAtOnePass"] == 1_999_999_900
    assert [row["uniqueTokensUnchanged"] for row in result[0]["repeatSensitivity"]] == [100, 100, 100]
    assert [row["exposuresIfRepeated"] for row in result[0]["repeatSensitivity"]] == [100, 300, 1000]
    assert inv.token_gap(0, policy)[0]["passesOverExistingPoolToReachExposures"] is None
    with pytest.raises(ValueError):
        inv.token_gap(True, policy)


def test_external_observations_and_private_exclusions_grant_no_training(frozen):
    assert len(frozen["proposals"]) == 7
    assert all(row["revision"] is row["contentSha256"] is None and not row["trainingAllowed"] and not row["acquired"] for row in frozen["proposals"])
    assert all(not row["contentRead"] and not row["trainingAllowed"] for row in frozen["excludedSources"])
    openstax = next(row for row in frozen["proposals"] if "openstax" in row["sourceId"])
    assert "permission" in openstax["blockingReason"]


def test_source_use_interface_requires_a_separate_gen2_corpus_release(frozen):
    receipt = inv.require_source_permission("vf-src-verified-synthetic-pipeline-v1", "training")
    assert receipt["sourcePermissionVerified"] and receipt["gen2CorpusReleaseRequired"]
    assert not receipt["trainingRunAllowed"]
    for identity, use in (("vf-src-legacy-dataset-txt-v0", "training"), ("vf-proposed-freertos-kernel", "training"),
                          ("vf-src-verified-synthetic-pipeline-v1", "redistribution"), ("not-registered", "training")):
        with pytest.raises(ValueError, match="SOURCE_USE_DENIED"):
            inv.require_source_permission(identity, use)


def test_frozen_inventory_refuses_overwrite_and_detects_policy_tampering(frozen):
    with pytest.raises(ValueError, match="never overwrite"):
        inv.freeze()
    path = inv.ROOT / "policy.v1.json"
    saved = path.read_bytes()
    try:
        path.write_bytes(saved + b"\n")
        with pytest.raises(ValueError, match="hash mismatch"):
            inv.verify()
    finally:
        path.write_bytes(saved)


def test_inventory_recomputes_offline_from_exact_inputs(frozen):
    with patch("socket.socket.connect", side_effect=AssertionError("Network forbidden")):
        assert inv.verify(recompute=True)["recomputed"]
