"""Pinned acquisition, exact source-use decisions and preserved corpus exclusions."""
import copy
import json
from pathlib import Path
import shutil

import pytest

from data_governance.source_expansion import acquisition, admission, candidate
from data_governance.ingestion.normalization import Quarantine
from model.gen2_tokenizer.release import require_corpus_fitting

PACKET = acquisition.OUTPUT / "6d8a00721ddcb89a037c75282cabbd417300a19672a8706cfc474780baf40328"


@pytest.fixture(scope="module")
def verified_packet():
    return acquisition.verify(PACKET)


def test_exact_packet_and_recorded_source_use_inventory(verified_packet):
    decision, packet = admission.verify()
    assert packet == verified_packet
    assert len(decision["sources"]) == 5
    assert sum(row["kind"] == "candidate-source" for row in packet["files"]) == 24
    assert packet["denied"] == [] and packet["acquiredBytes"] == 546522
    assert decision["reviewer"] == {"kind": "coding-agent", "name": "Codex"}
    assert decision["corpusTrainingAllowed"] is False


@pytest.mark.parametrize("name", ["../LICENSE", "/LICENSE", "C:/LICENSE", "file?x", "dir//file", "a/./b", "a\\b", "x\0y"])
def test_source_paths_are_bounded(name):
    with pytest.raises(ValueError, match="Unsafe selected"):
        acquisition.safe_path(name)


@pytest.mark.parametrize("revision", ["main", "HEAD", "0" * 39, "a" * 40 + "?x=1"])
def test_mutable_or_malformed_revision_is_denied(revision):
    source = {**acquisition.CATALOG["sources"][0], "revision": revision}
    with pytest.raises(ValueError, match="pinned repository commit"):
        acquisition.address(source)


def test_redirects_are_not_followed():
    with pytest.raises(ValueError, match="redirects are forbidden"):
        acquisition.NoRedirect().redirect_request(None, None, 302, "found", {}, "http://127.0.0.1/private")


def test_git_blob_identity_matches_independent_known_hash():
    assert acquisition.blob_sha(b"test content\n") == "d670460b4b4aece5915caf5c68d12f560a9fe3e4"


def test_complete_mit_grant_accepts_only_reviewed_spelling_variation():
    source = acquisition.CATALOG["sources"][2]
    raw = (PACKET / source["sourceId"] / source["licensePath"]).read_bytes()
    acquisition.verify_mit(raw)
    acquisition.verify_mit(raw.replace(b"NON-INFRINGEMENT", b"NONINFRINGEMENT"))
    for altered in (raw + b"\nOnly for nonprofit use.", raw.replace(b"without restriction", b"with restriction"), raw.replace(b"permission notice shall be included", b"permission notice may be omitted")):
        with pytest.raises(ValueError, match="Complete MIT notice"):
            acquisition.verify_mit(altered)


@pytest.mark.parametrize("suffix", ["\nAll rights reserved.", "\nSPDX-License-Identifier: GPL-3.0", "\nNon-commercial use only.", "\nNo derivatives."])
def test_selected_file_exceptions_cannot_inherit_root_permission(suffix):
    source = acquisition.CATALOG["sources"][0]
    license_raw = (PACKET / source["sourceId"] / source["licensePath"]).read_bytes()
    with pytest.raises(Quarantine, match="license-exception"):
        acquisition.review_selected(source, "file.md", ("Reviewed public text." + suffix).encode(), license_raw)


def test_secrets_are_rejected_before_source_review():
    source = acquisition.CATALOG["sources"][0]
    license_raw = (PACKET / source["sourceId"] / source["licensePath"]).read_bytes()
    raw = ("password=" + "fixture" * 3).encode()
    with pytest.raises(Quarantine, match="secret-pattern"):
        acquisition.review_selected(source, "fixture.md", raw, license_raw)


def test_ancestor_license_and_child_tree_changes_are_rejected():
    source = acquisition.CATALOG["sources"][2]
    tree = json.loads((PACKET / source["sourceId"] / "tree.json").read_text(encoding="utf-8"))
    acquisition.verify_tree(source, tree)
    extra = copy.deepcopy(tree)
    extra["tree"].append({"path": "docs/setup/LICENSE.custom", "type": "blob"})
    with pytest.raises(ValueError, match="Unreviewed ancestor"):
        acquisition.scope_license_paths(source, extra)
    changed = copy.deepcopy(tree)
    changed["directoryTrees"][-1]["body"]["sha"] = "0" * 40
    with pytest.raises(ValueError, match="tree identity changed"):
        acquisition.verify_tree(source, changed)
    missing = copy.deepcopy(tree)
    missing["directoryTrees"].pop()
    with pytest.raises(ValueError, match="Incomplete ancestor"):
        acquisition.verify_tree(source, missing)


@pytest.mark.parametrize("source_index", [0, 1, 3])
def test_training_source_input_approval_is_exact(source_index):
    source = acquisition.CATALOG["sources"][source_index]
    row = admission.require_source_input(source["sourceId"], source["selectedPaths"][0], "tokenizer-fitting-input")
    assert row["sourceId"] == source["sourceId"]
    with pytest.raises(ValueError, match="BYTES_NOT_APPROVED"):
        admission.require_source_input(source["sourceId"], "unselected.md", "pretraining-input")


@pytest.mark.parametrize("source_index", [2, 4])
@pytest.mark.parametrize("usage", ["tokenizer-fitting-input", "pretraining-input", "redistribution", "runtime-retrieval"])
def test_validation_families_cannot_be_fitted(source_index, usage):
    source = acquisition.CATALOG["sources"][source_index]
    with pytest.raises(ValueError, match="USE_NOT_APPROVED"):
        admission.require_source_input(source["sourceId"], source["selectedPaths"][0], usage)
    assert admission.require_source_input(source["sourceId"], source["selectedPaths"][0], "validation-input")["kind"] == "candidate-source"


def test_source_decision_cannot_turn_into_corpus_approval(tmp_path, monkeypatch):
    decision = json.loads(admission.DECISION_PATH.read_text(encoding="utf-8"))
    decision["corpusTrainingAllowed"] = True
    path = tmp_path / "admission.json"
    path.write_text(json.dumps(decision), encoding="utf-8")
    monkeypatch.setattr(admission, "DECISION_PATH", path)
    with pytest.raises(ValueError, match="scope changed"):
        admission.verify()


def test_source_decision_cannot_change_reserved_family_purpose(tmp_path, monkeypatch):
    decision = json.loads(admission.DECISION_PATH.read_text(encoding="utf-8"))
    decision["sources"][2]["allowedUses"].append("pretraining-input")
    path = tmp_path / "admission.json"
    path.write_text(json.dumps(decision), encoding="utf-8")
    monkeypatch.setattr(admission, "DECISION_PATH", path)
    with pytest.raises(ValueError, match="reserved family"):
        admission.verify()


@pytest.fixture(scope="module")
def expanded():
    manifests = list(candidate.OUTPUT.glob("*/manifest.json"))
    assert len(manifests) == 1, "Build the exact expanded candidate before integration tests"
    return manifests[0].parent


def test_expanded_candidate_reproduces_and_preserves_old_quarantines(expanded):
    candidate.verify(expanded, recompute=True)
    score = candidate.read(expanded / "scorecard.json")
    assert score["candidateDocuments"] == 62 and score["newSourceFiles"] == 24
    assert score["newSourceUseDecisionsRecorded"] == 5
    assert score["inheritedCorpusQuarantinesPreserved"] == 7
    assert score["inheritedDomainExclusionsPreserved"] == 50
    old_ids = {json.loads(line)["id"] for line in (candidate.PRIOR / "quarantine.jsonl").read_text(encoding="utf-8").splitlines()}
    new_ids = {json.loads(line)["id"] for line in (expanded / "quarantine.jsonl").read_text(encoding="utf-8").splitlines()}
    assert old_ids <= new_ids


def test_no_reserved_validation_document_enters_training(expanded):
    for line in (expanded / "train.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        assert row["reservedSplit"] == "train"
    for line in (expanded / "validation.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        assert row["reservedSplit"] == "validation"


@pytest.mark.parametrize("usage", ["tokenizer-fitting", "pretraining", "sft", "validation-evaluation"])
def test_source_approval_does_not_bypass_expanded_corpus_gate(expanded, usage):
    with pytest.raises(ValueError, match="EXPANDED_CORPUS_RELEASE_GATES_UNMET"):
        candidate.require_use(expanded, usage)
    with pytest.raises(ValueError, match="GEN2_CORPUS_NOT_ADMITTED"):
        require_corpus_fitting(expanded)


def test_modified_acquisition_is_rejected(tmp_path, verified_packet):
    copied = tmp_path / PACKET.name
    shutil.copytree(PACKET, copied)
    item = next(row for row in verified_packet["files"] if row["kind"] == "candidate-source")
    (copied / item["path"]).write_bytes(b"modified public test source")
    with pytest.raises(ValueError, match="evidence bytes changed"):
        acquisition.verify(copied)
