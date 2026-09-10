"""Fail-closed family isolation, paraphrase/code contamination and token accounting."""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from data_governance.splitting import partition as p, release as r
from data_governance.splitting.signatures import canonical, sha, features, code_signature, compare


def ancestry(name, **extra):
    return {"kind": "owned-synthetic", "sourceFamilyId": "source:" + name, "documentFamilyId": "document:" + name,
            "repositoryId": "owned-generator", "templateFamilyId": "template:" + name,
            "circuitFamilyId": None, "boardVariant": "BOARD_A", **extra}


def doc(identity, text, *, lineage=None):
    return {"id": identity, "recordId": identity, "lineage": lineage or ancestry(identity),
            "rawSha256": sha("raw:" + text), "normalizedSha256": sha(text), "features": features([text])}


def guard(text="", *, identity="sealed", keys=()):
    return {"id": identity, "features": features([text]), "keys": sorted(keys)}


@pytest.mark.parametrize("text", [
    "Calculate the current through a resistor with voltage 12 volts and resistance 600 ohms.",
    "For a 600 ohm resistor at 12 volts, determine current and voltage.",
    "Please give current for resistance 600 ohms with potential 12 volts. Explain the resistor calculation.",
])
def test_deliberate_exact_and_paraphrased_holdout_rejected(text):
    protected = guard("Calculate the current through a resistor with voltage 12 volts and resistance 600 ohms.")
    result = p.plan_partitions([doc("candidate", text)], [protected])
    assert result["splitCounts"] == {"quarantine": 1}
    assert result["protectedMatches"]


def test_code_variable_renaming_and_comment_changes_rejected():
    original = "```cpp\nint measure(int sample) { return sample * 4 + 7; }\n```"
    renamed = "```cpp\n// independent-looking comment\nint transform(int reading) { return reading * 4 + 7; }\n```"
    result = p.plan_partitions([doc("candidate", renamed)], [guard(original)])
    assert result["protectedMatches"][0]["reason"] == "renamed-code"


def test_parameter_and_board_variant_of_protected_template_rejected():
    family = ancestry("voltage-divider", boardVariant="BOARD_A")
    other = ancestry("other", templateFamilyId=family["templateFamilyId"], boardVariant="BOARD_B")
    result = p.plan_partitions([doc("candidate", "Distinct question text", lineage=other)], [guard(keys=p.lineage_keys(family))])
    assert result["protectedMatches"][0]["reason"] == "protected-lineage-family"


def test_numeric_code_variant_is_a_family_collision():
    original = "```cpp\nint run(int value) { return value * 17 + 31; }\n```"
    changed = "```cpp\nint run(int value) { return value * 19 + 43; }\n```"
    result = p.plan_partitions([doc("candidate", changed)], [guard(original)])
    assert result["protectedMatches"][0]["reason"] == "parameterized-code-family"


def test_containment_detects_holdout_inside_long_padding():
    protected = "Voltage drop across resistors connected sequentially requires identical branch current and total resistance."
    padded = "This newly authored explanation discusses many unrelated observations. " + protected + " Additional bibliography and material follows." * 30
    result = p.plan_partitions([doc("candidate", padded)], [guard(protected)])
    assert result["protectedMatches"][0]["reason"] in {"protected-content-containment", "protected-exact-containment"}


def test_unrelated_control_survives():
    result = p.plan_partitions([doc("candidate", "A quartz oscillator maintains periodic timing through piezoelectric mechanical resonance.")],
                               [guard("Calculate current through a 600 ohm resistor supplied with 12 volts.")])
    assert not result["protectedMatches"] and result["assignments"][0]["split"] != "quarantine"


def test_known_id_cannot_be_relabelled_as_training():
    row = doc("new-document", "Unrelated-looking text")
    row["recordId"] = "historical-id"
    protected = {**guard(), "recordId": "historical-id"}
    assert p.plan_partitions([row], [protected])["protectedMatches"][0]["reason"] == "protected-record-id"


@pytest.mark.parametrize("field,reason", [("rawSha256", "protected-raw-bytes"), ("normalizedSha256", "protected-normalized-bytes")])
def test_protected_byte_hashes_cannot_be_hidden_by_text_or_metadata(field, reason):
    row = doc("changed-id", "short")
    assert p.plan_partitions([row], [{**guard(), field: row[field]}])["protectedMatches"][0]["reason"] == reason


def test_connected_family_inherits_protection_transitively():
    a = doc("a", "Sealed explanation with enough unique words to form protected textual evidence.")
    b = doc("b", "Distinct content for member b", lineage=ancestry("b", templateFamilyId=a["lineage"]["templateFamilyId"]))
    c = doc("c", "Different content for member c", lineage=ancestry("c", documentFamilyId=b["lineage"]["documentFamilyId"]))
    result = p.plan_partitions([a, b, c], [guard("Sealed explanation with enough unique words to form protected textual evidence.")])
    assert result["splitCounts"] == {"quarantine": 3}
    assert result["components"][0]["documents"] == 3


@pytest.mark.parametrize("key", ["sourceFamilyId", "documentFamilyId", "templateFamilyId", "circuitFamilyId", "scenarioFamilyId"])
def test_shared_ancestry_is_never_split_between_partitions(key):
    a, b = ancestry("one", **{key: "shared"}), ancestry("two", **{key: "shared"})
    result = p.plan_partitions([doc("a", "alpha", lineage=a), doc("b", "beta", lineage=b)])
    assert len(result["components"]) == 1
    assert len({row["split"] for row in result["assignments"]}) == 1


def test_repository_documents_group_whole_repository():
    a = ancestry("a", kind="repository-document", repositoryId="repo:commit:abcd")
    b = ancestry("b", kind="repository-document", repositoryId="repo:commit:abcd")
    assert len(p.plan_partitions([doc("a", "alpha", lineage=a), doc("b", "beta", lineage=b)])["components"]) == 1


def test_duplicate_copies_do_not_increase_accepted_records():
    result = p.plan_partitions([doc("a", "Exact raw and normalized payload"), doc("b", "Exact raw and normalized payload")])
    assert sum(row["split"] != "quarantine" for row in result["assignments"]) == 1
    assert result["assignments"][1]["duplicateOf"] == "a"


def test_all_partitions_can_be_populated_without_splitting_families():
    rows = [doc(f"fixture-{i:03}", f"record {i}") for i in range(100)]
    result = p.plan_partitions(rows)
    assert set(result["splitCounts"]) == {"train", "validation", "test"}
    assert all(result["splitCounts"][name] > 0 for name in ("train", "validation", "test"))


def test_reordering_input_does_not_change_split_or_duplicate_representative():
    rows = [doc(f"record-{i}", f"unique {i}") for i in range(20)]
    assert p.plan_partitions(rows) == p.plan_partitions(list(reversed(rows)))


def test_pre_render_reservation_binds_family_and_seed():
    families = [ancestry("first"), ancestry("second")]
    reservation = p.reserve_before_render(families)
    for row in reservation["assignments"]:
        assert p.require_render_assignment(reservation, row["lineage"], row["split"]) == row
    policy = {**p.POLICY, "seed": "different-seed"}
    with pytest.raises(ValueError, match="policy changed"):
        p.require_render_assignment(reservation, families[0], "train", policy=policy)


def test_render_reservation_rejects_relabelled_assignment_or_unreserved_family():
    reservation = p.reserve_before_render([ancestry("recipe")])
    with pytest.raises(ValueError, match="Unreserved"):
        p.require_render_assignment(reservation, ancestry("unreserved"), "train")
    row = reservation["assignments"][0]
    row["split"] = "test" if row["split"] != "test" else "train"
    with pytest.raises(ValueError, match="assignment changed"):
        p.require_render_assignment(reservation, row["lineage"], row["split"])


@pytest.mark.parametrize("key", ["kind", "sourceFamilyId", "documentFamilyId", "repositoryId", "templateFamilyId"])
def test_missing_ancestry_never_defaults_to_random_record_split(key):
    lineage = ancestry("missing")
    lineage.pop(key)
    with pytest.raises(ValueError):
        p.plan_partitions([doc("a", "text", lineage=lineage)])


def test_comparison_budget_fails_closed_instead_of_skipping_checks():
    policy = {**p.POLICY, "maxPairComparisons": 1}
    with pytest.raises(ValueError, match="budget exceeded"):
        p.plan_partitions([doc("a", "text"), doc("b", "different")], [guard("protect")], policy=policy)


def test_code_strings_operators_and_control_flow_preserved():
    assert code_signature('printf("http://fixture.invalid"); // comment') == code_signature('printf("http://fixture.invalid");')
    assert code_signature("if (x > 1) return x + 4;") != code_signature("if (x < 1) return x - 4;")
    assert compare(features(["if (x > 1) return x + 4;"]), features(["if (x < 1) return x - 4;"]), p.POLICY) is None


@pytest.mark.parametrize("literal", ["READY", "Ready  "])
def test_changed_code_string_case_or_spaces_are_not_collapsed(literal):
    left = features(['```cpp\nint run(int value) { print("Ready"); return value; }\n```'])
    right = features(['```cpp\nint run(int value) { print("' + literal + '"); return value; }\n```'])
    assert compare(left, right, p.POLICY) is None
    assert compare(left, right, p.POLICY, protected=True) is None


def test_long_unique_oracle_marker_embedded_in_padding_is_protected():
    marker = "sealed-calibration-answer-fabricated-7ae32d914c"
    result = p.plan_partitions([doc("copied", "Additional prose " + marker + " more unrelated discussion")], [guard(marker)])
    assert result["protectedMatches"][0]["reason"] == "protected-exact-containment"


def test_repetition_never_adds_unique_tokens():
    for passes in (1, 3, 10):
        result = r.exposure_accounting({"train": 100, "validation": 20, "test": 30}, passes)
        assert result["uniqueAcceptedDocumentProxyTokens"]["train"] == 100
        assert result["hypotheticalTrainingExposures"] == 100 * passes
        assert result["observedTrainingExposures"] == result["gen2ReleasedTrainingTokens"] == 0


@pytest.mark.parametrize("passes", [0, -1, 11, True, 1.5])
def test_invalid_repetition_is_not_accepted(passes):
    with pytest.raises(ValueError):
        r.exposure_accounting({"train": 100}, passes)


@pytest.mark.parametrize("usage", p.POLICY["excludedUses"])
def test_partition_files_cannot_authorize_training_uses(monkeypatch, usage):
    calls = []
    monkeypatch.setattr(r, "verify", lambda path, **kwargs: calls.append(kwargs))
    with pytest.raises(ValueError, match="TRAINING_CORPUS_RELEASE_REQUIRED"):
        r.require_partition_use(Path("fixture"), "train", usage)
    assert calls == [{"recompute": True}]


def test_cross_split_audit_rejects_manually_relabelled_related_family():
    rows = [doc("a", "alpha", lineage=ancestry("shared")), doc("b", "beta", lineage=ancestry("shared"))]
    with pytest.raises(ValueError, match="collision"):
        r.cross_split_audit(rows, [{"documentId": "a", "split": "train"}, {"documentId": "b", "split": "test"}], [])


@pytest.fixture(scope="module")
def current_result():
    with patch("socket.socket.connect", side_effect=AssertionError("Partition check must stay offline")):
        return r.compute()


def test_current_ingestion_and_all_guard_families_are_bound(current_result):
    manifest, blobs, scorecard = current_result
    assert scorecard["inputDocuments"] == 227
    assert scorecard["inputEligibleProxyTokens"] == 112418
    assert manifest["plan"]["guards"]["kindCounts"]["frozen-foundation-evaluation"] == 900
    assert manifest["plan"]["guards"]["kindCounts"]["historical-tokenizer-heldout"] >= 23
    assert manifest["plan"]["guards"]["kindCounts"]["extraction-regression-only"] == 17
    assert sum(scorecard["splitCounts"].values()) == 227
    assert not any(scorecard["crossSplitAudit"]["counts"].values())
    assert manifest["trainingAllowed"] is False
    assert set(blobs) == {"train.jsonl", "validation.jsonl", "test.jsonl", "decisions.jsonl", "components.jsonl", "protected-matches.jsonl", "duplicate-matches.jsonl", "scorecard.json"}


def test_immutable_partition_publication_reproduction_and_tamper(tmp_path, current_result, monkeypatch):
    # One fully computed current release is reused here; the final tool evidence
    # separately performs fresh end-to-end recomputation from all source bytes.
    monkeypatch.setattr(r, "compute", lambda: current_result)
    result = r.build(tmp_path)
    path = Path(result["releasePath"])
    assert r.verify(path, recompute=True)["recomputed"]
    before = (path / "manifest.json").read_bytes()
    assert r.build(tmp_path)["manifestSha256"] == sha(before)
    (path / "train.jsonl").write_bytes(b"tampered heldout copy")
    with pytest.raises(ValueError, match="checksum mismatch"):
        r.verify(path)


def test_partial_partition_publication_resumes_identical_bytes(tmp_path, current_result, monkeypatch):
    monkeypatch.setattr(r, "compute", lambda: current_result)
    original = r.ingestion.write_immutable
    writes = []
    def crash(path, data):
        original(path, data)
        writes.append(path)
        if len(writes) == 1:
            raise RuntimeError("simulated interruption")
    monkeypatch.setattr(r.ingestion, "write_immutable", crash)
    with pytest.raises(RuntimeError):
        r.build(tmp_path)
    saved = writes[0].read_bytes()
    monkeypatch.setattr(r.ingestion, "write_immutable", original)
    result = r.build(tmp_path)
    assert writes[0].read_bytes() == saved
    assert r.verify(result["releasePath"], recompute=True)["status"] == "passed"


def test_foundation_oracle_and_context_are_scanned_not_just_prompt():
    candidate = doc("candidate", "The hidden calibration target requires a precision quartz resonator with silver electrodes.")
    protected = guard("The hidden calibration target requires a precision quartz resonator with silver electrodes.")
    assert p.plan_partitions([candidate], [protected])["splitCounts"] == {"quarantine": 1}
