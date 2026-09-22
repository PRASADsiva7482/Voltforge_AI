"""Independent corpus controls; no fixture is an admitted training source."""
from copy import deepcopy
import json
from pathlib import Path
import random
import shutil

import pytest

from data_governance.splitting.partition import plan_partitions, POLICY as MATCH_POLICY
from data_governance.splitting.signatures import features, canonical, sha
from data_governance.pretraining import acquisition, acquisition_v2, indexed, mixture, corpus


def lineage(name):
    return {"kind": "owned-synthetic", "repositoryId": "fixture-controls", "sourceFamilyId": name,
            "documentFamilyId": name, "templateFamilyId": name}


def doc(name, text, **kwargs):
    return {"id": name, "recordId": name, "lineage": lineage(name), "features": features([text]), **kwargs}


def guard(name, text, **kwargs): return {"id": name, "keys": [], "features": features([text]), **kwargs}


def recall_fixture():
    rng = random.Random(2301)
    rows, guards = [], []
    for i in range(80):
        words = ["w" + sha(f"{i}:{j}")[:12] for j in range(12)]
        original = " ".join(words)
        guards.append(guard(f"guard-{i:03}", original))
        if i % 4 == 0: candidate = original
        elif i % 4 == 1: candidate = original + " additional prose"
        elif i % 4 == 2:
            rng.shuffle(words)
            candidate = " ".join(words[:-1] + ["replacementword"])
        else: candidate = " ".join("other" + sha(f"{i}:{j}")[:14] for j in range(12))
        rows.append(doc(f"doc-{i:03}", candidate))
    guards.extend([guard("guard-literal", "single_long_literal_without_six_words"), guard("guard-code", "int accumulate(int x) { return x + 13; }"),
                   guard("guard-id", "", recordId="protected-id"), guard("guard-raw", "", rawSha256="a" * 64),
                   guard("guard-lineage", "", keys=["source:protected-family"])])
    rows.extend([doc("doc-literal", "prefix single_long_literal_without_six_words suffix"), doc("doc-code", "int renamed(int item) { return item + 19; }"),
                 doc("protected-id", "unrelated"), doc("doc-raw", "unrelated raw", rawSha256="a" * 64),
                 doc("doc-family", "unrelated family", lineage=lineage("protected-family"))])
    rows.extend([doc("copy-" + row["id"], "", features=row["features"]) for row in rows[:8]])
    return rows, guards


def test_indexed_matches_equal_exhaustive_original_policy():
    rows, guards = recall_fixture()
    indexed_result = indexed.scan(rows, guards)
    exhaustive = plan_partitions(rows, guards)
    for key in ("protectedMatches", "duplicateMatches"):
        assert indexed_result[key] == exhaustive[key]
    assert len(indexed_result["protectedMatches"]) >= 50
    assert indexed_result["thresholdsChanged"] is False


def test_metadata_boilerplate_does_not_hide_answer_or_code_contamination():
    shared = "An invented evaluation-only board identifier and protocol envelope sentence."
    prose = "A fictional particle record carries zeta aperture observations across phases."
    left = indexed.payload_features({"protocol": {"label": shared}, "answer": prose})
    right = indexed.payload_features({"protocol": {"label": shared}, "answer": "Quartz telemetry records contain independent rotating optical alignment measurements."})
    from data_governance.splitting.signatures import compare
    assert compare(left, right, MATCH_POLICY) is None
    assert compare(indexed.payload_features({"answer": shared}), features([shared]), MATCH_POLICY, protected=True)
    assert compare(indexed.payload_features({"context": shared}), features([shared]), MATCH_POLICY, protected=True)
    code = "int preserve(int counter) { return counter * 37; }"
    assert compare(indexed.payload_features({"code": code}), features([code]), MATCH_POLICY, protected=True)


def test_index_retains_exact_case_and_space_in_code_literals():
    rows = [doc("literal-a", 'void send() { report("Hello World"); }'), doc("literal-b", 'void send() { report("hello  world"); }')]
    assert indexed.scan(rows, []) ["duplicateMatches"] == []


def test_resources_fail_closed_on_common_words():
    rows = [doc(str(i), "shared aperture contains phase quaternion tensor values " + str(i)) for i in range(30)]
    with pytest.raises(ValueError, match="comparison budget"): indexed.scan(rows, [], maximum_pairs=10)


def test_character_resource_bound(monkeypatch):
    monkeypatch.setitem(indexed.POLICY, "maximumFeatureCharacters", 10)
    with pytest.raises(ValueError, match="character budget"): indexed.Index([doc("oversize", "A sufficiently lengthy resource control document.")])


def test_reserved_bridge_and_repository_siblings_never_cross_splits():
    text = "A repeated coherent optical controller observes a long phase transition."
    rows = [doc("one", text, reservedSplit="train"), doc("two", text, reservedSplit="test")]
    decisions, _ = indexed.partition(rows, [])
    assert {row["reason"] for row in decisions} == {"cross-reservation-bridge"}
    assert {row["split"] for row in decisions} == {"quarantine"}
    rows = [doc("repo" + str(i), "independent" + str(i), lineage={"kind": "repository-document", "repositoryId": "same-repo", "sourceFamilyId": "file" + str(i), "documentFamilyId": "file" + str(i)}) for i in range(3)]
    decisions, _ = indexed.partition(rows, [])
    assert len({row["split"] for row in decisions}) == 1


def test_inherited_exclusion_remains_excluded():
    decisions, _ = indexed.partition([doc("old-excluded", "protected historical program", inheritedExclusion=True)], [])
    assert decisions[0]["split"] == "quarantine"


@pytest.mark.parametrize("path", ["../LICENSE", "/etc/file", "nested/../../file", "file?query", "file#fragment", "file\\name"])
def test_acquisition_rejects_path_injection(path):
    with pytest.raises(ValueError): acquisition_v2.url(acquisition.CATALOG["sources"][0], path)


@pytest.mark.parametrize("revision", ["main", "V11.1.0", "a" * 39, "g" * 40])
def test_acquisition_requires_exact_commit(revision):
    with pytest.raises(ValueError): acquisition_v2.url(acquisition.CATALOG["sources"][0] | {"revision": revision}, "LICENSE")


def test_acquisition_forbids_redirects():
    with pytest.raises(ValueError, match="redirect"): acquisition_v2.NoRedirect().redirect_request(None, None, 302, "", {}, "http://localhost/private")


@pytest.fixture(scope="module")
def acquired():
    manifest = acquisition_v2.verify(corpus.ACQUISITION)
    return manifest


def test_complete_mit_grant_is_required_and_notices_survive(acquired):
    source = acquisition.CATALOG["sources"][1]
    license_bytes = (corpus.ACQUISITION / source["sourceId"] / source["licensePath"]).read_bytes()
    raw = (corpus.ACQUISITION / source["sourceId"] / "list.c").read_bytes()
    result = acquisition_v2.review_license(source, license_bytes, raw, "list.c")
    assert result["completeMITGrantChecked"] and result["noticesPreserved"]
    assert b"All Rights Reserved" in raw
    for changed in (raw.replace(b"Permission is hereby granted", b"Permission is not granted"), raw.replace(b"SPDX-License-Identifier: MIT", b"SPDX-License-Identifier: GPL-3.0-only"), raw + b"\nNon-commercial use only"):
        with pytest.raises(ValueError): acquisition_v2.review_license(source, license_bytes, changed, "list.c")


def test_old_stricter_acquisition_remains_verifiable():
    report = corpus.read(corpus.AI / "evaluation/reports/llm-task-023-acquisition.json")
    assert acquisition.verify(Path(report["releasePath"]))["trainingAllowed"] is False


def pool(tokens=100):
    return [{"id": domain, "domain": domain, "split": "train", "tokens": tokens, "normalizedSha256": sha(domain)} for domain in acquisition.POLICY["mixtureWeights"]]


@pytest.mark.parametrize("total", [0, 1, 3, 11, 99, 250, 750])
def test_exact_mixture_budget_and_per_document_repetition_cap(total):
    result = mixture.plan(pool(), total)
    assert sum(result["quotas"].values()) == total
    assert result["feasible"]
    assert result["scheduledExposures"] == total
    for domain in acquisition.POLICY["mixtureWeights"]:
        assert sum(row["tokenCount"] for row in result["schedule"] if row["domain"] == domain) == result["quotas"][domain]
        assert len([row for row in result["schedule"] if row["domain"] == domain]) <= 3


def test_impossible_mixture_never_generates_a_training_schedule():
    result = mixture.plan(pool(), 1000000)
    assert not result["feasible"] and result["schedule"] == []
    assert result["actualTrainingExposures"] == 0 and all(result["deficitTokens"].values())


@pytest.mark.parametrize("alteration", ["validation", "duplicate-id", "duplicate-bytes", "negative-tokens", "unregistered-domain"])
def test_mixture_rejects_leakage_and_inflated_unique_counts(alteration):
    rows = pool()
    if alteration == "validation": rows[0]["split"] = "validation"
    elif alteration == "duplicate-id": rows[1]["id"] = rows[0]["id"]
    elif alteration == "duplicate-bytes": rows[1]["normalizedSha256"] = rows[0]["normalizedSha256"]
    elif alteration == "negative-tokens": rows[0]["tokens"] = -1
    else: rows[0]["domain"] = "unregistered"
    with pytest.raises(ValueError): mixture.plan(rows, 100)


@pytest.mark.parametrize("weights", [{"a": 0.3}, {"a": -1, "b": 2}, {"a": "NaN"}, {}])
def test_invalid_weights_fail(weights):
    with pytest.raises(ValueError): mixture.quotas(100, weights)


def test_unverified_approvals_and_measured_budgets_cannot_enable_training():
    rows = [{**row, "sourceId": "unknown", "familyId": row["id"]} for row in pool()]
    result = mixture.release_readiness(rows, [{"sourceId": "unknown", "trainingApproved": True}], {"crossSplitDuplicateEdges": 0, "protectedMatchesInIncluded": 0, "crossSplitLineageKeys": 0}, budget={"approved": True}, pilot_evidence={"passed": True}, gen2_tokenizer={"approved": True})
    assert result["trainingAllowed"] is False
    assert "source-rights-not-admitted:unknown" in result["blockers"]
    assert "measured-scaling-receipt-verifier-not-implemented" in result["blockers"]


@pytest.fixture(scope="module")
def artifact():
    report = corpus.read(corpus.AI / "evaluation/reports/llm-task-023-corpus-build.json")
    return Path(report["releasePath"])


def test_candidate_reproduces_and_preserves_all_old_exclusions(artifact):
    result = corpus.verify(artifact, recompute=True)
    score = result["scorecard"]
    assert score["inheritedQuarantinedRecordsNotReadmitted"] == 50
    assert score["trainingAllowed"] is False and score["releasedGen2TrainingTokens"] == 0
    assert {"language", "code", "math", "electronics"} <= set(score["allCandidateAccounting"]["domain"])


def test_modified_source_or_shard_is_rejected(artifact, tmp_path):
    target = tmp_path / artifact.name
    shutil.copytree(artifact, target)
    (target / "train.jsonl").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="shard changed"): corpus.verify(target)


@pytest.mark.parametrize("use", ["pretraining", "tokenizer-fitting", "sft", "retrieval-evaluation"])
def test_public_use_gate_denies_training(artifact, monkeypatch, use):
    calls = []
    monkeypatch.setattr(corpus, "verify", lambda path, **kw: calls.append(kw) or {})
    with pytest.raises(ValueError, match="RELEASE_GATES_UNMET"): corpus.require_use(artifact, use)
    assert calls == [{"recompute": True}]
