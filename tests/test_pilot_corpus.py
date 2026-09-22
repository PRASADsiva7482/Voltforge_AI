"""Input-release acceptance and meaningful fail-closed regression controls."""
from copy import deepcopy
import json
from pathlib import Path
import pytest

from data_governance.pilot_corpus import release as r, supplement
from data_governance.pretraining import indexed, mixture


@pytest.fixture(scope="module")
def prepared():
    payloads, plan, score = r.compute()
    rows = [json.loads(line) for split in r.SPLITS for line in payloads[split + ".jsonl"].decode("utf-8").splitlines()]
    return payloads, plan, score, rows


def gate(prepared, *, rows=None, rights=None, audit=None):
    payloads = prepared[0]
    return r.quality(rows if rows is not None else prepared[3], rights if rights is not None else json.loads(payloads["sources.json"]), audit if audit is not None else json.loads(payloads["leakage.json"])["audit"])


def test_all_domains_have_independent_input_coverage(prepared):
    score = prepared[2]
    assert score["trainingUniqueProxyTokens"] == 919275
    assert score["splitCounts"]["train"] == 42
    assert score["splitCounts"]["validation"] >= 9
    assert score["independentReferenceChecks"] == 4
    assert score["protectedDescriptors"] == 1005
    for domain in ("language", "code", "math", "electronics"):
        assert score["coverage"][domain]["train"]["families"] >= 2
        assert score["coverage"][domain]["validation"]["families"] >= 1
    assert score["actualTrainingExposures"] == score["releasedGen2TrainingTokens"] == 0


def test_prior_exclusions_and_reserved_roles_survive(prepared):
    previous = {row["id"]: row for row in r.rows_at(r.PRIOR)}
    current = {row["id"]: row for row in prepared[3]}
    assert len(json.loads(prepared[0]["inherited-exclusions.json"])) == 50
    for identity, old in previous.items():
        assert current[identity]["split"] == old["split"]
        assert current[identity]["text"] == old["text"]
        assert current[identity]["tokens"] == old["tokens"]
    assert sum(row["split"] == "quarantine" for row in previous.values()) == 8


def test_input_budget_and_repetition_schedule_are_exact(prepared):
    _, plan = gate(prepared)
    assert plan["quotas"] == {"language": 8000, "code": 6000, "electronics": 4000, "math": 2000}
    assert plan["scheduledExposures"] == 20000
    train = {row["id"]: row for row in prepared[3] if row["split"] == "train"}
    seen = set()
    for item in plan["schedule"]:
        assert (item["documentId"], item["pass"]) not in seen
        seen.add((item["documentId"], item["pass"]))
        assert 1 <= item["pass"] <= 3
        assert item["tokenCount"] <= train[item["documentId"]]["tokens"]
    assert plan["actualTrainingExposures"] == 0


@pytest.mark.parametrize("field", ["rawSha256", "normalizedSha256", "sourceId", "reservedSplit", "allowedUses"])
def test_changed_exact_source_permission_is_denied(prepared, field):
    rights = json.loads(prepared[0]["sources.json"])
    used = next(row for row in prepared[3] if row["split"] == "train")
    permission = next(row for row in rights["documents"] if row["documentId"] == used["id"])
    permission[field] = [] if field == "allowedUses" else "unapproved"
    with pytest.raises(ValueError, match="source rights|reserved role"):
        gate(prepared, rights=rights)


def test_missing_rights_and_duplicate_approvals_are_denied(prepared):
    rights = json.loads(prepared[0]["sources.json"])
    used = next(row for row in prepared[3] if row["split"] == "train")
    rights["documents"] = [row for row in rights["documents"] if row["documentId"] != used["id"]]
    with pytest.raises(ValueError, match="source rights"):
        gate(prepared, rights=rights)
    rights["documents"].append(rights["documents"][0])
    with pytest.raises(ValueError, match="Duplicate source permission"):
        gate(prepared, rights=rights)


@pytest.mark.parametrize("field", ["decision", "inputBindings", "modelReleaseApproved", "trainingRunApproved"])
def test_source_ledger_cannot_claim_new_scope(prepared, monkeypatch, field):
    decision = r.read(r.DECISION)
    decision[field] = [] if field == "inputBindings" else "unapproved"
    original = r.read
    monkeypatch.setattr(r, "read", lambda path: decision if Path(path) == r.DECISION else original(path))
    with pytest.raises(ValueError, match="decision binding or scope"):
        r.verify_rights(prepared[3])


@pytest.mark.parametrize("key", ["crossSplitDuplicateEdges", "protectedMatchesInIncluded", "crossSplitLineageKeys"])
def test_dirty_or_missing_leakage_audit_is_denied(prepared, key):
    audit = json.loads(prepared[0]["leakage.json"])["audit"]
    audit[key] = 1
    with pytest.raises(ValueError, match="contamination"):
        gate(prepared, audit=audit)
    del audit[key]
    with pytest.raises(ValueError, match="contamination"):
        gate(prepared, audit=audit)


@pytest.mark.parametrize("mutation", ["text", "split", "inheritedExclusion"])
def test_input_change_cannot_be_admitted(prepared, mutation):
    rows = deepcopy(prepared[3])
    row = next(row for row in rows if row["split"] == "train")
    row[mutation] = {"text": row["text"] + " Changed text.", "split": "validation", "inheritedExclusion": True}[mutation]
    with pytest.raises(ValueError):
        gate(prepared, rows=rows)


def test_quarantined_python_cannot_gain_input_permission(prepared):
    rows = deepcopy(prepared[3])
    row = next(row for row in rows if row["split"] == "quarantine")
    row["split"] = row["reservedSplit"] = "train"
    row["inheritedExclusion"] = False
    with pytest.raises(ValueError, match="source rights"):
        gate(prepared, rows=rows)


def test_duplicate_capacity_cannot_inflate_budget(prepared):
    rows = deepcopy(prepared[3])
    rows.append(deepcopy(next(row for row in rows if row["split"] == "train")))
    with pytest.raises(ValueError, match="Duplicate corpus identity"):
        gate(prepared, rows=rows)
    training = [row for row in prepared[3] if row["split"] == "train"]
    duplicate = {**training[0], "id": "renamed-duplicate"}
    with pytest.raises(ValueError, match="Duplicate records"):
        mixture.plan(training + [duplicate], 20000)


def test_excessive_passes_and_infeasible_exposures_are_denied(prepared):
    training = [row for row in prepared[3] if row["split"] == "train"]
    with pytest.raises(ValueError, match="Repetition cap"):
        mixture.plan(training, 20000, maximum_passes=4)
    oversized = mixture.plan(training, 2000000000)
    assert not oversized["feasible"] and oversized["scheduledExposures"] == 0 and oversized["schedule"] == []


@pytest.mark.parametrize("limit", ["maximumDocuments", "maximumNormalizedCorpusBytes", "maximumNormalizedDocumentBytes"])
def test_resource_bounds_fail_closed(prepared, monkeypatch, limit):
    monkeypatch.setitem(r.POLICY, limit, 1)
    with pytest.raises(ValueError, match="budget exceeded"):
        r.bounded_rows(prepared[3])


@pytest.mark.parametrize("domain", ["language", "code", "electronics", "math"])
def test_missing_validation_domain_fails_coverage(prepared, domain):
    rows = [row for row in prepared[3] if not (row["domain"] == domain and row["split"] == "validation")]
    with pytest.raises(ValueError, match="Insufficient independent validation"):
        gate(prepared, rows=rows)


@pytest.mark.parametrize("index", range(4))
def test_independent_reference_check_rejects_wrong_answer(index):
    row = supplement.render()[index]
    supplement.verify_example(row)
    for key in row["values"]:
        changed = deepcopy(row)
        changed["values"][key] = "999"
        with pytest.raises(ValueError):
            supplement.verify_example(changed)


def test_reference_families_are_reserved_without_guard_access():
    rows, checks = r.owned_rows()
    assert len({row["familyId"] for row in rows}) == 4
    assert all(row["reservedSplit"] == "validation" for row in rows)
    assert len(checks) == 4
    assert len({row["normalizedSha256"] for row in rows}) == 4


def test_relative_input_paths_bind_to_same_bytes(monkeypatch):
    monkeypatch.chdir(r.AI)
    assert r.binding(Path("data_governance/pilot_corpus/policy.v1.json")) == r.binding(r.ROOT / "policy.v1.json")


def test_independent_known_content_contamination_stays_quarantined():
    row = r.owned_rows()[0][0]
    description = r.corpus.describe(row)
    guard = {**description, "id": "independent-control-guard", "keys": []}
    decisions, _ = indexed.partition([description], [guard])
    assert decisions[0]["split"] == "quarantine"


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/absolute", "nested\\file", ""])
def test_unsafe_paths_fail_closed(tmp_path, name):
    with pytest.raises(ValueError, match="Unsafe"):
        r.safe_file(tmp_path, name)


@pytest.mark.parametrize("usage,split", [("production-training", "train"), ("runtime-activation", "train"), ("tokenizer-fitting-input", "validation"), ("bounded-pilot-pretraining-input", "test"), ("validation-input", "train"), ("test-input", "train"), ("corpus-review", "unknown")])
def test_unapproved_usage_is_denied_before_reading(tmp_path, usage, split):
    with pytest.raises(ValueError, match="PILOT_INPUT_USE_NOT_APPROVED|RESERVED_SPLIT_USE_DENIED"):
        r.require_use(tmp_path, usage, split=split)


def test_revocation_blocks_all_inputs_and_malformed_ledger_fails(monkeypatch):
    source_id = r.read(r.DECISION)["documents"][0]["sourceId"]
    original = r.read
    for entry in ({"kind": "source", "id": source_id, "reason": "source removal"}, {"kind": "corpus", "id": "release", "reason": "withdrawn"}, {"kind": "unknown", "id": "release", "reason": "invalid"}):
        monkeypatch.setattr(r, "read", lambda path: {"schemaVersion": 1, "entries": [entry]} if Path(path) == r.REVOCATIONS else original(path))
        with pytest.raises(ValueError, match="REVOKED|Invalid revocation"):
            r.check_revocations({"contentId": "release"})


@pytest.fixture(scope="module")
def published():
    manifests = list(r.OUTPUT.glob("*/manifest.json"))
    assert len(manifests) == 1, "Exactly one accepted v1 pilot corpus must be selected"
    return manifests[0].parent


def test_published_release_reproduces(prepared, published):
    manifest = r.verify(published, recompute=True)
    for name, raw in prepared[0].items():
        assert (published / name).read_bytes() == raw
    assert manifest["inputUseAllowed"] and not manifest["trainingRunApproved"]
    assert not manifest["productionTrainingAllowed"] and not manifest["modelReleaseApproved"]
    assert set(item["path"] for item in manifest["files"]) == r.FILE_NAMES


def test_published_fitting_handoff_reads_only_verified_train(prepared, published):
    rows = r.read_inputs(published, "tokenizer-fitting-input", split="train")
    assert len(rows) == 42
    assert sum(row["tokens"] for row in rows) == 919275
    assert all(row["split"] == "train" and not row["inheritedExclusion"] for row in rows)
    assert not {row["id"] for row in rows} & {row["id"] for row in prepared[3] if row["split"] in ("validation", "test", "quarantine")}


def test_published_pilot_handoff_enforces_exposure_cap(published):
    spans = r.read_inputs(published, "bounded-pilot-pretraining-input", split="train")
    assert sum(len(item["proxyTokenIds"]) for item in spans) == 20000
    assert all(item["pass"] <= 3 and len(item["proxyTokenIds"]) == item["tokenCount"] for item in spans)
    assert all(item["trainingRunApproved"] is False for item in spans)
    assert {domain: sum(item["tokenCount"] for item in spans if item["domain"] == domain) for domain in r.POLICY["mixtureWeights"]} == {"language": 8000, "code": 6000, "electronics": 4000, "math": 2000}


def test_published_old_candidate_stays_denied(published):
    from model.gen2_tokenizer.release import require_corpus_fitting
    with pytest.raises(ValueError):
        r.require_use(r.PRIOR, "tokenizer-fitting-input", split="train")
    # The frozen task-024 fixture adapter is intentionally not rewritten here.
    with pytest.raises(ValueError):
        require_corpus_fitting(published)


def test_published_corrupt_shard_is_rejected_before_handoff(published, tmp_path, monkeypatch):
    import shutil
    output = tmp_path / "pilot-input"
    target = output / published.name
    shutil.copytree(published, target)
    monkeypatch.setattr(r, "OUTPUT", output)
    with (target / "train.jsonl").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="shard changed"):
        r.read_inputs(target, "tokenizer-fitting-input", split="train")


@pytest.mark.parametrize("field,value", [("elapsedSeconds", 301), ("elapsedSeconds", 0), ("peakResidentBytes", 2147483649), ("peakResidentBytes", 0), ("freshProcess", False), ("planSha256", "wrong")])
def test_published_resource_mismatch_is_rejected(published, tmp_path, monkeypatch, field, value):
    manifest = r.read(published / "manifest.json")
    plan = {key: value for key, value in manifest.items() if key not in ("contentId", "resourceMeasurement")}
    receipt = r.read(r.AI / manifest["resourceMeasurement"]["path"])
    receipt[field] = value
    receipt.pop("contentId")
    receipt["contentId"] = r.sha(r.canonical(receipt))
    target = tmp_path / receipt["contentId"] / "measurement.json"
    target.parent.mkdir()
    target.write_bytes(r.data(receipt))
    monkeypatch.setattr(r, "RESOURCE_OUTPUT", tmp_path)
    with pytest.raises(ValueError, match="not bound|budget exceeded"):
        r.verify_resources(target, plan)
