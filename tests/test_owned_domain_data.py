"""Task-022 correctness and failure controls; these are not model evaluations."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from data_governance.splitting.partition import reserve_before_render, require_render_assignment
from data_governance.splitting.signatures import canonical, sha, features
from synthetic_data.foundation_domain import recipes, calculations, firmware, entities, release


@pytest.mark.parametrize("family,unit,parameters", [(family, row[4], p) for family, row in recipes.RECIPES.items() for p in row[5]])
def test_each_generated_answer_has_independent_review_and_negative_controls(family, unit, parameters):
    result = calculations.receipt(family, parameters, recipes.answer(family, parameters), unit)
    assert result["status"] == "passed"
    assert len(result["negativeControlsRejected"]) == 3


@pytest.mark.parametrize("family,expected", [
    ("weighted-summer", -2), ("sensor-bridge", 5 / 42), ("capacitor-decay", 0.016094379124341),
    ("even-parity", 1), ("signed-saturation", 32767), ("rollover-elapsed", 36),
    ("scheduler-load", 0.35), ("uart-frame-time", 1 / 24), ("i2c-clock-budget", 0.00027),
    ("linear-regulator-loss", 0.48), ("ideal-buck-ripple", 7 / 24), ("ideal-adc-bin", 341),
    ("uncorrelated-average", 0.002), ("hysteresis-state", 1), ("proportional-clamp", 0.6),
    ("ring-occupancy", 5), ("buffer-budget", 496),
])
def test_hand_calculated_reference_values(family, expected):
    row = recipes.RECIPES[family]
    assert recipes.answer(family, row[5][0]) == pytest.approx(expected, abs=1e-12)
    assert calculations.oracle(family, row[5][0])[0] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("family,invalid", [
    ("weighted-summer", {"rf": 0}), ("sensor-bridge", {"lb": -1}), ("capacitor-decay", {"vt": 6}),
    ("even-parity", {"payload": 256}), ("signed-saturation", {"a": 32768}), ("rollover-elapsed", {"then": -1}),
    ("scheduler-load", {"c1": 11}), ("uart-frame-time", {"d": 10}), ("i2c-clock-budget", {"hz": 0}),
    ("linear-regulator-loss", {"vin": 4}), ("ideal-buck-ripple", {"l": 0}), ("ideal-adc-bin", {"vin": 3.3}),
    ("uncorrelated-average", {"n": 0}), ("hysteresis-state", {"high": 19}), ("proportional-clamp", {"gain": -1}),
    ("ring-occupancy", {"n": 15}), ("buffer-budget", {"w": 0}),
])
def test_formula_domain_rejects_invalid_physical_or_structural_assumptions(family, invalid):
    p = recipes.RECIPES[family][5][0] | invalid
    with pytest.raises(ValueError): calculations.oracle(family, p)


def test_adc_bins_at_boundaries_and_one_lsb():
    for vin, expected in ((0, 0), (0.25, 1), (0.5, 2), (0.999, 3)):
        p = dict(bits=2, vin=vin, vref=1)
        assert calculations.oracle("ideal-adc-bin", p)[0] == expected


def test_saturation_and_hysteresis_inclusive_edges():
    assert calculations.oracle("signed-saturation", dict(a=-32768, b=0))[0] == -32768
    for x, previous, expected in ((20, 0, 1), (24, 1, 0), (22, 0, 0), (22, 1, 1)):
        assert calculations.oracle("hysteresis-state", dict(x=x, previous=previous, low=20, high=24))[0] == expected


@pytest.mark.parametrize("query", ["ESP32", "Arduino Uno ESP32 R3", "PICO_W", "Arduino Uno R4", "ARDUINO_UNO "])
def test_recipe_compiler_never_substitutes_false_or_unselected_board_alias(query):
    with pytest.raises(ValueError): firmware.exact_board(query)


@pytest.mark.parametrize("kind,query", [("component", "LED"), ("component", "OLED"), ("component", "RELAY"), ("board", "Arduino Uno ESP32 R3")])
def test_generic_entities_do_not_inherit_ratings(kind, query):
    result = entities.evidence(kind, query)
    assert result["status"] == "unknown" and result["genericRatingsKnown"] is False
    if query == "LED": assert all("ssd1306" not in row["recordId"] for row in result["records"])


def test_reservation_keeps_board_and_ring_algorithm_families_together():
    plan = reserve_before_render(release.descriptors())
    splits = {sha(canonical(row["lineage"])): row["split"] for row in plan["assignments"]}
    for family in firmware.PROGRAMS:
        assert len({splits[sha(canonical(release.lineage(family, board)))] for board in firmware.BOARDS}) == 1
    assert splits[sha(canonical(release.lineage("ring-occupancy")))] == splits[sha(canonical(release.lineage("serial-queue-drain", "ARDUINO_UNO")))]
    changed = deepcopy(plan)
    changed["assignments"][0]["split"] = "quarantine"
    with pytest.raises(ValueError): require_render_assignment(changed, plan["lineages"][0], "train")


def example(identity, split, text):
    return {"recordId": identity, "lineage": release.lineage(identity), "reservedSplit": split,
            "question": text, "answer": "", "projectAssumptions": "", "normalizedText": text}


def test_post_render_bridge_quarantines_both_reserved_splits():
    text = "A unique telemetry fixture contains a bounded array cursor transition sequence."
    rows = [example("fixture-a", "train", text), example("fixture-b", "test", text)]
    decisions, _ = release.apply_reserved_splits(rows, [])
    assert {row["split"] for row in decisions} == {"quarantine"}
    assert {row["reason"] for row in decisions} == {"post-render-cross-reservation-bridge"}


def test_injected_protected_content_is_excluded():
    text = "A protected fixture describes a fictional photonic sampling aperture alignment process."
    guard = {"id": "protected-test-control", "keys": [], "features": features([text])}
    decisions, evidence = release.apply_reserved_splits([example("leak-fixture", "train", text)], [guard])
    assert decisions[0]["split"] == "quarantine"
    assert len(evidence["protectedMatches"]) == 1


@pytest.fixture(scope="module")
def artifact():
    report = json.loads((release.AI / "evaluation/reports/llm-task-022-domain-build.json").read_text(encoding="utf-8"))
    path = Path(report["releasePath"])
    read = lambda name: json.loads((path / name).read_text(encoding="utf-8"))
    return path, read("reservation.json"), read("compiler-receipts.json"), read("environment.json")


def test_release_reproduces_and_has_nonvacuous_domain_coverage(artifact):
    result = release.verify(artifact[0], recompute=True)
    coverage = result["coverage"]
    assert set(release.POLICY["requiredDomains"]) <= set(coverage["candidateDomainCounts"])
    assert coverage["domainsWithoutIncludedRecords"] == ["firmware-debug"]
    assert coverage["domainCorpusReady"] is False
    assert coverage["compilerSuccesses"] == coverage["expectedCompilerFailures"] == 12
    assert coverage["includedFamilies"] >= 15
    assert coverage["trainingAllowed"] is False


def test_missing_receipt_and_changed_source_binding_fail_closed(artifact):
    _, reservation, receipts, environment = artifact
    changed = deepcopy(receipts)
    changed.pop(next(iter(changed)))
    with pytest.raises(ValueError, match="compiler cases"): release.render(reservation["reservation"], changed, environment)
    changed = deepcopy(receipts)
    changed[next(iter(changed))]["receipt"]["sourceSha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="bound to source"): release.render(reservation["reservation"], changed, environment)
    changed = deepcopy(receipts)
    changed[next(iter(changed))]["environmentSha256"] = "0" * 64
    with pytest.raises(ValueError, match="environment"): release.render(reservation["reservation"], changed, environment)


def test_mutated_shard_is_rejected_and_write_once_never_overwrites(artifact, tmp_path):
    target = tmp_path / artifact[0].name
    shutil.copytree(artifact[0], target)
    (target / "train.jsonl").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bytes changed"): release.verify(target)
    with pytest.raises(ValueError, match="Immutable output collision"): release.write_immutable(target / "train.jsonl", b"replacement")


@pytest.mark.parametrize("use", ["tokenizer-fitting", "pretraining", "sft", "tuning", "retrieval-evaluation"])
def test_candidate_release_does_not_authorize_training(artifact, monkeypatch, use):
    # Full verify is covered above; here ensure the public use gate invokes it.
    calls = []
    monkeypatch.setattr(release, "verify", lambda path, **kw: calls.append(kw) or {})
    with pytest.raises(ValueError, match="TRAINING_CORPUS_RELEASE_REQUIRED"): release.require_use(artifact[0], use)
    assert calls == [{"recompute": True}]
