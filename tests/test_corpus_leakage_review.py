"""Independent precision labels, bounded diagnostics and preservation regressions."""
import copy
import json
import random
from pathlib import Path

import pytest

from data_governance.leakage_review import controls, evaluation, review
from data_governance.pretraining import indexed
from data_governance.splitting.partition import lineage_keys
from data_governance.splitting.signatures import features


def test_authored_precision_matrix_exposes_false_positives_without_losing_positives():
    result = evaluation.control_report()
    assert result["cases"] == 56 and result["independentFamilies"] == 10
    assert result["activeMatcherConfusionOnAuthoredControls"] == {
        "truePositive": 32, "falseNegative": 0, "falsePositive": 16, "trueNegative": 8,
    }
    assert result["classificationCounts"] == {"identity-literal-code": 30, "lexical-review-required": 18, "no-match": 8}
    assert not any(row["admissionAllowed"] or row["existingDecisionChanged"] for row in result["rows"])


@pytest.mark.parametrize("index", range(8))
def test_negative_glossary_is_not_automatically_readmitted(index):
    case = next(row for row in controls.cases() if row["id"] == f"scene-{index}-wide-glossary")
    result = review.explain(case["document"], case["guard"])
    assert result["classification"] == "lexical-review-required"
    assert result["existingLeakageBlock"] is True and result["reviewComplete"] is False
    assert result["admissionAllowed"] is False
    assert all(row["protectedWordCoverage"] >= .9 for row in result["lexicalObservations"])
    assert any(row["minimumCoveringWordWindow"] > row["sharedDistinctWords"] for row in result["lexicalObservations"])


@pytest.mark.parametrize("index", range(2))
def test_real_paraphrase_controls_remain_blocked_for_review(index):
    case = next(row for row in controls.cases() if row["id"] == f"paraphrase-{index}")
    result = review.explain(case["document"], case["guard"])
    assert result["classification"] == "lexical-review-required"
    assert result["existingLeakageBlock"] is True and result["admissionAllowed"] is False


def test_family_identity_has_precedence_over_lexical_diagnostics():
    document = controls.document("family-a", controls.UNRELATED, family="shared-repository")
    guard = controls.guard("guard-a", controls.SCENES[0])
    guard["keys"] = sorted(lineage_keys(document["lineage"]))
    result = review.explain(document, guard)
    assert result["activeMatcherReason"] == "protected-lineage-family"
    assert result["classification"] == "identity-literal-code"
    assert result["lexicalObservations"] == []


def test_reports_do_not_export_protected_text_words_or_positions():
    for case in controls.cases():
        result = review.explain(case["document"], case["guard"])
        raw = json.dumps(result, ensure_ascii=False)
        assert "features" not in result and "literal" not in result and "text" not in result
        for literal in case["guard"]["features"]["literal"]:
            assert literal not in raw
        for item in result["lexicalObservations"]:
            assert "start" not in item and "end" not in item


def test_no_match_does_not_establish_source_approval():
    result = review.explain(controls.document("d", controls.UNRELATED), controls.guard("g", controls.SCENES[0]))
    assert result["classification"] == "no-match" and result["existingLeakageBlock"] is False
    assert result["admissionAllowed"] is False


def test_unknown_future_reason_fails_closed(monkeypatch):
    monkeypatch.setattr(indexed, "protected_reason", lambda *_: "new-unreviewed-signal")
    with pytest.raises(ValueError, match="Unreviewed matcher reason"):
        review.explain(controls.document("d", "text"), controls.guard("g", "text"))


def test_diagnostics_leave_input_features_and_matcher_policy_unchanged():
    case = next(row for row in controls.cases() if row["id"] == "scene-0-glossary")
    original = copy.deepcopy((case, indexed.MATCH_POLICY))
    review.explain(case["document"], case["guard"])
    assert (case, indexed.MATCH_POLICY) == original


def test_sliding_word_window_matches_independent_brute_force():
    rng = random.Random(23024)
    for _ in range(200):
        sequence = [rng.choice("abcdef") for _ in range(rng.randrange(20))]
        required = set(rng.sample(list("abcdef"), rng.randrange(1, 5)))
        windows = [end - start for start in range(len(sequence)) for end in range(start + 1, len(sequence) + 1) if required <= set(sequence[start:end])]
        assert review.shortest_cover(sequence, required) == (min(windows) if windows else None)
    assert review.shortest_cover([], set()) is None


@pytest.mark.parametrize("field", [
    "maximumLexicalPairsPerMatch", "maximumLiteralCharactersPerMatch", "maximumWordTokensPerMatch",
    "maximumBagWordVisitsPerMatch", "maximumWindowTokenVisitsPerMatch",
])
def test_diagnostic_resource_limits_never_turn_into_an_allow(field, monkeypatch):
    case = next(row for row in controls.cases() if row["id"] == "scene-0-glossary")
    monkeypatch.setitem(review.POLICY, field, 0)
    with pytest.raises(ValueError, match="budget exceeded"):
        review.explain(case["document"], case["guard"])


def test_index_review_keeps_exact_first_guard_and_duplicate_edges():
    selected = [row for row in controls.cases() if row["id"] in {"scene-0-glossary", "scene-0-exact", "scene-1-exact"}]
    documents = [row["document"] for row in selected]
    guards = list({row["guard"]["id"]: row["guard"] for row in selected}.values())
    expected = indexed.scan(documents, guards)
    result = review.review_scan(documents, guards)
    matches = [{"documentId": row["documentId"], "guardId": row["guardId"], "reason": row["activeMatcherReason"]} for row in result["matches"]]
    assert matches == expected["protectedMatches"]
    assert result["duplicateMatches"] == expected["duplicateMatches"]
    assert result["existingDecisionsChanged"] == 0 and result["automaticReadmissions"] == 0


def test_match_budget_fails_closed(monkeypatch):
    monkeypatch.setitem(review.POLICY, "maximumReviewMatches", 0)
    with pytest.raises(ValueError, match="match budget exceeded"):
        review.review_scan([controls.document("d", controls.SCENES[0])], [controls.guard("g", controls.SCENES[0])])


@pytest.fixture(scope="module")
def retained_review():
    paths = sorted(evaluation.ARTIFACTS.glob("*/manifest.json"))
    assert len(paths) == 1, "Build the immutable task-023 precision review before this integration check"
    return paths[0].parent


def test_real_review_and_partition_preservation_reproduce(retained_review):
    evaluation.verify(retained_review, recompute=True)
    real = json.loads((retained_review / "real-review.json").read_text(encoding="utf-8"))
    assert real["documents"] == 38 and real["protectedGuards"] == 1005
    assert real["partitionDecisionsUnchanged"] is True
    assert real["quarantinedDocuments"] == 7
    assert real["quarantineFromFamilyPropagation"] == 5
    assert real["splitCounts"] == {"train": 26, "validation": 0, "test": 5, "quarantine": 7}
    assert real["review"]["classificationCounts"] == {"lexical-review-required": 2}
    assert real["trainingAllowed"] is False and real["automaticReadmissions"] == 0


def test_tampered_review_bytes_are_rejected(retained_review, tmp_path):
    copied = tmp_path / retained_review.name
    copied.mkdir()
    for name in ("manifest.json", "controls.json", "real-review.json"):
        (copied / name).write_bytes((retained_review / name).read_bytes())
    (copied / "controls.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="evidence bytes changed"):
        evaluation.verify(copied)
