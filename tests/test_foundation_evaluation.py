"""Independent evaluation must reject contamination and fabricated neural credit."""
from copy import deepcopy
import base64
import json
import math
from pathlib import Path
import shutil
from unittest.mock import patch

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from evaluation.foundation import suite, grading, harness
from evaluation.foundation.observer import InvocationLedger


@pytest.fixture(scope="module")
def frozen_copy(tmp_path_factory):
    root = tmp_path_factory.mktemp("foundation-suite")
    for relative in suite.SOURCES:
        shutil.copyfile(suite.ROOT / relative, root / relative)
    previous = suite.ROOT
    try:
        suite.ROOT = root
        suite.freeze()
    finally:
        suite.ROOT = previous
    return root


@pytest.fixture
def frozen(frozen_copy, monkeypatch):
    monkeypatch.setattr(suite, "ROOT", frozen_copy)
    monkeypatch.setattr(harness, "ROOT", frozen_copy)
    return suite.build_cases(), suite.read(frozen_copy / "acceptance-policy.v1.json"), suite.verify_suite()


def signed(payload, private, key_id):
    return {"payload": payload, "keyId": key_id, "signature": base64.b64encode(private.sign(suite.canonical(payload).encode())).decode()}


def reviewer(key_id, role="expert"):
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return private, {"publicKeyBase64": base64.b64encode(public).decode(), "roles": [role], "reviewerId": key_id}


def exact_case(frozen):
    return next(case for case in frozen[0]["acceptance"] if case["oracle"]["kind"] == "numeric-json")


def grade(frozen, case, text, reviews=(), trust=None):
    request = harness.request_for(case, "raw_model", 17018)
    return grading.grade(case, {"text": text}, request, policy=frozen[1], suite_sha=frozen[2]["suiteSha256"], policy_sha=frozen[2]["policySha256"], reviews=reviews, trust=trust)


def test_real_cohort_counts_and_independence_units(frozen):
    report = frozen[2]
    assert report["counts"] == {"development": 100, "validation": 100, "acceptance": 500, "adversarial": 100, "compile": 100}
    assert set(report["acceptanceCategories"].values()) == {50}
    assert report["coreScenarioFamilies"] == 100
    assert report["coreTemplateFamilies"] == 73
    assert report["trainingUseAllowed"] is False
    assert report["confidentialSealingClaimed"] is False


def test_freeze_never_overwrites_history(frozen):
    with pytest.raises(ValueError, match="never overwrite"):
        suite.freeze()


def test_altered_case_cannot_borrow_a_frozen_identity(frozen):
    case = deepcopy(exact_case(frozen))
    case["oracle"]["expected"]["value"] = 42
    with pytest.raises(ValueError, match="frozen fingerprint"):
        harness.evaluate([case], lambda _: {"text": "42"}, lane="rule_only", seed=17018)


def test_frozen_fixture_tampering_is_detected(frozen):
    path = suite.ROOT / "fixtures/v1/acceptance.jsonl"
    saved = path.read_bytes()
    try:
        path.write_bytes(saved + b"\n")
        with pytest.raises(ValueError, match="hash mismatch"):
            suite.verify_suite()
    finally:
        path.write_bytes(saved)


@pytest.mark.parametrize("field", ["templateFamilyId", "sourceFamilyId", "scenarioText"])
def test_split_relabelling_does_not_legalize_a_related_case(frozen, field):
    splits = deepcopy(frozen[0])
    splits["development"][0][field] = splits["acceptance"][0][field]
    with pytest.raises(ValueError, match="Cross-split"):
        suite.validate_isolation(splits)


@pytest.mark.parametrize("kind", ["family", "source", "renamed-text", "changed-numbers", "oracle"])
def test_training_leakage_gate_rejects_evaluation_material(frozen, kind):
    case = exact_case(frozen)
    if kind == "family":
        row = {"templateFamilyId": case["templateFamilyId"], "text": "unrelated"}
    elif kind == "source":
        row = {"sourceFamilyId": case["sourceFamilyId"], "text": "unrelated"}
    elif kind == "oracle":
        row = {"text": suite.canonical(case["oracle"])}
    else:
        text = case["scenarioText"].replace("7", "918") if kind == "changed-numbers" else case["scenarioText"]
        row = {"sourceFamilyId": "renamed-new-source", "text": text}
    with pytest.raises(ValueError, match="FOUNDATION_EVALUATION_LEAKAGE"):
        suite.check_training_candidates([row])


def test_unrelated_training_text_is_not_denied(frozen):
    assert suite.check_training_candidates([{"text": "The synthetic workshop inventory contains copper fasteners and cardboard cartons."}])["recordsChecked"] == 1


def test_numerical_oracle_matches_independent_values():
    assert suite.numerical_oracle("ohm-current", 12, 240, 0, 0)["expected"] == {"value": 0.05, "unit": "A"}
    assert suite.numerical_oracle("rc-cutoff", 1000, 1, 0, 1e-6)["expected"]["value"] == pytest.approx(159.15494309189535)
    assert suite.numerical_oracle("timer-astable", 1000, 2000, 0, 1e-6)["expected"]["value"] == pytest.approx(288.53900817779265)


def test_correct_numeric_answer_passes_exact_grading_but_not_neural_admission(frozen):
    case = exact_case(frozen)
    answer = json.dumps(case["oracle"]["expected"])
    assert grade(frozen, case, answer)["status"] == "pass"
    result = harness.evaluate([case], lambda _: {"text": answer, "forwardCalls": 999, "generatedTokens": 999}, lane="raw_model", seed=17018)
    assert result["decision"] == "fail"
    assert result["observedForwardCalls"] == 0
    assert result["rows"][0]["reason"] == "unobserved-or-ineligible-neural-generation"


@pytest.mark.parametrize("answer", ["NaN", '{"value":NaN,"unit":"A"}', '{"value":true,"unit":"A"}', '{"value":1,"unit":"A","approved":true}', '{"value":1,"value":2,"unit":"A"}', "This uses Ohm's law, so it is correct."])
def test_nonfinite_boolean_extra_keys_and_keyword_answers_do_not_pass(frozen, answer):
    assert grade(frozen, exact_case(frozen), answer)["status"] == "fail"


def test_changed_numeric_values_require_a_changed_answer(frozen):
    first = exact_case(frozen)
    second = next(case for case in frozen[0]["acceptance"] if case["scenarioFamilyId"] == first["scenarioFamilyId"] and case["variant"] == 1)
    assert grade(frozen, second, json.dumps(first["oracle"]["expected"]))["status"] == "fail"


def test_prompt_echo_and_constants_cannot_earn_quality_credit(frozen):
    cases = frozen[0]["acceptance"]
    echoes = harness.evaluate(cases, lambda request: {"text": request["message"]}, lane="rule_only", seed=17018)
    assert echoes["corePassedFractionIncludingPending"] == 0
    constants = harness.evaluate(cases, lambda _: {"text": "A resistor is part of a circuit."}, lane="rule_only", seed=17018)
    assert constants["collapseCases"] == 500
    assert "answer-collapse" in constants["failedMetrics"]


def test_requests_never_expose_oracles_or_expert_criteria(frozen):
    for case in frozen[0]["acceptance"]:
        request = harness.request_for(case, "raw_model", 17018)
        assert "oracle" not in request and "criterion" not in request and "expected" not in request
        assert set(request) == {"caseId", "lane", "seed", "message", "context", "history", "sessionId"}


def test_multi_turn_is_sequential_and_session_scoped(frozen):
    case = next(case for case in frozen[0]["acceptance"] if case["category"] == "multi_turn")
    calls = []
    def adapter(request):
        calls.append(deepcopy(request))
        return {"text": "acknowledged"}
    harness.evaluate([case], adapter, lane="rule_only", seed=17018)
    assert len(calls) == 3
    assert [len(call["history"]) for call in calls] == [0, 2, 4]
    assert len({call["sessionId"] for call in calls}) == 1


def test_missing_or_duplicate_results_cannot_shrink_denominator(frozen):
    cases = [exact_case(frozen)]
    with pytest.raises(ValueError, match="denominator"):
        harness.score([], cases=cases, lane="raw_model", seed=17018, policy=frozen[1], suite=frozen[2])


def test_intervals_resample_families_not_correlated_variants(frozen):
    rows = [{"templateFamilyId": family, "status": status} for family, status in (("a", "pass"), ("b", "fail")) for _ in range(5)]
    interval = harness.family_interval(rows, frozen[1])
    assert interval["independentTemplateFamilies"] == 2
    assert interval["lower"] == 0 and interval["upper"] == 1


def test_open_ended_grading_requires_two_bound_independent_signed_reviews(frozen):
    case = next(case for case in frozen[0]["acceptance"] if case["oracle"]["kind"] == "expert")
    text = "A reviewed synthetic answer about the requested subject."
    assert grade(frozen, case, text)["status"] == "pending"
    keys = {name: reviewer(name) for name in ("reviewer-a", "reviewer-b")}
    payload = {"caseId": case["id"], "caseSha256": suite.sha(suite.canonical(case).encode()), "outputSha256": suite.sha(text.encode()),
               "policySha256": frozen[2]["policySha256"], "suiteSha256": frozen[2]["suiteSha256"], "lane": "raw_model", "seed": 17018,
               "scores": {key: 2 for key in frozen[1]["grading"]["expertRubric"]["dimensions"]}}
    reviews = [signed({**payload, "reviewerId": name}, private, name) for name, (private, _) in keys.items()]
    trust = {name: key for name, (_, key) in keys.items()}
    assert grade(frozen, case, text, reviews, trust)["status"] == "pass"
    assert grade(frozen, case, text + " altered", reviews, trust)["status"] == "pending"
    assert grade(frozen, case, text, [reviews[0], reviews[0]], trust)["status"] == "pending"
    assert grade(frozen, case, text, reviews, {})["status"] == "pending"
    changed = {**payload, "reviewerId": "reviewer-b", "scores": {**payload["scores"], "correctness": 1}}
    disagreement = [reviews[0], signed(changed, keys["reviewer-b"][0], "reviewer-b")]
    assert grade(frozen, case, text, disagreement, trust)["status"] == "pending"
    same_key_trust = {**trust, "reviewer-b": {**trust["reviewer-a"], "reviewerId": "reviewer-b"}}
    same_key_reviews = [reviews[0], signed({**payload, "reviewerId": "reviewer-b"}, keys["reviewer-a"][0], "reviewer-b")]
    assert grade(frozen, case, text, same_key_reviews, same_key_trust)["status"] == "pending"


def test_compile_receipt_binds_source_exact_board_and_toolchain(frozen):
    case = frozen[0]["compile"][0]
    source = "void setup() {}\nvoid loop() {}\n"
    text = "```cpp\n" + source + "```"
    private, key = reviewer("compiler", "compiler")
    bindings = {"caseId": case["id"]}
    payload = {**bindings, "sourceSha256": suite.sha(source.encode()), "fqbn": case["oracle"]["fqbn"],
               "toolchainManifestSha256": case["oracle"]["toolchainManifestSha256"], "compilerProfile": case["context"]["compilerProfile"],
               "coreVersion": case["context"]["coreVersion"], "extraLibraries": [], "exitCode": 0}
    receipt = signed(payload, private, "compiler")
    assert grading.compile_grade(case, text, receipt, {"compiler": key}, bindings)["status"] == "pass"
    other = deepcopy(case)
    other["oracle"]["fqbn"] = "wrong:board:tuple"
    assert grading.compile_grade(other, text, receipt, {"compiler": key}, bindings)["status"] == "pending"
    assert grading.compile_grade(case, text + "fabricated success", None, {}, bindings)["status"] == "pending"


def test_actual_random_decoder_probe_has_no_release_credit():
    ledger = InvocationLedger()
    request = {"caseId": "probe", "message": "synthetic", "seed": 17018}
    response = ledger.random_probe(request)
    evidence = ledger.inspect(response, request)
    assert evidence["observed"] and evidence["forwardCalls"] == 3 and evidence["generatedTokens"] == 3
    assert evidence["logitsElements"] == 96
    assert evidence["releaseEligible"] is False
    assert ledger.inspect({**response, "text": "rule substitution"}, request)["observed"] is False
    assert ledger.inspect(response, {**request, "caseId": "replay"})["observed"] is False
    assert ledger.inspect({"text": response["text"], "witness": vars(response["witness"])}, request)["observed"] is False
    assert InvocationLedger().inspect(response, request)["observed"] is False


def test_candidate_admission_keeps_checkpoint_security_block():
    ledger = InvocationLedger()
    with patch("model.runtime_service.checkpoint_runtime_security_block", return_value=("SECURITY_BLOCK", "blocked")):
        with pytest.raises(ValueError, match="SECURITY_BLOCK"):
            ledger.candidate_adapter(object(), object(), {}, {})


def test_model_unavailable_is_blocked_with_zero_invocations(frozen):
    result = harness.evaluate([exact_case(frozen)], lambda _: {"text": "", "unavailable": "NO_APPROVED_MODEL_ARTIFACT"}, lane="raw_model", seed=17018)
    assert result["decision"] == "blocked"
    assert result["observedForwardCalls"] == result["observedGeneratedTokens"] == 0
    assert result["releaseApproved"] is False


def test_seed_coverage_and_signed_measurements_cannot_be_skipped(frozen):
    with pytest.raises(ValueError, match="All frozen"):
        harness.finalize([], None, {})
    reports = [{"seed": seed, "lane": "raw_model", "decision": "pass-for-this-seed", "suiteSha256": frozen[2]["suiteSha256"], "policySha256": frozen[2]["policySha256"]} for seed in frozen[1]["seeds"]["generation"]]
    assert harness.finalize(reports, {"selfGraded": True}, {})["decision"] == "blocked"


def test_full_current_baseline_has_no_neural_credit(frozen, monkeypatch):
    from evaluation.foundation import baseline
    monkeypatch.setattr(baseline, "ROOT", suite.ROOT)
    result = baseline.run_baseline()
    assert result["observedNeuralForwardCalls"] == result["observedModelTokens"] == 0
    assert len(result["scorecards"]) == 12
    assert all(row["fullCohort"] and len(row["rows"]) == 700 for row in result["scorecards"])
    assert all(row["decision"] == "blocked" for row in result["scorecards"] if row["lane"] != "rule_only")
    assert result["regressions"]["changedValues"]["inverseFrequencyRelationshipPassed"]
    assert result["inProcessSyntheticProbes"]["blocked_socket_attempts"] == 0


def test_signed_aggregate_requires_real_denominators_and_server_measurements(frozen):
    reports = [{"seed": seed, "lane": "raw_model", "decision": "pass-for-this-seed", "fullCohort": True,
                "observedArtifactIds": ["vfdlm-g2-core-v0.1.0"], "suiteSha256": frozen[2]["suiteSha256"],
                "policySha256": frozen[2]["policySha256"]} for seed in frozen[1]["seeds"]["generation"]]
    private, key = reviewer("independent-evaluator", "evaluator")
    payload = {"suiteSha256": frozen[2]["suiteSha256"], "policySha256": frozen[2]["policySha256"],
               "scorecardsSha256": suite.sha(suite.canonical(reports).encode()), "lane": "raw_model",
               "metricCounts": {name: {"correct": 50, "denominator": 50} for name in ("citationCorrectness", "supportedClaimAccuracy", "retrievalRecallAt5")},
               "citationCorrectness": 1, "supportedClaimAccuracy": 1, "retrievalRecallAt5": 1,
               "browserScenarios": 25, "modelTokensPerSecond": 20, "p95FirstModelTokenSeconds": 3,
               "cancellationSeconds": 2, "criticalFailures": 0, "inputTokens": 2048, "outputTokens": 256,
               "concurrency": [1, 4, 8], "concurrencyMeasurements": [{"concurrency": n, "requests": n * 10,
                   "p95FirstModelTokenSeconds": 3, "modelTokensPerSecond": 20, "cancellationSeconds": 2} for n in (1, 4, 8)],
               **{name: True for name in ("acceptanceCustodyVerified", "trainingLeakageAuditPassed", "randomRootLineageVerified", "actualBrowserRun", "actualServerRun", "rollbackVerified")}}
    def finalize(value):
        return harness.finalize(reports, signed(value, private, "independent-evaluator"), {"independent-evaluator": key})
    assert finalize(payload)["decision"] == "pass"
    assert finalize(payload)["releaseApproved"] is False
    assert finalize({**payload, "citationCorrectness": True})["decision"] == "fail"
    assert finalize({**payload, "metricCounts": {}})["decision"] == "fail"
    assert finalize({**payload, "concurrencyMeasurements": []})["decision"] == "fail"
    assert finalize({**payload, "actualBrowserRun": False})["decision"] == "fail"
    assert finalize({**payload, "modelTokensPerSecond": 21})["decision"] == "fail"
    assert finalize({**payload, "concurrency": [True, 4, 8]})["decision"] == "fail"
