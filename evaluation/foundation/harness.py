"""Lane-separated evaluation with fixed denominators and clustered uncertainty."""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import math
import random

from .grading import grade, compile_grade, verify_signed_receipt, without_echo
from .observer import InvocationLedger
from .suite import ROOT, canonical, read, sha, verify_suite


def request_for(case, lane, seed, message=None, history=None):
    # Never expose expected values, rubric, grading keys or metadata to an adapter.
    return {"caseId": case["id"], "lane": lane, "seed": seed, "message": case["prompt"] if message is None else message,
            "context": deepcopy(case["context"]), "history": deepcopy(history or []),
            "sessionId": f"eval:{case['id']}:{lane}:{seed}"}


def percentile(values, fraction):
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (index - lower)


def family_interval(rows, policy):
    if not rows:
        return None
    clusters = defaultdict(list)
    for row in rows:
        clusters[row["templateFamilyId"]].append(1 if row["status"] == "pass" else 0)
    groups = list(clusters.values())
    rng = random.Random(policy["seeds"]["bootstrap"])
    samples = []
    for _ in range(policy["seeds"]["bootstrapReplicates"]):
        sampled = [rng.choice(groups) for _ in groups]
        samples.append(sum(map(sum, sampled)) / sum(map(len, sampled)))
    return {"confidence": 0.95, "lower": percentile(samples, 0.025), "upper": percentile(samples, 0.975),
            "independentTemplateFamilies": len(groups), "method": "family-cluster-percentile-bootstrap",
            "pendingCountedAsNotPassed": True}


def evaluate(cases, adapter, *, lane, seed, ledger=None, reviews=None, compiler_receipts=None, trust=None):
    policy = read(ROOT / "acceptance-policy.v1.json")
    suite = verify_suite()
    if lane not in policy["lanes"] or seed not in policy["seeds"]["generation"]:
        raise ValueError("Unknown lane or unfrozen seed")
    ledger = InvocationLedger() if ledger is None else ledger
    reviews, compiler_receipts, trust = reviews or {}, compiler_receipts or {}, trust or {}
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate case input")
    frozen_digests = read(ROOT / "manifest.v1.json")["caseDigests"]
    if any(frozen_digests.get(case["id"]) != sha(canonical(case).encode()) for case in cases):
        raise ValueError("Case does not match its frozen fingerprint")
    rows = []
    for case in cases:
        history = []
        turns_valid = True
        turn_observations = []
        try:
            for turn in case["turns"]:
                prior_request = request_for(case, lane, seed, turn, history)
                prior_response = adapter(prior_request)
                prior_observation = ledger.inspect(prior_response, prior_request)
                turn_observations.append(prior_observation)
                if lane != "rule_only" and not prior_observation["releaseEligible"]:
                    turns_valid = False
                history.extend([{"role": "user", "content": turn}, {"role": "assistant", "content": prior_response.get("text", "")}])
            request = request_for(case, lane, seed, history=history)
            response = adapter(request)
            observation = ledger.inspect(response, request)
            grading = grade(case, response, request, policy=policy, suite_sha=suite["suiteSha256"], policy_sha=suite["policySha256"], reviews=reviews.get(case["id"], ()), trust=trust)
            substitution = lane != "rule_only" and not response.get("unavailable") and (not observation["releaseEligible"] or not turns_valid)
            if response.get("unavailable"):
                grading = {"status": "blocked", "reason": response["unavailable"]}
            elif substitution:
                grading = {"status": "fail", "reason": "unobserved-or-ineligible-neural-generation", "criticalFailure": True}
            bindings = {"caseId": case["id"], "outputSha256": sha(response.get("text", "").encode()),
                        "suiteSha256": suite["suiteSha256"], "policySha256": suite["policySha256"], "lane": lane, "seed": seed}
            compilation = None
            if case["split"] == "compile":
                compilation = compile_grade(case, response.get("text", ""), compiler_receipts.get(case["id"]), trust, bindings)
            row = {"caseId": case["id"], "category": case["category"], "split": case["split"],
                   "templateFamilyId": case["templateFamilyId"], "status": grading["status"], "reason": grading["reason"],
                   "outputSha256": bindings["outputSha256"], "observation": observation, "priorTurnObservations": turn_observations,
                   "criticalFailure": grading.get("criticalFailure", False), "compile": compilation,
                   "normalizedAnswer": without_echo(response.get("text", ""), request)}
        except Exception as error:
            row = {"caseId": case["id"], "category": case["category"], "split": case["split"],
                   "templateFamilyId": case["templateFamilyId"], "status": "fail", "reason": "adapter-or-grader-error:" + type(error).__name__,
                   "outputSha256": None, "observation": {"observed": False, "releaseEligible": False, "forwardCalls": 0, "logitsElements": 0, "generatedTokens": 0},
                   "criticalFailure": lane != "rule_only", "compile": None, "normalizedAnswer": "", "priorTurnObservations": turn_observations}
        rows.append(row)
    return score(rows, cases=cases, lane=lane, seed=seed, policy=policy, suite=suite)


def score(rows, *, cases, lane, seed, policy, suite):
    rows = deepcopy(rows)
    ids = [row["caseId"] for row in rows]
    expected = {case["id"] for case in cases}
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError("Missing/duplicate/unexpected result rows; denominator cannot shrink")
    core = [row for row in rows if row["split"] == "acceptance"]
    by_category = defaultdict(list)
    for row in core:
        by_category[row["category"]].append(row)
    duplicates = defaultdict(list)
    for row in core:
        if row["normalizedAnswer"]:
            duplicates[row["normalizedAnswer"]].append(row)
    collapsed = {row["caseId"] for group in duplicates.values() if len({row["templateFamilyId"] for row in group}) > 1 for row in group}
    statuses = Counter(row["status"] for row in rows)
    core_rate = sum(row["status"] == "pass" for row in core) / len(core) if core else None
    categories = {key: {"passed": sum(row["status"] == "pass" for row in group), "denominator": len(group),
                        "rate": sum(row["status"] == "pass" for row in group) / len(group)} for key, group in by_category.items()}
    multi = [row for row in core if row["category"] == "multi_turn"]
    compilations = [row for row in rows if row["split"] == "compile"]
    adversarial = [row for row in rows if row["split"] == "adversarial"]
    critical = sum(row["criticalFailure"] for row in rows)
    pending = statuses["pending"] + statuses["blocked"]
    pending_compiles = sum((row["compile"] or {}).get("status") == "pending" for row in compilations)
    full_cohort = len(core) == 500 and len([row for row in rows if row["split"] == "adversarial"]) == 100 and len(compilations) == 100
    failures = []
    if core_rate is not None and core_rate < policy["thresholds"]["overallCorrectness"]:
        failures.append("overall-correctness")
    if any(row["rate"] < policy["thresholds"]["minimumCategoryCorrectness"] for row in categories.values()):
        failures.append("per-category-correctness")
    if multi and sum(row["status"] == "pass" for row in multi) / len(multi) < policy["thresholds"]["multiTurnSuccess"]:
        failures.append("multi-turn-correctness")
    if core and len(collapsed) / len(core) > policy["thresholds"]["maximumAnswerCollapse"]:
        failures.append("answer-collapse")
    if compilations and sum(row["status"] == "pass" and (row["compile"] or {}).get("status") == "pass" for row in compilations) / len(compilations) < policy["thresholds"]["compileSuccess"]:
        failures.append("compile-success")
    if adversarial and sum(row["status"] == "pass" for row in adversarial) / len(adversarial) < policy["thresholds"]["adversarialSuccess"]:
        failures.append("adversarial-success")
    observations = [observation for row in rows for observation in (*row.get("priorTurnObservations", ()), row["observation"])]
    artifact_ids = {item["artifactId"] for item in observations if item["releaseEligible"]}
    if len(artifact_ids) > 1:
        failures.append("mixed-model-identities")
    if critical:
        failures.append("critical-failure")
    decision = "diagnostic-only" if lane == "rule_only" else "fail" if critical else "blocked" if pending or pending_compiles or not full_cohort else "fail" if failures else "pass-for-this-seed"
    for row in rows:
        row.pop("normalizedAnswer", None)
    return {"lane": lane, "seed": seed, "decision": decision, "releaseApproved": False, "fullCohort": full_cohort,
            "suiteSha256": suite["suiteSha256"], "policySha256": suite["policySha256"], "statuses": dict(statuses),
            "corePassedFractionIncludingPending": core_rate, "coreMetricComplete": bool(core) and all(row["status"] not in {"pending", "blocked"} for row in core),
            "categories": categories, "familyInterval": family_interval(core, policy), "collapseCases": len(collapsed),
            "criticalFailures": critical, "failedMetrics": failures, "pendingCompilerReceipts": pending_compiles,
            "observedArtifactIds": sorted(artifact_ids),
            "observedForwardCalls": sum(item["forwardCalls"] for item in observations),
            "observedGeneratedTokens": sum(item["generatedTokens"] for item in observations),
            "remainingReleaseGates": ["all-seeds", "signed-retrieval-and-claim-metrics", "custody-and-training-leakage", "browser-and-server-measurements", "independent-release-approval"],
            "rows": rows}


def finalize(seed_reports, measurements, trust):
    """Evaluate signed aggregate measurements; this is still not registry activation."""
    policy, binding = read(ROOT / "acceptance-policy.v1.json"), verify_suite()
    if len(seed_reports) != 3 or {r["seed"] for r in seed_reports} != set(policy["seeds"]["generation"]):
        raise ValueError("All frozen generation seeds required exactly once")
    lanes = {r["lane"] for r in seed_reports}
    if len(lanes) != 1 or next(iter(lanes)) not in set(policy["lanes"]) - {"rule_only"}:
        raise ValueError("One neural/product lane required")
    if any(r["decision"] != "pass-for-this-seed" or r["suiteSha256"] != binding["suiteSha256"] or r["policySha256"] != binding["policySha256"] for r in seed_reports):
        return {"decision": "blocked", "releaseApproved": False, "reason": "seed-gates-incomplete"}
    identities = {tuple(r.get("observedArtifactIds", ())) for r in seed_reports}
    if len(identities) != 1 or len(next(iter(identities))) != 1 or any(r.get("fullCohort") is not True for r in seed_reports):
        return {"decision": "blocked", "releaseApproved": False, "reason": "complete-cohort-and-one-candidate-required"}
    required = {"suiteSha256": binding["suiteSha256"], "policySha256": binding["policySha256"],
                "scorecardsSha256": sha(canonical(seed_reports).encode()), "lane": next(iter(lanes))}
    try:
        payload, _ = verify_signed_receipt(measurements, trust, "evaluator", required)
    except Exception:
        return {"decision": "blocked", "releaseApproved": False, "reason": "signed-independent-measurements-required"}
    minimums = {"citationCorrectness": 0.95, "supportedClaimAccuracy": 0.95, "retrievalRecallAt5": 0.95,
                "browserScenarios": 25, "modelTokensPerSecond": 20}
    maximums = {"p95FirstModelTokenSeconds": 3, "cancellationSeconds": 2, "criticalFailures": 0}
    def valid_number(value):
        return type(value) in {int, float} and math.isfinite(value) and value >= 0
    failures = [key for key, limit in minimums.items() if not valid_number(payload.get(key)) or payload[key] < limit]
    for metric in ("citationCorrectness", "supportedClaimAccuracy", "retrievalRecallAt5"):
        counts = payload.get("metricCounts", {}).get(metric, {})
        correct, denominator = counts.get("correct"), counts.get("denominator")
        if (type(correct) is not int or type(denominator) is not int or denominator < 50
                or not 0 <= correct <= denominator or not valid_number(payload.get(metric))
                or not math.isclose(payload[metric], correct / denominator, rel_tol=0, abs_tol=1e-12)):
            failures.append(metric + "-denominator")
    failures += [key for key, limit in maximums.items() if not valid_number(payload.get(key)) or payload[key] > limit]
    if type(payload.get("browserScenarios")) is not int:
        failures.append("browser-scenario-count")
    failures += [key for key in ("acceptanceCustodyVerified", "trainingLeakageAuditPassed", "randomRootLineageVerified", "actualBrowserRun", "actualServerRun", "rollbackVerified") if payload.get(key) is not True]
    if payload.get("inputTokens") != 2048 or payload.get("outputTokens") != 256 or canonical(payload.get("concurrency")) != "[1,4,8]":
        failures.append("measurement-workload")
    concurrent = payload.get("concurrencyMeasurements", [])
    if (not isinstance(concurrent, list) or len(concurrent) != 3
            or any(not isinstance(row, dict) for row in concurrent)
            or canonical([row.get("concurrency") for row in concurrent]) != "[1,4,8]"
            or any(type(row.get("requests")) is not int or row["requests"] < row["concurrency"]
                   or any(not valid_number(row.get(key)) for key in ("p95FirstModelTokenSeconds", "modelTokensPerSecond", "cancellationSeconds")) for row in concurrent)):
        failures.append("per-concurrency-measurements")
    elif any(concurrent[0][key] != payload.get(key) for key in ("p95FirstModelTokenSeconds", "modelTokensPerSecond", "cancellationSeconds")):
        failures.append("single-session-measurement-mismatch")
    return {"decision": "fail" if failures else "pass", "releaseApproved": False, "failedGates": failures,
            "scope": "Independent evaluation result only; signed release review and activation remain separate."}
