"""Deterministic token quotas and repetition accounting, without model training."""
from collections import Counter, defaultdict
from fractions import Fraction
import math

from .acquisition import POLICY


def quotas(total, weights):
    if type(total) is not int or total < 0 or not weights: raise ValueError("Invalid exposure budget")
    parsed = {key: Fraction(str(value)) for key, value in weights.items()}
    if any(value <= 0 for value in parsed.values()) or sum(parsed.values()) != 1: raise ValueError("Mixture weights must be positive and sum to one")
    exact = {key: total * value for key, value in parsed.items()}
    result = {key: int(value) for key, value in exact.items()}
    for key in sorted(result, key=lambda key: (-(exact[key] - result[key]), key))[:total - sum(result.values())]: result[key] += 1
    return result


def plan(records, total, *, weights=None, maximum_passes=None):
    weights = POLICY["mixtureWeights"] if weights is None else weights
    maximum_passes = POLICY["maximumDocumentPasses"] if maximum_passes is None else maximum_passes
    if type(maximum_passes) is not int or maximum_passes < 1 or maximum_passes > POLICY["maximumDocumentPasses"]: raise ValueError("Repetition cap exceeds reviewed policy")
    if len({row["id"] for row in records}) != len(records) or len({row["normalizedSha256"] for row in records}) != len(records): raise ValueError("Duplicate records cannot inflate unique-token capacity")
    if any(row["split"] != "train" or type(row["tokens"]) is not int or row["tokens"] <= 0 or row["domain"] not in weights for row in records): raise ValueError("Only positive-token training records may enter mixture planning")
    quotas_by_domain = quotas(total, weights)
    pools = {domain: sorted([row for row in records if row["domain"] == domain], key=lambda row: row["id"]) for domain in weights}
    unique = {domain: sum(row["tokens"] for row in pool) for domain, pool in pools.items()}
    deficits = {domain: max(0, quota - unique[domain] * maximum_passes) for domain, quota in quotas_by_domain.items()}
    schedule = []
    if not any(deficits.values()):
        for domain, pool in pools.items():
            remaining = quotas_by_domain[domain]
            for epoch in range(maximum_passes):
                for row in pool:
                    count = min(remaining, row["tokens"])
                    if count: schedule.append({"documentId": row["id"], "domain": domain, "pass": epoch + 1, "startToken": 0, "tokenCount": count})
                    remaining -= count
                if not remaining: break
            if remaining: raise ValueError("Mixture capacity arithmetic mismatch")
    return {"requestedExposures": total, "targetWeights": weights, "quotas": quotas_by_domain, "uniqueAvailableTokens": unique,
            "maximumPasses": maximum_passes, "deficitTokens": deficits, "feasible": not any(deficits.values()), "schedule": schedule,
            "scheduledExposures": sum(row["tokenCount"] for row in schedule), "actualTrainingExposures": 0,
            "tokenCountMethod": "existing-v1.1.0-tokenizer-proxy", "qualityValidated": False}


def accounting(records):
    totals = {}
    for field in ("sourceId", "domain", "language", "split"):
        counts = defaultdict(lambda: {"records": 0, "uniqueTokens": 0, "families": set()})
        for row in records:
            entry = counts[row[field]]
            entry["records"] += 1
            entry["uniqueTokens"] += row["tokens"]
            entry["families"].add(row["familyId"])
        totals[field] = {key: {**value, "families": len(value["families"])} for key, value in sorted(counts.items())}
    return totals


def release_readiness(records, sources, leakage, *, budget=None, pilot_evidence=None, gen2_tokenizer=None):
    blockers = []
    source_map = {row["sourceId"]: row for row in sources}
    used = {row["sourceId"] for row in records}
    for source_id in sorted(used):
        source = source_map.get(source_id)
        required = {"origin", "revision", "contentSha256", "license", "privacy", "approval", "retention", "deletionProcedure"}
        if not source or not required.issubset(source) or source.get("trainingApproved") is not True or not all(source[key] for key in required): blockers.append("source-rights-not-admitted:" + source_id)
    for domain in POLICY["mixtureWeights"]:
        for split, minimum in (("train", POLICY["minimumIndependentTrainFamiliesPerDomain"]), ("validation", POLICY["minimumIndependentValidationFamiliesPerDomain"])):
            families = {row["familyId"] for row in records if row["domain"] == domain and row["split"] == split}
            if len(families) < minimum: blockers.append("insufficient-independent-" + split + "-families:" + domain)
    if any(leakage.get(key, 1) for key in ("crossSplitDuplicateEdges", "protectedMatchesInIncluded", "crossSplitLineageKeys")): blockers.append("leakage-audit-not-clean")
    # External receipts must be bound and verified by future task-owned adapters.
    # This v1 cannot mint approval by accepting a caller's boolean or token count.
    blockers.append("measured-scaling-budget-unavailable" if budget is None else "measured-scaling-receipt-verifier-not-implemented")
    blockers.append("heldout-mixture-pilot-unavailable" if pilot_evidence is None else "mixture-pilot-receipt-verifier-not-implemented")
    blockers.append("gen2-tokenizer-count-unavailable" if gen2_tokenizer is None else "gen2-tokenizer-receipt-verifier-not-implemented")
    return {"status": "blocked", "trainingAllowed": False, "releasedGen2TrainingTokens": 0, "blockers": blockers,
            "sequenceIssue": POLICY["knownSequenceIssue"]}
