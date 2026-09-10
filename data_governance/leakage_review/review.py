"""Explain existing leakage decisions using counts only, without permitting use."""
from collections import Counter
import json
from pathlib import Path

from data_governance.pretraining import indexed
from data_governance.splitting.signatures import words

ROOT = Path(__file__).resolve().parent
POLICY = json.loads((ROOT / "policy.v1.json").read_text(encoding="utf-8"))
LEXICAL = frozenset({"lexical-paraphrase", "protected-content-containment"})
STRONG = frozenset({
    "protected-record-id", "protected-raw-bytes", "protected-normalized-bytes", "protected-lineage-family",
    "exact-content", "renamed-code", "parameterized-code-family", "protected-exact-containment",
})


def shortest_cover(sequence, required):
    """Length only, not token values/locations. O(n) sliding window."""
    if not required:
        return None
    counts = Counter()
    covered = 0
    start = 0
    best = None
    for end, word in enumerate(sequence):
        if word in required:
            covered += counts[word] == 0
            counts[word] += 1
        while covered == len(required):
            length = end - start + 1
            best = length if best is None else min(best, length)
            first = sequence[start]
            if first in required:
                counts[first] -= 1
                covered -= counts[first] == 0
            start += 1
    return best


def lexical_observations(document_features, guard_features):
    pairs = len(document_features["text"]) * len(guard_features["text"])
    if pairs > POLICY["maximumLexicalPairsPerMatch"]:
        raise ValueError("Leakage review lexical-pair budget exceeded")
    literals = document_features["literal"]
    if sum(len(text) for text in literals) > POLICY["maximumLiteralCharactersPerMatch"]:
        raise ValueError("Leakage review character budget exceeded")
    sequences = []
    count = 0
    for text in sorted(literals):
        sequence = words(text)
        count += len(sequence)
        if count > POLICY["maximumWordTokensPerMatch"]:
            raise ValueError("Leakage review word budget exceeded")
        sequences.append((set(sequence), sequence))
    observations = []
    bag_visits = 0
    window_visits = 0
    for left in document_features["text"]:
        for right in guard_features["text"]:
            bag_visits += len(left) + len(right)
            if bag_visits > POLICY["maximumBagWordVisitsPerMatch"]:
                raise ValueError("Leakage review word-set work budget exceeded")
            if not left or not right:
                continue
            common = left & right
            jaccard = len(common) / len(left | right)
            coverage = len(common) / len(right)
            if jaccard < indexed.MATCH_POLICY["nearTextJaccard"] and not (
                len(common) >= indexed.MATCH_POLICY["minimumNearWords"] and coverage >= indexed.MATCH_POLICY["protectedContainment"]
            ):
                continue
            spans = []
            for bag, sequence in sequences:
                if bag == left:
                    window_visits += len(sequence)
                    if window_visits > POLICY["maximumWindowTokenVisitsPerMatch"]:
                        raise ValueError("Leakage review window work budget exceeded")
                    spans.append(shortest_cover(sequence, common))
            spans = [span for span in spans if span is not None]
            observations.append({
                "documentDistinctWords": len(left), "protectedDistinctWords": len(right),
                "sharedDistinctWords": len(common), "jaccard": jaccard,
                "protectedWordCoverage": coverage,
                "minimumCoveringWordWindow": min(spans) if spans else None,
                "windowMeaning": "Smallest window containing the shared word set in any order; not a copied phrase",
            })
    return sorted(observations, key=lambda row: (-row["protectedWordCoverage"], -row["jaccard"], row["documentDistinctWords"], row["protectedDistinctWords"]))


def explain(document, guard):
    reason = indexed.protected_reason(document, guard)
    if reason is not None and reason not in STRONG | LEXICAL:
        raise ValueError("Unreviewed matcher reason; no approval decision returned")
    if reason in STRONG:
        classification = "identity-literal-code"
    elif reason in LEXICAL:
        classification = "lexical-review-required"
    else:
        classification = "no-match"
    observations = lexical_observations(document["features"], guard["features"]) if reason in LEXICAL else []
    return {
        "documentId": document["id"], "guardId": guard["id"], "activeMatcherReason": reason,
        "classification": classification, "existingLeakageBlock": reason is not None,
        "reviewComplete": reason not in LEXICAL, "admissionAllowed": False,
        "existingDecisionChanged": False, "lexicalObservations": observations,
        "claim": POLICY["interpretation"][classification],
    }


def review_scan(documents, guards):
    result = indexed.scan(documents, guards)
    if len(result["protectedMatches"]) > POLICY["maximumReviewMatches"]:
        raise ValueError("Leakage review match budget exceeded")
    by_id = {item["id"]: item for item in documents}
    by_guard = {item["id"]: item for item in guards}
    details = []
    for item in result["protectedMatches"]:
        detail = explain(by_id[item["documentId"]], by_guard[item["guardId"]])
        if detail["activeMatcherReason"] != item["reason"] or not detail["existingLeakageBlock"] or detail["admissionAllowed"]:
            raise ValueError("Leakage review changed an active exclusion")
        details.append(detail)
    return {
        "matchingPolicy": result["matchingPolicy"], "matchingPolicySha256": result["matchingPolicySha256"],
        "reviewPolicy": POLICY["policyId"], "classificationCounts": dict(Counter(row["classification"] for row in details)),
        "matches": details, "duplicateMatches": result["duplicateMatches"],
        "verifiedPairs": result["verifiedPairs"], "matcherChanged": False,
        "existingDecisionsChanged": 0, "automaticReadmissions": 0, "trainingAllowed": False,
    }
