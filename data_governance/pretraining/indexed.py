"""Exact candidate index for the frozen lexical/code comparator.

The index prunes comparisons; it does not change thresholds or remove protected
signals. Hash equality, token overlap and literal anchors cover every way the
frozen comparator can match. Resource limits fail closed, including common-word
degeneracy. This is bounded-scale evidence, not a billion-token performance claim.
"""
from collections import defaultdict

from data_governance.splitting.partition import POLICY as MATCH_POLICY, lineage_keys, assign_component
from data_governance.splitting.signatures import compare, canonical, sha, content_segments, features
from .acquisition import POLICY


def payload_features(payload):
    # Only named protocol/provenance containers are excluded. Natural-language
    # questions, answers, context, board-specific facts and all code remain.
    return features(content_segments({k: v for k, v in payload.items() if k not in {"protocol", "sourceEvidence", "lineage"}}))


class Index:
    def __init__(self, rows, *, protected=False):
        self.rows = {row["id"]: row for row in rows}
        if len(self.rows) != len(rows): raise ValueError("Duplicate indexed identity")
        self.protected = protected
        self.hashes, self.tokens, self.anchors, self.identities = (defaultdict(set) for _ in range(4))
        characters = 0
        for row in rows:
            identity = row["id"]
            for key in ("rawSha256", "normalizedSha256", "recordId"):
                if row.get(key): self.identities[key, row[key]].add(identity)
            for key in row.get("keys", []): self.identities["lineage", key].add(identity)
            if row.get("lineage"):
                for key in lineage_keys(row["lineage"]): self.identities["lineage", key].add(identity)
            for name in ("exact", "code", "codeFamily"):
                for digest in row["features"][name]: self.hashes[name, digest].add(identity)
            for segment in row["features"]["text"]:
                for token in segment: self.tokens[token].add(identity)
            for literal in row["features"]["literal"]:
                characters += len(literal)
                if protected: self.anchors[literal[:12]].add(identity)
        if characters > POLICY["maximumFeatureCharacters"]: raise ValueError("Index character budget exceeded")

    def candidates(self, row):
        found = set()
        if row["id"] in self.rows: found.add(row["id"])
        for key in ("rawSha256", "normalizedSha256", "recordId"):
            if row.get(key): found.update(self.identities.get((key, row[key]), ()))
        for key in lineage_keys(row["lineage"]): found.update(self.identities.get(("lineage", key), ()))
        for name in ("exact", "code", "codeFamily"):
            for digest in row["features"][name]: found.update(self.hashes.get((name, digest), ()))
        for segment in row["features"]["text"]:
            for token in segment: found.update(self.tokens.get(token, ()))
        if self.protected:
            for literal in row["features"]["literal"]:
                for start in range(len(literal) - 11): found.update(self.anchors.get(literal[start:start + 12], ()))
        return found


def protected_reason(row, guard):
    if row["id"] == guard["id"] or row.get("recordId") and row.get("recordId") == guard.get("recordId"): return "protected-record-id"
    if row.get("rawSha256") and row.get("rawSha256") == guard.get("rawSha256"): return "protected-raw-bytes"
    if row.get("normalizedSha256") and row.get("normalizedSha256") == guard.get("normalizedSha256"): return "protected-normalized-bytes"
    if lineage_keys(row["lineage"]) & set(guard.get("keys", ())): return "protected-lineage-family"
    return compare(row["features"], guard["features"], MATCH_POLICY, protected=True)


def duplicate_reason(left, right):
    if left.get("rawSha256") and left.get("rawSha256") == right.get("rawSha256"): return "exact-raw-bytes"
    if left.get("normalizedSha256") and left.get("normalizedSha256") == right.get("normalizedSha256"): return "exact-normalized-bytes"
    return compare(left["features"], right["features"], MATCH_POLICY)


def scan(rows, guards, *, maximum_pairs=None):
    if len(rows) > POLICY["maximumDocuments"]: raise ValueError("Indexed document budget exceeded")
    maximum_pairs = POLICY["maximumVerifiedPairs"] if maximum_pairs is None else maximum_pairs
    documents, protected = Index(rows), Index(guards, protected=True)
    duplicate_matches, protected_matches, verified = [], [], 0

    def compare_one(left, right, kind):
        nonlocal verified
        verified += 1
        if verified > maximum_pairs: raise ValueError("Indexed comparison budget exceeded; no unchecked output")
        return protected_reason(left, right) if kind == "protected" else duplicate_reason(left, right)

    for row in sorted(rows, key=lambda x: x["id"]):
        for identity in sorted(protected.candidates(row)):
            reason = compare_one(row, protected.rows[identity], "protected")
            if reason:
                protected_matches.append({"documentId": row["id"], "guardId": identity, "reason": reason})
                break  # Same deterministic first guard as the exhaustive v1.
        for identity in sorted(documents.candidates(row)):
            if identity <= row["id"]: continue
            reason = compare_one(row, documents.rows[identity], "duplicate")
            if reason: duplicate_matches.append({"left": row["id"], "right": identity, "reason": reason})
    return {"protectedMatches": protected_matches, "duplicateMatches": duplicate_matches,
            "verifiedPairs": verified, "exhaustivePairUpperBound": len(rows) * len(guards) + len(rows) * (len(rows) - 1) // 2,
            "matchingPolicy": MATCH_POLICY["policyId"], "matchingPolicySha256": sha(canonical(MATCH_POLICY)), "thresholdsChanged": False}


def partition(rows, guards, *, maximum_pairs=None):
    report = scan(rows, guards, maximum_pairs=maximum_pairs)
    parents = {row["id"]: row["id"] for row in rows}
    by_id = {row["id"]: row for row in rows}

    def find(identity):
        while parents[identity] != identity:
            parents[identity] = parents[parents[identity]]
            identity = parents[identity]
        return identity

    def union(a, b):
        a, b = find(a), find(b)
        parents[max(a, b)] = min(a, b)

    owners = {}
    for row in rows:
        for key in lineage_keys(row["lineage"]):
            if key in owners: union(row["id"], owners[key])
            else: owners[key] = row["id"]
    for match in report["duplicateMatches"]: union(match["left"], match["right"])
    protected_roots = {find(row["documentId"]) for row in report["protectedMatches"]}
    excluded_roots = {find(row["id"]) for row in rows if row.get("inheritedExclusion")}
    groups = defaultdict(list)
    for identity in parents: groups[find(identity)].append(identity)
    duplicate_pairs = {(row["left"], row["right"]): row["reason"] for row in report["duplicateMatches"]}
    decisions = []
    for root, members in sorted(groups.items()):
        keys = sorted(set().union(*(lineage_keys(by_id[identity]["lineage"]) for identity in members)))
        component = sha(canonical(keys))
        reservations = {by_id[identity]["reservedSplit"] for identity in members if by_id[identity].get("reservedSplit")}
        split = next(iter(reservations)) if len(reservations) == 1 else assign_component(component)
        reason = "protected-connected-family" if root in protected_roots else "inherited-exclusion" if root in excluded_roots else "cross-reservation-bridge" if len(reservations) > 1 else None
        representatives = []
        for identity in sorted(members):
            duplicate = next((other for other in representatives if duplicate_pairs.get((min(identity, other), max(identity, other))) not in {None, "parameterized-code-family"}), None)
            decisions.append({"documentId": identity, "componentId": component, "split": "quarantine" if reason or duplicate else split,
                              "reason": reason or ("near-duplicate" if duplicate else "partitioned"), "duplicateOf": duplicate})
            if not duplicate: representatives.append(identity)
    # Components contain all discovered lineage/duplicate edges. A nonzero
    # cross-split edge indicates an implementation error, not a review override.
    included = {row["documentId"]: row["split"] for row in decisions if row["split"] != "quarantine"}
    cross = sum(left in included and right in included and included[left] != included[right] for left, right in duplicate_pairs)
    protected_count = sum(row["documentId"] in included for row in report["protectedMatches"])
    lineage_splits = defaultdict(set)
    for row in rows:
        if row["id"] in included:
            for key in lineage_keys(row["lineage"]): lineage_splits[key].add(included[row["id"]])
    lineage_collisions = sum(len(splits) > 1 for splits in lineage_splits.values())
    if cross or protected_count or lineage_collisions: raise ValueError("Indexed partition leakage")
    report["audit"] = {"crossSplitDuplicateEdges": cross, "protectedMatchesInIncluded": protected_count, "crossSplitLineageKeys": lineage_collisions,
                       "includedDocuments": len(included), "nonemptySplits": sorted(set(included.values())), "semanticCompletenessClaimed": False}
    return sorted(decisions, key=lambda row: row["documentId"]), report
