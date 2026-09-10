"""Group lineage and duplicate components before assigning immutable partitions."""
from collections import Counter, defaultdict
import json
from pathlib import Path

from .signatures import canonical, sha, features, compare

ROOT = Path(__file__).resolve().parent
POLICY = json.loads((ROOT / "policy.v1.json").read_text(encoding="utf-8"))


def lineage_keys(lineage):
    required = ("kind", "sourceFamilyId", "documentFamilyId", "repositoryId")
    if any(not isinstance(lineage.get(key), str) or not lineage[key].strip() for key in required):
        raise ValueError("Missing source/document/repository ancestry")
    keys = {"source:" + lineage["sourceFamilyId"], "document:" + lineage["documentFamilyId"]}
    if lineage["kind"] == "repository-document":
        keys.add("repository:" + lineage["repositoryId"])
    elif lineage["kind"] != "owned-synthetic":
        raise ValueError("Unsupported lineage kind")
    # The owned generator repository is permission provenance, not a statistical
    # family. Its recipes, circuits and templates are the indivisible units.
    if lineage["kind"] == "owned-synthetic" and not lineage.get("templateFamilyId"):
        raise ValueError("Synthetic input requires a pre-render template family")
    for key in ("templateFamilyId", "circuitFamilyId", "scenarioFamilyId"):
        if lineage.get(key):
            keys.add(key + ":" + lineage[key])
    if lineage.get("boardVariant"):
        keys.add("board-within-template:" + lineage["templateFamilyId"] + ":" + lineage["boardVariant"])
    return keys


def assign_component(component_id, policy=POLICY):
    weights = policy["splitWeights"]
    if set(weights) != {"train", "validation", "test"} or any(type(v) is not int or v <= 0 for v in weights.values()):
        raise ValueError("Invalid partition weights")
    bucket = int(sha(policy["seed"] + ":" + component_id), 16) % sum(weights.values())
    for name in ("train", "validation", "test"):
        bucket -= weights[name]
        if bucket < 0:
            return name


def plan_partitions(records, protected=(), *, policy=POLICY):
    """Records are descriptors + extracted features; never mutate their payloads."""
    records = sorted(records, key=lambda row: row["id"])
    protected = sorted(protected, key=lambda row: row["id"])
    if len(records) > policy["maxDocuments"] or len(records) * len(protected) + len(records) * (len(records) - 1) // 2 > policy["maxPairComparisons"]:
        raise ValueError("Partition comparison budget exceeded; no unchecked release")
    if len({row["id"] for row in records}) != len(records) or len({row["id"] for row in protected}) != len(protected):
        raise ValueError("Duplicate document/guard identity")
    parents = {row["id"]: row["id"] for row in records}
    by_id = {row["id"]: row for row in records}
    keys = {row["id"]: lineage_keys(row["lineage"]) for row in records}
    key_owner, matches, collisions = {}, [], []

    def find(identity):
        while parents[identity] != identity:
            parents[identity] = parents[parents[identity]]
            identity = parents[identity]
        return identity

    def union(a, b):
        a, b = find(a), find(b)
        parents[max(a, b)] = min(a, b)

    for row in records:
        for key in sorted(keys[row["id"]]):
            if key in key_owner:
                union(row["id"], key_owner[key])
            else:
                key_owner[key] = row["id"]
        for guard in protected:
            reason = None
            if row["id"] == guard["id"] or row.get("recordId") and row.get("recordId") == guard.get("recordId"):
                reason = "protected-record-id"
            elif row.get("rawSha256") and row.get("rawSha256") == guard.get("rawSha256"):
                reason = "protected-raw-bytes"
            elif row.get("normalizedSha256") and row.get("normalizedSha256") == guard.get("normalizedSha256"):
                reason = "protected-normalized-bytes"
            elif keys[row["id"]] & set(guard.get("keys", ())):
                reason = "protected-lineage-family"
            else:
                reason = compare(row["features"], guard["features"], policy, protected=True)
            if reason:
                collisions.append({"documentId": row["id"], "guardId": guard["id"], "reason": reason})
                break
    # All-pairs matching at this reviewed pilot scale avoids probabilistic LSH
    # misses. The explicit limit fails closed before quadratic work grows large.
    for i, left in enumerate(records):
        for right in records[i + 1:]:
            reason = "exact-raw-bytes" if left.get("rawSha256") and left.get("rawSha256") == right.get("rawSha256") else "exact-normalized-bytes" if left.get("normalizedSha256") and left.get("normalizedSha256") == right.get("normalizedSha256") else compare(left["features"], right["features"], policy)
            if reason:
                union(left["id"], right["id"])
                matches.append({"left": left["id"], "right": right["id"], "reason": reason})
    components = defaultdict(list)
    for identity in by_id:
        components[find(identity)].append(identity)
    tainted = {find(row["documentId"]) for row in collisions}
    direct = {row["documentId"]: row for row in collisions}
    assignments, groups = [], []
    pairs = {(row["left"], row["right"]): row["reason"] for row in matches}
    for root, ids in sorted(components.items()):
        group_keys = sorted(set().union(*(keys[identity] for identity in ids)))
        component_id = sha(canonical(group_keys))
        split = "quarantine" if root in tainted else assign_component(component_id, policy)
        representatives = []
        for identity in sorted(ids):
            duplicate = next((other for other in representatives if (min(identity, other), max(identity, other)) in pairs
                              and pairs[(min(identity, other), max(identity, other))] != "parameterized-code-family"), None)
            reason = direct.get(identity, {}).get("reason", "connected-to-protected-family") if split == "quarantine" else "near-duplicate" if duplicate else "partitioned"
            decision = {"documentId": identity, "componentId": component_id, "split": "quarantine" if duplicate else split,
                        "reason": reason, "duplicateOf": duplicate, "lineage": by_id[identity]["lineage"],
                        "rawSha256": by_id[identity].get("rawSha256"), "normalizedSha256": by_id[identity].get("normalizedSha256")}
            if not duplicate:
                representatives.append(identity)
            assignments.append(decision)
        groups.append({"componentId": component_id, "lineageKeys": group_keys, "documents": len(ids), "split": split,
                       "protected": root in tainted})
    return {"assignments": sorted(assignments, key=lambda row: row["documentId"]), "components": sorted(groups, key=lambda row: row["componentId"]),
            "protectedMatches": collisions, "duplicateMatches": matches,
            "splitCounts": dict(Counter(row["split"] for row in assignments)),
            "reasonCounts": dict(Counter(row["reason"] for row in assignments)),
            "semanticCompletenessClaimed": False}


def reserve_before_render(lineages, *, policy=POLICY):
    """Task-022 API: reserve descriptors first; render only after a bound plan exists."""
    rows = [{"id": sha(canonical(row)), "lineage": row, "features": features([])} for row in lineages]
    result = plan_partitions(rows, policy=policy)
    return {"policySha256": sha(canonical(policy)), "seed": policy["seed"], "descriptorsSha256": sha(canonical(sorted(lineages, key=canonical))),
            "lineages": sorted(lineages, key=canonical),
            "assignments": result["assignments"], "renderingStarted": False, "trainingAllowed": False}


def require_render_assignment(reservation, lineage, split, *, policy=POLICY):
    if reservation.get("policySha256") != sha(canonical(policy)) or reservation.get("seed") != policy["seed"]:
        raise ValueError("Reservation policy changed")
    if reservation != reserve_before_render(reservation["lineages"], policy=policy):
        raise ValueError("Reservation assignment changed")
    identity = sha(canonical(lineage))
    matches = [row for row in reservation["assignments"] if row["documentId"] == identity and row["lineage"] == lineage and row["split"] == split]
    if len(matches) != 1 or split not in {"train", "validation", "test"}:
        raise ValueError("Unreserved or relabelled render family")
    return matches[0]
