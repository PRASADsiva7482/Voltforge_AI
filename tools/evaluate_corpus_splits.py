"""Content-free adversarial leakage receipts, including nonempty partition controls."""
import argparse
import json
from pathlib import Path
import sys

AI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI))
from data_governance.splitting import partition as p, release as r
from data_governance.splitting.signatures import canonical, sha, features
from data_governance.ingestion.pipeline import exclusive_json


def lineage(name):
    return {"kind": "owned-synthetic", "sourceFamilyId": "fixture-source:" + name, "documentFamilyId": "fixture-doc:" + name,
            "repositoryId": "fixture-only-repository", "templateFamilyId": "fixture-template:" + name, "boardVariant": "BOARD_A"}


def candidate(identity, text, family=None):
    return {"id": identity, "recordId": identity, "lineage": family or lineage(identity), "rawSha256": sha("raw:" + text),
            "normalizedSha256": sha(text), "features": features([text])}


def evaluate():
    protected_text = "Calculate the current through a resistor with voltage 12 volts and resistance 600 ohms."
    code = "```cpp\nint measure(int sample) { return sample * 4 + 7; }\n```"
    scenarios = [
        ("exact-copy", candidate("new-id", protected_text), {"id": "guard", "features": features([protected_text]), "keys": []}),
        ("paraphrase", candidate("new-id", "Please give current for resistance 600 ohms with potential 12 volts. Explain the resistor calculation."),
         {"id": "guard", "features": features([protected_text]), "keys": []}),
        ("renamed-code", candidate("new-id", "```cpp\n// a different comment\nint transform(int reading) { return reading * 4 + 7; }\n```"),
         {"id": "guard", "features": features([code]), "keys": []}),
        ("parameterized-code", candidate("new-id", "```cpp\nint measure(int sample) { return sample * 19 + 43; }\n```"),
         {"id": "guard", "features": features([code]), "keys": []}),
        ("board-template-variant", candidate("new-id", "Different content", {**lineage("shared"), "boardVariant": "BOARD_B"}),
         {"id": "guard", "features": features([]), "keys": sorted(p.lineage_keys(lineage("shared")))}),
        ("embedded-marker", candidate("new-id", "Padding sealed-independent-fixture-answer-91aa72 Additional unrelated text"),
         {"id": "guard", "features": features(["sealed-independent-fixture-answer-91aa72"]), "keys": []}),
        ("historical-id", candidate("old-heldout", "Changed text"),
         {"id": "guard", "recordId": "old-heldout", "features": features([]), "keys": []}),
    ]
    receipts = []
    for name, row, guard in scenarios:
        result = p.plan_partitions([row], [guard])
        if result["splitCounts"] != {"quarantine": 1}:
            raise ValueError("A deliberately leaked fixture was accepted")
        receipts.append({"scenario": name, "candidateSha256": row["normalizedSha256"], "outcome": "rejected", "reason": result["protectedMatches"][0]["reason"]})
    control = candidate("unrelated", "A quartz oscillator maintains periodic timing through piezoelectric mechanical resonance.")
    result = p.plan_partitions([control], [scenarios[0][2]])
    if result["assignments"][0]["split"] == "quarantine":
        raise ValueError("Unrelated negative control rejected")
    rows = [candidate(f"control-{i:03d}", f"fixture {i}") for i in range(100)]
    result = p.plan_partitions(rows)
    audit = r.cross_split_audit(rows, result["assignments"], [])
    if set(audit["nonemptySplits"]) != {"train", "validation", "test"}:
        raise ValueError("Nonempty partition fixture failed")
    reservation = p.reserve_before_render([lineage(f"recipe-{i}") for i in range(12)])
    for entry in reservation["assignments"]:
        p.require_render_assignment(reservation, entry["lineage"], entry["split"])
    contaminated = [candidate("a", "one", lineage("shared")), candidate("b", "two", lineage("shared"))]
    try:
        r.cross_split_audit(contaminated, [{"documentId": "a", "split": "train"}, {"documentId": "b", "split": "test"}], [])
    except ValueError:
        deliberate_collision_rejected = True
    else:
        raise ValueError("Deliberate cross-split lineage collision escaped")
    return {"schemaVersion": 1, "taskId": "LLM-TASK-021", "status": "passed", "scenarios": receipts,
            "leakedFixturesRejected": len(receipts), "unrelatedControlAccepted": True,
            "nonemptyPartitionControl": {"documents": 100, "splitCounts": result["splitCounts"], "crossSplitAudit": audit},
            "deliberateCrossSplitCollisionRejected": deliberate_collision_rejected,
            "preRenderReservationsVerified": 12, "trainingAllowed": False, "rawProtectedContentStored": False,
            "scope": "Authored synthetic positive/negative controls, not real corpus adequacy or general semantic recall.",
            "sourceFingerprints": [{"path": "data_governance/splitting/" + name, "sha256": r.inventory.file_hash(p.ROOT / name)} for name in r.SOURCES]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Evidence already exists")
    result = evaluate()
    exclusive_json(args.report, result)
    print(json.dumps({"status": "passed", "leakedFixturesRejected": result["leakedFixturesRejected"],
                      "nonemptyControlSplits": result["nonemptyPartitionControl"]["splitCounts"]}))


if __name__ == "__main__":
    main()
