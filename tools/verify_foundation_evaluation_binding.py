"""Bind the new evaluation policy without rewriting the frozen task-017 design."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

AI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI))
from evaluation.foundation.suite import canonical, read, sha, verify_suite

PATH = AI / "foundation/evaluation-binding.v1.json"
INPUTS = (
    "foundation/contract.v1.json", "foundation/release-policy.v1.json",
    "foundation/compatibility-map.v1.json", "docs/LLM_FOUNDATION_DESIGN.v1.md",
    "evaluation/foundation/acceptance-policy.v1.json", "evaluation/foundation/manifest.v1.json",
)


def create():
    if PATH.exists():
        raise ValueError("Evaluation binding already exists; never overwrite history")
    suite = verify_suite()
    binding = {"schemaVersion": 1, "bindingId": "vf-foundation-evaluation-binding-v1", "taskId": "LLM-TASK-018",
               "freezeStatus": "locked", "thresholdState": "frozen-independent-suite",
               "suiteSha256": suite["suiteSha256"], "policySha256": suite["policySha256"],
               "historicalDesignRewritten": False, "modelReleaseApproved": False, "activationAllowed": False,
               "scope": "Satisfies task-017 pending threshold/suite reference only; every training, lineage and release gate remains required.",
               "inputs": [{"path": path, "sha256": sha((AI / path).read_bytes())} for path in INPUTS]}
    PATH.write_text(json.dumps(binding, indent=2) + "\n", encoding="utf-8", newline="\n")
    return verify()


def verify(binding=None):
    binding = read(PATH) if binding is None else binding
    if (binding.get("schemaVersion"), binding.get("bindingId"), binding.get("taskId"), binding.get("freezeStatus"), binding.get("thresholdState")) != (1, "vf-foundation-evaluation-binding-v1", "LLM-TASK-018", "locked", "frozen-independent-suite"):
        raise ValueError("Invalid evaluation binding identity")
    if any(binding.get(key) is not False for key in ("historicalDesignRewritten", "modelReleaseApproved", "activationAllowed")):
        raise ValueError("Evaluation binding cannot rewrite history or grant release approval")
    inputs = binding["inputs"]
    if len(inputs) != len(INPUTS) or {row["path"] for row in inputs} != set(INPUTS):
        raise ValueError("Incomplete evaluation binding inputs")
    for row in inputs:
        if sha((AI / row["path"]).read_bytes()) != row["sha256"]:
            raise ValueError("Evaluation binding hash mismatch: " + row["path"])
    suite = verify_suite()
    if any(binding[key] != suite[key] for key in ("suiteSha256", "policySha256")):
        raise ValueError("Evaluation binding suite/policy mismatch")
    return {"status": "passed", "thresholdState": binding["thresholdState"], "bindingSha256": sha(canonical(binding).encode()),
            "suite": suite, "historicalDesignFilesPreserved": 4, "modelReleaseApproved": False, "activationAllowed": False,
            "scope": binding["scope"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.report and args.report.exists():
        parser.error("Report exists; use a new path")
    result = create() if args.create else verify()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
