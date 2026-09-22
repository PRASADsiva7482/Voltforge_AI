"""Manage independent frozen Gen2 evaluation; a baseline never approves a model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

AI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI))
from evaluation.foundation import suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("draft", "freeze", "verify", "baseline", "check-leakage"))
    parser.add_argument("--report", type=Path)
    parser.add_argument("--candidate-jsonl", type=Path)
    args = parser.parse_args()
    if args.report and args.report.exists():
        parser.error("Report already exists; select a new path to preserve prior evidence")
    if args.command == "draft":
        splits = suite.build_cases()
        suite.validate_isolation(splits)
        result = {"status": "draft-valid", "counts": {key: len(value) for key, value in splits.items()}}
    elif args.command == "freeze":
        result = suite.freeze()
    elif args.command == "verify":
        result = suite.verify_suite()
    elif args.command == "check-leakage":
        if args.candidate_jsonl is None:
            parser.error("check-leakage requires --candidate-jsonl")
        records = [json.loads(line, object_pairs_hook=suite.unique_object) for line in args.candidate_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
        result = suite.check_training_candidates(records)
    else:
        from evaluation.foundation.baseline import run_baseline
        result = run_baseline()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"report": str(args.report), "status": result["status"], "sha256": suite.sha(args.report.read_bytes())}))
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
