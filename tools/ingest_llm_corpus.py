"""Explicit offline build/verify step; runtime serving never imports this CLI."""
import argparse
import json
from pathlib import Path
import sys

AI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI))
from data_governance.ingestion import pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "fixtures", "verify"))
    parser.add_argument("--release", type=Path)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.report and args.report.exists():
        parser.error("Report exists; use a new evidence path")
    if args.action == "verify":
        if not args.release:
            parser.error("verify requires --release")
        result = pipeline.verify(args.release, recompute=args.recompute)
    elif args.action == "build":
        result = pipeline.build(pipeline.approved_plan(), AI, AI / "corpus/ingestion/v1", AI / "corpus/.work")
    else:
        result = pipeline.build(pipeline.fixture_plan(), pipeline.FIXTURES, AI / "corpus/extraction-fixtures/v1", AI / "corpus/.work")
    if args.report:
        pipeline.exclusive_json(args.report, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Raw parser errors can contain document excerpts; CLI diagnostics are
        # deliberately content-free. Detailed denial counters live in manifests.
        print(json.dumps({"status": "failed", "errorType": type(error).__name__, "rawContentLogged": False}))
        sys.exit(1)
