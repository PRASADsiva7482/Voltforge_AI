"""Generate or verify read-only precision evidence for the task-023 candidate."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_governance.leakage_review import evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("controls")
    commands.add_parser("build")
    verify = commands.add_parser("verify")
    verify.add_argument("--release", type=Path, required=True)
    verify.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    if args.command == "controls":
        result = evaluation.control_report()
        print(json.dumps({key: value for key, value in result.items() if key != "rows"}))
    elif args.command == "build":
        target = evaluation.build()
        print(json.dumps({"status": "passed-review-only", "releasePath": str(target), "trainingAllowed": False, "automaticReadmissions": 0}))
    else:
        manifest = evaluation.verify(args.release, recompute=args.recompute)
        print(json.dumps({"status": "passed-review-verification", "contentId": manifest["contentId"], "recomputed": args.recompute, "trainingAllowed": False}))


if __name__ == "__main__":
    main()
