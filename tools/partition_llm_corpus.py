"""Build or verify the Gen2 family split candidate release offline."""
import argparse
import json
from pathlib import Path
import sys

AI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI))
from data_governance.splitting import release
from data_governance.ingestion.pipeline import exclusive_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("draft", "build", "verify"))
    parser.add_argument("--release", type=Path)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.report and args.report.exists():
        parser.error("Evidence path exists")
    if args.action == "draft":
        manifest, _, scorecard = release.compute()
        result = {"status": "draft", "releaseId": manifest["releaseId"], "guards": manifest["plan"]["guards"]["kindCounts"], "scorecard": scorecard}
    elif args.action == "build":
        result = release.build()
    else:
        if not args.release:
            parser.error("verify requires --release")
        result = release.verify(args.release, recompute=args.recompute)
    if args.report:
        exclusive_json(args.report, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
