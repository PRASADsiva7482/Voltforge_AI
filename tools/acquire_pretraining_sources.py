"""Explicit bounded download of pinned task-023 candidate sources."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_governance.pretraining import acquisition_v2 as acquisition

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("action", choices=("acquire", "verify"))
parser.add_argument("--release", type=Path)
parser.add_argument("--report", type=Path)
args = parser.parse_args()
if args.action == "verify" and not args.release: parser.error("verify requires --release")
result = acquisition.acquire() if args.action == "acquire" else {"status": "passed", "contentId": acquisition.verify(args.release)["contentId"]}
if args.report: acquisition.write_immutable(args.report, acquisition.data(result))
print(json.dumps(result, indent=2))
