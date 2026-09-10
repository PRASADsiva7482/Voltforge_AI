"""Assemble/recompute candidate corpus; no implicit source or training approval."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_governance.pretraining import corpus

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("action", choices=("build", "verify", "source-impact", "require-use"))
parser.add_argument("--release", type=Path)
parser.add_argument("--report", type=Path)
parser.add_argument("--recompute", action="store_true")
parser.add_argument("--source-id")
parser.add_argument("--usage")
args = parser.parse_args()
if args.action != "build" and not args.release: parser.error("this action requires --release")
if args.action == "source-impact" and not args.source_id: parser.error("source-impact requires --source-id")
if args.action == "require-use" and not args.usage: parser.error("require-use requires --usage")
if args.action == "build": result = corpus.build()
elif args.action == "verify": result = corpus.verify(args.release, recompute=args.recompute)
elif args.action == "source-impact": result = corpus.source_impact(args.release, args.source_id)
else: result = corpus.require_use(args.release, args.usage)
if args.report: corpus.write_immutable(args.report, corpus.data(result))
print(json.dumps(result, indent=2))
