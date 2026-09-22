"""Audit/freeze Gen2 source permissions and token gaps without admitting data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

AI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI))
from data_governance.foundation import inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("draft", "freeze", "verify"))
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.report and args.report.exists():
        parser.error("Report exists; select a new path")
    result = inventory.build_inventory() if args.command == "draft" else inventory.freeze() if args.command == "freeze" else inventory.verify(recompute=args.recompute)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {key: result[key] for key in ("status", "sourceCounts", "unitCounts")}
    if "existingPool" in result:
        summary["existingPool"] = {key: result["existingPool"][key] for key in ("currentApprovedShardRecords", "uniqueEligibleRecords", "proxyTokens", "excluded")}
    else:
        summary.update({key: result[key] for key in ("uniqueEligibleRecords", "existingPoolProxyTokens", "gen2ReleaseReadyTokens")})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
