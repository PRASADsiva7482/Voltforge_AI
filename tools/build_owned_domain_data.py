"""Build/verify task-022 domain candidates; never starts model training."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synthetic_data.foundation_domain import release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "verify"))
    parser.add_argument("--release", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--recompile", action="store_true")
    args = parser.parse_args()
    if args.action == "verify" and not args.release: parser.error("verify requires --release")
    result = release.build(args.output_root) if args.action == "build" else release.verify(args.release, recompute=args.recompute, recompile=args.recompile)
    if args.report: release.write_immutable(args.report, release.json_bytes(result))
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
