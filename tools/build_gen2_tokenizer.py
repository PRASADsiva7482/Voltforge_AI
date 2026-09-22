"""Offline task-024 fixture preparation; no command can admit or fit a corpus."""
import argparse
import json
from pathlib import Path
import sys

AI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI))
from model.gen2_tokenizer.comparison import compare_fixtures
from model.gen2_tokenizer.contract import TokenizerError
from model.gen2_tokenizer.release import require_corpus_fitting, verify_fixture
from data_governance.ingestion.pipeline import build_lock, write_immutable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fixtures = commands.add_parser("fixtures")
    fixtures.add_argument("--report", type=Path)
    verify = commands.add_parser("verify-fixture")
    verify.add_argument("--release", type=Path, required=True)
    verify.add_argument("--recompute", action="store_true")
    corpus = commands.add_parser("check-corpus")
    corpus.add_argument("--release", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "fixtures":
            result = compare_fixtures()
        elif args.command == "verify-fixture":
            tokenizer, manifest = verify_fixture(args.release, recompute=args.recompute)
            result = {"status": "passed-fixture-verification", "binding": tokenizer.binding(), "contentId": manifest["contentId"], "recomputed": args.recompute}
        else:
            require_corpus_fitting(args.release)
            raise AssertionError("No admitted corpus adapter exists")
        if getattr(args, "report", None):
            # Measurements vary; existing receipts must never be overwritten.
            with build_lock(args.report.parent / ".gen2-tokenizer-report.lock"):
                write_immutable(args.report, (json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii"))
        print(json.dumps(result if args.command != "fixtures" else {
            "status": result["status"], "candidates": [{key: row[key] for key in ("targetVocabSize", "actualVocabSize", "targetReached", "fixtureRelease")} for row in result["candidates"]],
            "tokenizerReleaseAccepted": False, "report": str(args.report) if args.report else None,
        }))
        return 0
    except (TokenizerError, ValueError, OSError) as error:
        print(json.dumps({"status": "blocked", "code": error.code if isinstance(error, TokenizerError) else "GEN2_BUILD_OR_ARTIFACT_FAILED", "tokenizerReleaseAccepted": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
