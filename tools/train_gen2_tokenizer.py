"""Train and verify the owned Gen2 tokenizer using the admitted task023 corpus."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

AI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("probe", "fit", "_fit-worker", "compare", "publish", "verify", "integration"))
    parser.add_argument("--target", type=int)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    if args.command == "probe":
        from model.gen2_tokenizer_training.fitting import fit
        tokenizer, lineage, seconds = fit(args.target)
        print(json.dumps({"status": "passed-prepublication-fit-probe", "target": args.target, "actualVocabSize": tokenizer.vocab_size, "trainingDocuments": lineage["documents"], "trainingBytes": lineage["bytes"], "fitSeconds": seconds, "releasePublished": False}))
    elif args.command == "fit":
        from model.gen2_tokenizer_training.corpus import POLICY
        result = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "_fit-worker", "--target", str(args.target)], cwd=AI, text=True, encoding="utf-8", capture_output=True, timeout=POLICY["maximumFitSeconds"]+120)
        if result.returncode:
            raise RuntimeError(result.stderr or result.stdout)
        print(result.stdout, end="")
    else:
        from model.gen2_tokenizer_training import release
        if args.command == "_fit-worker":
            target, measurement = release.build_candidate(args.target)
            print(json.dumps({"status": "published-frozen-tokenizer-candidate", "path": target.relative_to(AI).as_posix(), "measurement": measurement}))
        elif args.command == "compare":
            from model.gen2_tokenizer_training.evaluation import compare
            target, report = compare()
            print(json.dumps({"status": report["status"], "path": target.relative_to(AI).as_posix(), "selection": report["selection"]}))
        elif args.command == "publish":
            if not args.comparison:
                parser.error("publish requires --comparison")
            target = release.publish(args.comparison)
            print(json.dumps({"status": "published-owned-tokenizer-release", "path": target.relative_to(AI).as_posix()}))
        elif args.command == "verify":
            if not args.release:
                parser.error("verify requires --release")
            tokenizer, manifest = release.load(args.release, recompute=args.recompute)
            print(json.dumps({"status": "passed", "contentId": manifest["contentId"], "vocabSize": tokenizer.vocab_size, "recomputed": args.recompute, "servingAllowed": False}))
        elif args.command == "integration":
            if not args.release:
                parser.error("integration requires --release")
            from model.gen2_tokenizer_training.integration import evaluate
            target, report = evaluate(args.release)
            print(json.dumps({"status": report["status"], "path": target.relative_to(AI).as_posix(), "model": report["model"]}))


if __name__ == "__main__":
    main()
