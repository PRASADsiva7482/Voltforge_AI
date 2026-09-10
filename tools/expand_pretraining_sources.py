"""Pinned source expansion for task 023; metadata discovery does not admit data."""
import argparse
import json
from pathlib import Path
import sys
import urllib.request

AI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI))

PINS = {
    "quii/learn-go-with-tests": "4675d96ad50b815d3e9fdc2122ab3dbc14636252",
    "rust-lang/book": "1500248d8f230566e4ec9f27fcbb8fe9e2898ab1",
    "squidfunk/mkdocs-material": "9d65447eb4039c153edefbc378029257886737ff",
    "DaveGamble/cJSON": "fb16e5cf358798aabb049655975cde8427101056",
    "rxi/log.c": "f9ea34994bd58ed342d2245cd4110bb5c6790153",
}


def discover():
    for repository, revision in PINS.items():
        request = urllib.request.Request(f"https://api.github.com/repos/{repository}/git/trees/{revision}?recursive=1", headers={"User-Agent": "VoltForge-pinned-source-review", "Accept-Encoding": "identity"})
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read(8_388_609)
        if len(raw) > 8_388_608:
            raise ValueError("Metadata response exceeds bounded discovery limit")
        tree = json.loads(raw)
        if tree.get("truncated") or tree.get("sha") != revision:
            raise ValueError("Incomplete pinned tree")
        paths = [row["path"] for row in tree["tree"] if row["type"] == "blob"]
        print(json.dumps({"repository": repository, "revision": revision,
                          "licensePaths": [path for path in paths if any(part.lower().startswith(("license", "copying", "notice")) for part in path.split("/"))],
                          "rootMarkdown": [path for path in paths if "/" not in path and path.lower().endswith(".md")],
                          "selectedAreaExamples": [path for path in paths if path.startswith(("src/ch03", "src/ch04", "docs/setup/changing-", "src/log"))][:30]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["discover", "acquire", "verify-acquisition", "scope-summary", "verify-admission", "build", "verify", "require-use", "source-impact"])
    parser.add_argument("--release", type=Path)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--usage")
    parser.add_argument("--source-id")
    args = parser.parse_args()
    if args.command == "discover":
        discover()
    elif args.command in {"verify-admission", "build", "verify", "require-use", "source-impact"}:
        from data_governance.source_expansion import admission, candidate
        if args.command == "verify-admission":
            decision, packet = admission.verify()
            print(json.dumps({"status": "passed-exact-source-input-admission", "sources": len(decision["sources"]), "selectedFiles": sum(row["kind"] == "candidate-source" for row in packet["files"]), "corpusTrainingAllowed": False}))
        elif args.command == "build":
            target, score = candidate.build()
            print(json.dumps({"status": "passed-expanded-candidate-build", "releasePath": str(target), "scorecard": score}))
        else:
            if args.release is None:
                parser.error("--release is required")
            if args.command == "source-impact":
                candidate.verify(args.release, recompute=False)
                source_ids = {row["sourceId"] for row in candidate.read(args.release / "sources.json")}
                if args.source_id not in source_ids:
                    parser.error("A known --source-id is required")
                affected = []
                decisions = {row["documentId"]: row for line in (args.release / "decisions.jsonl").read_text(encoding="utf-8").splitlines() if (row := json.loads(line))}
                for split in candidate.corpus.SPLITS:
                    for line in (args.release / (split + ".jsonl")).read_text(encoding="utf-8").splitlines():
                        row = json.loads(line)
                        if row["sourceId"] == args.source_id:
                            affected.append({"documentId": row["id"], "split": split, "tokens": row["tokens"], "normalizedSha256": row["normalizedSha256"], "decisionReason": decisions[row["id"]]["reason"]})
                print(json.dumps({"sourceId": args.source_id, "affectedDocuments": affected, "trainingArtifactsInThisCandidateWorkflow": [], "action": "report-only-no-deletion", "sourceUseDecision": str(admission.DECISION_PATH.relative_to(AI))}))
            elif args.command == "require-use":
                try:
                    result = candidate.require_use(args.release, args.usage)
                    print(json.dumps(result))
                except ValueError as error:
                    if str(error) != "EXPANDED_CORPUS_RELEASE_GATES_UNMET":
                        raise
                    print(json.dumps({"status": "blocked", "code": str(error), "trainingAllowed": False}))
                    raise SystemExit(2)
            else:
                manifest = candidate.verify(args.release, recompute=args.recompute)
                print(json.dumps({"status": "passed-expanded-candidate-verification", "contentId": manifest["contentId"], "recomputed": args.recompute, "trainingAllowed": False}))
    else:
        from data_governance.source_expansion import acquisition
        if args.command == "acquire":
            target = acquisition.acquire()
            result = acquisition.verify(target)
            print(json.dumps({"status": "passed-source-review-acquisition", "releasePath": str(target), "sourceFiles": sum(row["kind"] == "candidate-source" for row in result["files"]), "denied": result["denied"], "acquiredBytes": result["acquiredBytes"], "trainingAllowed": False}))
        else:
            if args.release is None:
                parser.error("--release is required")
            result = acquisition.verify(args.release)
            if args.command == "scope-summary":
                import re
                print(json.dumps({"manifestSha256": acquisition.sha((args.release / "manifest.json").read_bytes())}))
                for source in acquisition.CATALOG["sources"]:
                    notes = []
                    for name in source["scopeEvidencePaths"]:
                        raw = (args.release / source["sourceId"] / name).read_text(encoding="utf-8")
                        notes.append({"path": name, "licenseLines": [line[:240] for line in raw.splitlines() if re.search(r"license|licensed|copyright|permission|proprietary|commercial", line, re.I)][:14]})
                    print(json.dumps({"sourceId": source["sourceId"], "scopeEvidence": notes}))
            else:
                print(json.dumps({"status": "passed-acquisition-verification", "contentId": result["contentId"], "trainingAllowed": False}))
