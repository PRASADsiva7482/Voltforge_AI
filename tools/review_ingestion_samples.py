"""Record bounded extraction review against source bytes and authored expectations."""
import argparse
import base64
from collections import defaultdict
import json
from pathlib import Path
import sys

AI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI))
from data_governance.ingestion import pipeline as p

EXPECTATIONS = {
    "01-si.html": (["2.2 kΩ", "100 µF", "25 °C", "E = mc^(2)", "V_(out)", r"\(X_C=1/(2\pi fC)\)",
                    "<mfrac><mi>V</mi><mi>R</mi></mfrac>", "    digitalWrite(13, HIGH);"], ["Navigation only", "Footer only"]),
    "02-indent.py.txt": (["\tif resistance > 0:\n\t\treturn voltage / resistance\n"], ["\r"]),
    "03-unicode.txt": (["Café; R = 10 kΩ; C = 4.7 µF; x² + y² = z².\n"], ["Accept all cookies", "\r"]),
    "05-utf16.txt": (["UTF16 source: 220 Ω and 10 µA.\n"], ["\ufffd"]),
    "06-cp1252.txt": (["Legacy source: 47 µF, 25 °C.\n"], ["\ufffd"]),
    "07-code-math.pdf": (["V = I * R; R = 220 ohm; I = 0.01 A", "\n    digitalWrite(13, HIGH);\n"], ["\ufffd"]),
    "16-fenced.md": (["```python\n    print('keep indentation')\n\n\n```", "Equation: V = I * R."], ["\r"]),
    "17-margins.pdf": (["Formula: V = I * R"], ["Repeated synthetic document header", "Repeated synthetic document footer"]),
}


def iter_records(release, kind):
    for row in p.read_json(release / "manifest.json")["files"]:
        if row["kind"] == kind:
            with (release / row["path"]).open(encoding="utf-8") as stream:
                for line in stream:
                    yield json.loads(line)


def review(source_release, fixture_release):
    checks = [p.verify(source_release, recompute=True), p.verify(fixture_release, recompute=True)]
    fixtures = {row["lineage"]["inputPath"]: row for row in iter_records(fixture_release, "normalized")}
    rejected = {row["lineage"]["inputPath"]: row for row in iter_records(fixture_release, "quarantine")}
    decisions = []
    for entry in p.read_json(p.FIXTURES / "catalog.json")["files"]:
        name, expected = entry["path"], entry["expectedOutcome"]
        observed = "accepted" if name in fixtures else rejected[name]["reason"]
        if observed != expected:
            raise ValueError("Fixture quarantine decision mismatch")
        criteria = {"expectedOutcome": expected, "observedOutcome": observed}
        if name in fixtures:
            text = fixtures[name]["text"]
            present, absent = EXPECTATIONS[name]
            if not all(value in text for value in present) or any(value in text for value in absent):
                raise ValueError("Reviewed code/equation/SI extraction mismatch")
            criteria.update(textSha256=fixtures[name]["textSha256"], requiredFragments=len(present), forbiddenFragments=len(absent))
        decisions.append({"path": name, "inputSha256": entry["sha256"], "passed": True, **criteria})
    raw = {row["documentId"]: row for row in iter_records(source_release, "raw")}
    counts, samples = defaultdict(int), []
    for row in iter_records(source_release, "normalized"):
        unit = row["lineage"]["inputPath"]
        if counts[unit] == 2:
            continue
        counts[unit] += 1
        original = json.loads(base64.b64decode(raw[row["documentId"]]["rawBase64"]))
        extracted = json.loads(row["text"])
        # Check the actual nested values, including escaped multi-line firmware,
        # tool evidence, uncertainty, citations and proposal-only structured actions.
        if any(extracted[key] != original[key] for key in ("task", "input", "output")):
            raise ValueError("Sample extraction dropped or altered a task payload")
        samples.append({"documentId": row["documentId"], "inputPath": unit, "locator": row["lineage"]["locator"],
                        "rawSha256": row["lineage"]["rawSha256"], "textSha256": row["textSha256"], "task": original["task"],
                        "passed": True, "criteria": "Exact nested task/input/output values, firmware whitespace, evidence and output constraints preserved."})
    if len(samples) != 8 or len(counts) != 4:
        raise ValueError("Expected two reviewed samples per approved source shard")
    return {"schemaVersion": 1, "taskId": "LLM-TASK-020", "status": "passed", "reviewer": "Codex extraction review with executable expectations",
            "sourceManifestSha256": p.sources.file_hash(source_release / "manifest.json"),
            "fixtureManifestSha256": p.sources.file_hash(fixture_release / "manifest.json"), "offlineReproductionChecks": checks,
            "sourceSamples": samples, "fixtureDecisions": decisions, "sourceSamplesReviewed": 8, "fixtureDecisionsReviewed": 17,
            "limits": ["Extraction fidelity only; no new independent electronics correctness or model quality claim.",
                       "PDF review covers these authored digital-text fixtures; multi-column reading order, scanned text and complex visual equations need source-specific review.",
                       "No rejected document body is copied into release shards or this report. Synthetic secret sentinels remain only in the explicitly marked regression inputs."],
            "trainingAllowed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-release", type=Path, required=True)
    parser.add_argument("--fixture-release", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    if args.preview:
        sys.stdout.reconfigure(encoding="utf-8")
        for row in iter_records(args.fixture_release, "normalized"):
            print(p.canonical({"path": row["lineage"]["inputPath"], "text": row["text"]}))
        counts = defaultdict(int)
        for row in iter_records(args.source_release, "normalized"):
            unit = row["lineage"]["inputPath"]
            if counts[unit] < 2:
                counts[unit] += 1
                task = json.loads(row["text"])
                print(p.canonical({"path": unit, "task": task["task"], "prompt": task["input"]["user"]["text"],
                                   "answerExcerpt": str(task["output"]["assistantText"])[0:800]}))
        return
    if not args.report or args.report.exists():
        parser.error("A new --report path is required")
    result = review(args.source_release, args.fixture_release)
    p.exclusive_json(args.report, result)
    print(json.dumps({"status": result["status"], "sourceSamples": 8, "fixtureDecisions": 17}))


if __name__ == "__main__":
    main()
