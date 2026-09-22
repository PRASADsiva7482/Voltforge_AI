"""Reproducible precision evidence; real protected content never leaves the checker."""
from collections import Counter
from pathlib import Path
import json

from data_governance.ingestion.pipeline import build_lock, write_immutable
from data_governance.pretraining import corpus, indexed
from data_governance.splitting.signatures import canonical, sha
from . import controls
from .review import POLICY, explain, review_scan

AI = Path(__file__).resolve().parents[2]
CORPUS = AI / "corpus/pretraining-candidates/v1/8f55313b779ffa8c8beebc19cbb3761a354a0e219fdf794865882d4728dcf852"
ARTIFACTS = AI / "corpus/leakage-reviews/v1"


def source_fingerprints():
    paths = sorted(Path(__file__).parent.glob("*.py")) + [Path(__file__).parent / "policy.v1.json"]
    paths += [AI / name for name in (
        "data_governance/pretraining/indexed.py", "data_governance/splitting/signatures.py",
        "data_governance/splitting/policy.v1.json", "data_governance/pretraining/corpus.py",
        "data_governance/ingestion/pipeline.py",
    )]
    return [{"path": path.relative_to(AI).as_posix(), "sha256": sha(path.read_bytes())} for path in paths]


def control_report():
    rows = []
    matrix = Counter({"truePositive": 0, "falseNegative": 0, "falsePositive": 0, "trueNegative": 0})
    for case in controls.cases():
        result = explain(case["document"], case["guard"])
        positive = case["label"] == "protected-positive"
        blocked = result["existingLeakageBlock"]
        key = ("truePositive" if positive else "falsePositive") if blocked else ("falseNegative" if positive else "trueNegative")
        matrix[key] += 1
        rows.append({"id": case["id"], "family": case["family"], "label": case["label"], "rationale": case["rationale"], **result})
    if matrix["falseNegative"] or not matrix["falsePositive"]:
        raise ValueError("Precision controls did not preserve positives or expose unordered overlap")
    if any(row["admissionAllowed"] or row["existingDecisionChanged"] for row in rows):
        raise ValueError("Review controls authorized an admission")
    strong = [row for row in rows if row["classification"] == "identity-literal-code"]
    if any(row["label"] != "protected-positive" for row in strong):
        raise ValueError("Strong-evidence control labels require review")
    return {
        "cases": len(rows), "independentFamilies": len({row["family"] for row in rows}),
        "activeMatcherConfusionOnAuthoredControls": dict(matrix),
        "activeMatcherPrecisionOnAuthoredControls": matrix["truePositive"] / (matrix["truePositive"] + matrix["falsePositive"]),
        "activeMatcherRecallOnAuthoredControls": matrix["truePositive"] / (matrix["truePositive"] + matrix["falseNegative"]),
        "classificationCounts": dict(Counter(row["classification"] for row in rows)),
        "strongSignalControls": len(strong), "strongSignalNegativeControls": 0,
        "allProtectedPositiveControlsRemainBlocked": True, "automaticReadmissions": 0,
        "rows": rows,
        "limitations": [
            "Constructed controls are regression evidence, not an estimate of corpus-wide precision.",
            "Variants share families; case count does not equal independent scenario count.",
            "A lexical match may be a real paraphrase or an unrelated word overlap; review remains blocked.",
            "No new matcher threshold or corpus admission decision is selected by this review.",
        ],
    }


def real_report():
    checked = corpus.verify(CORPUS, recompute=False)
    rows, _, _, _ = corpus.load_inputs()
    documents = [corpus.describe(row) for row in rows]
    guards, guard_binding = corpus.splitting.protected_registry()
    reviewed = review_scan(documents, guards)
    decisions, active = indexed.partition(documents, guards)
    expected = [json.loads(line) for line in (CORPUS / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    if decisions != expected:
        raise ValueError("Existing corpus partition decisions changed")
    if [dict(documentId=row["documentId"], guardId=row["guardId"], reason=row["activeMatcherReason"]) for row in reviewed["matches"]] != active["protectedMatches"]:
        raise ValueError("Precision review changed protected matches")
    source_by_id = {row["id"]: row["sourceId"] for row in rows}
    direct_ids = {row["documentId"] for row in reviewed["matches"]}
    quarantined = [row for row in decisions if row["split"] == "quarantine"]
    summary = {
        "documents": len(rows), "protectedGuards": len(guards), "protectedRegistryBinding": guard_binding,
        "corpusManifestSha256": checked["manifestSha256"], "corpusPath": CORPUS.relative_to(AI).as_posix(),
        "review": reviewed, "quarantinedDocuments": len(quarantined),
        "quarantinedBySource": dict(Counter(source_by_id[row["documentId"]] for row in quarantined)),
        "quarantineFromFamilyPropagation": sum(row["documentId"] not in direct_ids for row in quarantined),
        "partitionDecisionsUnchanged": True, "partitionDecisionSha256": sha((CORPUS / "decisions.jsonl").read_bytes()),
        "inheritedExclusionsSha256": sha((CORPUS / "inherited-exclusions.json").read_bytes()),
        "splitCounts": checked["scorecard"]["splitCounts"],
        "trainingUniqueProxyTokens": checked["scorecard"]["trainingUniqueProxyTokens"],
        "realMatchAdjudication": "not-adjudicated; counts alone neither prove copying nor authorize readmission",
        "protectedPromptTextExported": False, "automaticReadmissions": 0, "trainingAllowed": False,
    }
    return summary


def compute():
    payloads = {
        "controls.json": (canonical(control_report()) + "\n").encode("utf-8"),
        "real-review.json": (canonical(real_report()) + "\n").encode("utf-8"),
    }
    manifest = {
        "schemaVersion": 1, "taskId": "LLM-TASK-023", "releaseKind": "leakage-precision-review-only",
        "policy": POLICY, "sourceFingerprints": source_fingerprints(),
        "corpusManifest": {"path": (CORPUS / "manifest.json").relative_to(AI).as_posix(), "sha256": sha((CORPUS / "manifest.json").read_bytes())},
        "files": [{"path": name, "sha256": sha(raw), "bytes": len(raw)} for name, raw in sorted(payloads.items())],
        "trainingAllowed": False, "matcherChanged": False, "automaticReadmissions": 0,
    }
    manifest["contentId"] = sha(canonical(manifest))
    return payloads, manifest


def build():
    with build_lock(ARTIFACTS / ".build.lock"):
        payloads, manifest = compute()
        target = ARTIFACTS / manifest["contentId"]
        for name, raw in payloads.items():
            write_immutable(target / name, raw)
        write_immutable(target / "manifest.json", (canonical(manifest) + "\n").encode("utf-8"))
    return target


def verify(path, *, recompute=False):
    path = Path(path).resolve()
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest["contentId"] != path.name or sha(canonical({key: value for key, value in manifest.items() if key != "contentId"})) != path.name:
        raise ValueError("Leakage review manifest changed")
    if manifest["policy"] != POLICY or manifest["sourceFingerprints"] != source_fingerprints() or manifest["trainingAllowed"] is not False or manifest["automaticReadmissions"] != 0 or manifest["matcherChanged"] is not False:
        raise ValueError("Leakage review policy or implementation changed")
    if manifest["corpusManifest"] != {"path": (CORPUS / "manifest.json").relative_to(AI).as_posix(), "sha256": sha((CORPUS / "manifest.json").read_bytes())}:
        raise ValueError("Leakage review corpus binding changed")
    if [row["path"] for row in manifest["files"]] != ["controls.json", "real-review.json"]:
        raise ValueError("Unexpected leakage review artifact paths")
    for row in manifest["files"]:
        target = path / row["path"]
        if target.is_symlink() or target.stat().st_size != row["bytes"] or sha(target.read_bytes()) != row["sha256"]:
            raise ValueError("Leakage review evidence bytes changed")
    corpus.verify(CORPUS, recompute=False)
    if recompute:
        payloads, expected = compute()
        if expected != manifest or any((path / name).read_bytes() != raw for name, raw in payloads.items()):
            raise ValueError("Leakage review does not reproduce")
    return manifest
