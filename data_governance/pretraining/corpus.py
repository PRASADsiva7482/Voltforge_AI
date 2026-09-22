"""Immutable mixed corpus candidates, provenance, token counts and release gaps."""
from collections import Counter
import json
from pathlib import Path

from data_governance.ingestion.normalization import normalize
from data_governance.splitting.signatures import features, words, normalized, canonical, sha
from data_governance.splitting import release as splitting
from synthetic_data.foundation_domain import release as domain
from model.tokenizer import VoltForgeTokenizer
from . import acquisition_v2 as acquisition, indexed, mixture
from .acquisition import AI, ROOT, POLICY, data, write_immutable, build_lock

ACQUISITION = AI / "corpus/acquisition/v2/80396fd5000abf7f4f7d62bd0ce895a63f0436ee19271a64d8f60fc381264bc3"
DOMAIN = AI / POLICY["oldDomainRelease"]
SPLITS = ("train", "validation", "test", "quarantine")


def read(path): return json.loads(path.read_text(encoding="utf-8"))


def fingerprints():
    paths = [path for path in ROOT.iterdir() if path.suffix in {".json", ".py"}]
    paths.extend((AI / "model/tokenizer.py", AI / "data_governance/ingestion/normalization.py", AI / "data_governance/ingestion/policy.v1.json"))
    paths.extend(path for path in (AI / POLICY["proxyTokenizerPath"]).iterdir() if path.is_file())
    paths.extend(AI / "data_governance/splitting" / name for name in ("partition.py", "release.py", "signatures.py", "policy.v1.json"))
    return [{"path": path.relative_to(AI).as_posix(), "sha256": sha(path.read_bytes())} for path in sorted(paths)]


def load_inputs():
    acquired = acquisition.verify(ACQUISITION)
    domain.verify(DOMAIN, recompute=True)
    source_defs = {source["sourceId"]: source for source in acquired["plan"]["catalog"]["sources"]}
    rows, sources = [], []
    for source in source_defs.values():
        selected = [row for row in acquired["files"] if row["sourceId"] == source["sourceId"]]
        sources.append({"sourceId": source["sourceId"], "origin": "https://github.com/" + source["repository"],
                       "revision": source["revision"], "contentSha256": sha(canonical(selected)), "license": source["licenseId"],
                       "privacy": source["privacy"], "approval": source["approval"], "trainingApproved": source["trainingApproved"],
                       "retention": source["retention"], "deletionProcedure": source["deletionProcedure"], "selectedFiles": selected})
    for item in acquired["files"]:
        if item["kind"] != "candidate-source": continue
        source = source_defs[item["sourceId"]]
        raw = (ACQUISITION / item["path"]).read_bytes()
        # Treat reST as preserved source markup. No directive/include execution,
        # rendering, hidden linked-file acquisition or inferred heading removal.
        text, normalization = normalize(raw, "text/x-code")
        ancestry = {"kind": "repository-document", "repositoryId": source["repository"], "sourceFamilyId": source["repository"],
                    "documentFamilyId": source["repository"] + ":" + item["upstreamPath"]}
        rows.append({"id": source["sourceId"] + ":" + item["upstreamPath"], "sourceId": source["sourceId"], "domain": source["domain"],
                     "language": source["language"], "familyId": source["repository"], "lineage": ancestry, "text": text,
                     "rawSha256": sha(raw), "normalizedSha256": sha(text), "normalization": normalization,
                     "representation": "source-code" if source["domain"] == "code" else "technical-prose-with-unexpanded-rst-markup",
                     "sourceEvidence": item, "trainingAllowed": False})
    old_manifest = read(DOMAIN / "manifest.json")
    source_id = domain.POLICY["sourceId"]
    sources.append({"sourceId": source_id, "origin": "Voltforge_AI/synthetic_data/foundation_domain", "revision": old_manifest["contentId"],
                    "contentSha256": sha((DOMAIN / "manifest.json").read_bytes()), "license": "owned-candidate-source-use-not-yet-admitted",
                    "privacy": "new authored reference examples; no private user data", "approval": {"status": "pending-task023-source-admission"},
                    "trainingApproved": False, "retention": "retain immutable task-022 bytes and exclusions", "deletionProcedure": "block use and report affected lineage; no in-place source edits"})
    old_rows = {row["recordId"]: row for line in (DOMAIN / "candidates.jsonl").read_text(encoding="utf-8").splitlines() if (row := json.loads(line))}
    old_decisions = [json.loads(line) for line in (DOMAIN / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    excluded = [row for row in old_decisions if row["split"] == "quarantine"]
    for decision in old_decisions:
        if decision["split"] == "quarantine": continue
        old = old_rows[decision["documentId"]]
        desc = domain.descriptor(old)
        # No task protocol JSON is used as base text. Preserve the already
        # authored problem/answer/assumption representation and exact ancestry.
        rows.append({"id": old["recordId"], "sourceId": source_id, "domain": "math" if old["domain"] in {"digital", "timing", "control", "memory"} else "electronics",
                     "language": "en", "familyId": old["familyId"], "lineage": old["lineage"], "text": old["normalizedText"],
                     "rawSha256": desc["rawSha256"], "normalizedSha256": desc["normalizedSha256"], "reservedSplit": decision["split"],
                     "representation": "owned-domain-question-explanation", "sourceEvidence": {"manifestSha256": sources[-1]["contentSha256"], "recordId": old["recordId"], "evidence": old["sourceEvidence"]},
                     "trainingAllowed": False, "_features": desc["features"]})
    return sorted(rows, key=lambda row: row["id"]), sources, excluded, acquired


def describe(row):
    signal = row.get("_features") or features([row["text"]])
    if row["representation"] == "technical-prose-with-unexpanded-rst-markup":
        # A prose file containing a Python example must still have lexical
        # protection for its prose. Add signals; never discard the code signals.
        lexical = set(words(row["text"]))
        if len(lexical) >= 6: signal["text"].append(lexical)
        literal = normalized(row["text"])
        signal["literal"].add(literal)
        signal["exact"].add(sha(literal))
    return {key: row[key] for key in ("id", "lineage", "rawSha256", "normalizedSha256")} | {"recordId": row["id"], "features": signal, "reservedSplit": row.get("reservedSplit")}


def compute():
    rows, sources, inherited, acquired = load_inputs()
    guards, protected_evidence = splitting.protected_registry()
    decisions, leakage = indexed.partition([describe(row) for row in rows], guards)
    by_id = {row["documentId"]: row for row in decisions}
    tokenizer = VoltForgeTokenizer()
    tokenizer.load(AI / POLICY["proxyTokenizerPath"])
    for row in rows:
        row.pop("_features", None)
        row.update(split=by_id[row["id"]]["split"], tokens=len(tokenizer.encode(row["text"], allowed_special=False)))
    included = [row for row in rows if row["split"] != "quarantine"]
    training = [row for row in included if row["split"] == "train"]
    total = sum(row["tokens"] for row in training)
    plans = [mixture.plan(training, target) for target in sorted({total, 2000000000, 20000000000})]
    readiness = mixture.release_readiness(included, sources, leakage["audit"])
    coverage = {"candidateRecords": len(rows), "inheritedQuarantinedRecordsNotReadmitted": len(inherited),
                "sourceFileAcquisitionDenials": acquired["denied"], "includedRecords": len(included),
                "splitCounts": {split: sum(row["split"] == split for row in rows) for split in SPLITS},
                "allCandidateAccounting": mixture.accounting(rows), "includedAccounting": mixture.accounting(included),
                "trainingUniqueProxyTokens": total, "actualTrainingExposures": 0, "releasedGen2TrainingTokens": 0,
                "languageScope": "English technical documentation and owned domain examples; C source. No broad general-language corpus claimed.",
                "trainingAllowed": False, "readiness": readiness}
    payloads = {"sources.json": data(sources), "scorecard.json": data(coverage), "mixture-plans.json": data(plans),
                "leakage.json": data(leakage), "inherited-exclusions.json": data(inherited),
                "decisions.jsonl": b"".join(data(row) for row in decisions)}
    for split in SPLITS: payloads[split + ".jsonl"] = b"".join(data(row) for row in rows if row["split"] == split)
    manifest = {"schemaVersion": 1, "releaseKind": "pretraining-candidate-only", "taskId": "LLM-TASK-023", "trainingAllowed": False,
                "policy": POLICY, "sourceFingerprints": fingerprints(), "protectedRegistry": protected_evidence,
                "inputManifests": [{"path": (path / "manifest.json").relative_to(AI).as_posix(), "sha256": sha((path / "manifest.json").read_bytes())} for path in (ACQUISITION, DOMAIN)],
                "files": [{"path": name, "sha256": sha(raw), "bytes": len(raw)} for name, raw in sorted(payloads.items())]}
    manifest["contentId"] = sha(canonical(manifest))
    return payloads, manifest, coverage


def build(output_root=None):
    output_root = Path(output_root or AI / "corpus/pretraining-candidates/v1").resolve()
    with build_lock(AI / "corpus/.work/pretraining-corpus.lock"):
        payloads, manifest, score = compute()
        target = output_root / manifest["contentId"]
        for name, raw in payloads.items(): write_immutable(target / name, raw)
        write_immutable(target / "manifest.json", data(manifest))
    return {"status": "passed-candidate-build-release-blocked", "releasePath": str(target), "manifestSha256": sha((target / "manifest.json").read_bytes()), "scorecard": score}


def verify(path, *, recompute=False):
    path = Path(path).resolve()
    manifest = read(path / "manifest.json")
    if manifest["contentId"] != path.name or sha(canonical({k: v for k, v in manifest.items() if k != "contentId"})) != path.name: raise ValueError("Corpus manifest identity changed")
    if manifest["trainingAllowed"] is not False or manifest["sourceFingerprints"] != fingerprints(): raise ValueError("Corpus implementation or use boundary changed")
    for row in manifest["files"]:
        target = (path / row["path"]).resolve()
        if target.parent != path or sha(target.read_bytes()) != row["sha256"] or target.stat().st_size != row["bytes"]: raise ValueError("Corpus shard changed")
    for row in manifest["inputManifests"]:
        if sha((AI / row["path"]).read_bytes()) != row["sha256"]: raise ValueError("Corpus source manifest changed")
    if recompute:
        payloads, expected, _ = compute()
        if expected != manifest or any((path / name).read_bytes() != raw for name, raw in payloads.items()): raise ValueError("Corpus candidate does not reproduce")
    return {"status": "passed", "releasePath": str(path), "recomputed": recompute, "manifestSha256": sha((path / "manifest.json").read_bytes()), "scorecard": read(path / "scorecard.json")}


def require_use(path, usage):
    result = verify(path, recompute=True)
    if usage not in POLICY["allowedUses"]: raise ValueError("PRETRAINING_CORPUS_RELEASE_GATES_UNMET")
    return result


def source_impact(path, source_id):
    verify(path, recompute=True)
    sources = read(Path(path) / "sources.json")
    if source_id not in {row["sourceId"] for row in sources}: raise ValueError("Unknown source")
    affected = []
    for split in SPLITS:
        for line in (Path(path) / (split + ".jsonl")).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["sourceId"] == source_id: affected.append({"documentId": row["id"], "split": split, "tokens": row["tokens"], "normalizedSha256": row["normalizedSha256"]})
    return {"sourceId": source_id, "affectedDocuments": affected, "trainingArtifacts": [], "action": "report-only-no-files-deleted"}
