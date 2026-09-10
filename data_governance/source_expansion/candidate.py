"""Expand the candidate with new families; preserve all old exclusions and roles."""
import json
from pathlib import Path

from data_governance.ingestion.normalization import normalize
from data_governance.ingestion.pipeline import build_lock, write_immutable
from data_governance.pretraining import corpus, indexed, mixture
from data_governance.splitting.signatures import canonical, sha
from model.tokenizer import VoltForgeTokenizer
from . import acquisition, admission

AI = acquisition.AI
PRIOR = AI / "corpus/pretraining-candidates/v1/8f55313b779ffa8c8beebc19cbb3761a354a0e219fdf794865882d4728dcf852"
OUTPUT = AI / "corpus/pretraining-candidates/source-expansion-v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fingerprints():
    paths = sorted(Path(__file__).parent.glob("*.py")) + sorted(Path(__file__).parent.glob("*.json"))
    paths += [AI / "model/tokenizer.py", AI / "data_governance/pretraining/indexed.py", AI / "data_governance/pretraining/corpus.py"]
    return [{"path": path.relative_to(AI).as_posix(), "sha256": sha(path.read_bytes())} for path in paths]


def inputs():
    corpus.verify(PRIOR, recompute=False)
    rights, packet = admission.verify()
    rows, old_sources, inherited, _ = corpus.load_inputs()
    old_rows = {row["id"]: row for split in corpus.SPLITS for line in (PRIOR / (split + ".jsonl")).read_text(encoding="utf-8").splitlines() if (row := json.loads(line))}
    for row in rows:
        old = old_rows[row["id"]]
        if row["text"] != old["text"] or row["rawSha256"] != old["rawSha256"] or row["normalizedSha256"] != old["normalizedSha256"]:
            raise ValueError("Inherited text or token accounting changed")
        row["tokens"] = old["tokens"]
        row["reservedSplit"] = old["split"] if old["split"] != "quarantine" else None
        row["inheritedExclusion"] = old["split"] == "quarantine"
    source_map = {row["sourceId"]: row for row in acquisition.CATALOG["sources"]}
    sources = list(old_sources)
    for source in source_map.values():
        permission = next(row for row in rights["sources"] if row["sourceId"] == source["sourceId"])
        sources.append({"sourceId": source["sourceId"], "origin": "https://github.com/" + source["repository"], "revision": source["revision"], "license": "MIT",
                        "sourceInputUseApproved": True, "approvedUses": permission["allowedUses"], "reservedSplit": source["reservedSplit"],
                        "approvalEvidenceSha256": sha(admission.DECISION_PATH.read_bytes()), "acquisitionManifestSha256": rights["acquisitionManifestSha256"],
                        "privacy": rights["privacy"], "retention": acquisition.POLICY["retention"], "deletionProcedure": acquisition.POLICY["deletionProcedure"]})
    root = AI / rights["acquisitionPath"]
    tokenizer = VoltForgeTokenizer()
    tokenizer.load(AI / "model/tokenizers/vfdlm-byte-bpe-v1.1.0")
    for item in packet["files"]:
        if item["kind"] != "candidate-source":
            continue
        source = source_map[item["sourceId"]]
        raw = (root / item["path"]).read_bytes()
        text, normalization = normalize(raw, "text/x-code")
        if len(text.encode("utf-8")) > acquisition.POLICY["maximumNormalizedDocumentBytes"]:
            raise ValueError("Expanded normalized document exceeds its byte budget")
        rows.append({"id": source["sourceId"] + ":" + item["upstreamPath"], "sourceId": source["sourceId"], "domain": source["domain"], "language": source["language"],
                     "familyId": source["repository"], "lineage": {"kind": "repository-document", "repositoryId": source["repository"], "sourceFamilyId": source["repository"], "documentFamilyId": source["repository"] + ":" + item["upstreamPath"]},
                     "text": text, "rawSha256": item["sha256"], "normalizedSha256": sha(text), "normalization": normalization,
                     "representation": "source-code" if source["domain"] == "code" else "technical-prose-with-unexpanded-rst-markup",
                     "sourceMarkup": "markdown with code/include directives preserved; no renderer or remote includes executed",
                     "sourceEvidence": item, "reservedSplit": source["reservedSplit"], "tokens": len(tokenizer.encode(text)),
                     "inheritedExclusion": False, "trainingAllowed": False})
    return sorted(rows, key=lambda row: row["id"]), sources, inherited, rights


def compute():
    rows, sources, inherited, rights = inputs()
    descriptions = [{**corpus.describe(row), "inheritedExclusion": row["inheritedExclusion"]} for row in rows]
    guards, guard_binding = corpus.splitting.protected_registry()
    decisions, leakage = indexed.partition(descriptions, guards)
    by_id = {row["documentId"]: row for row in decisions}
    for row in rows:
        row.pop("_features", None)
        row["split"] = by_id[row["id"]]["split"]
        if row["inheritedExclusion"] and row["split"] != "quarantine":
            raise ValueError("An old quarantine was readmitted")
        if row["split"] != "quarantine" and row["reservedSplit"] != row["split"]:
            raise ValueError("A predeclared source family changed its reserved role")
    included = [row for row in rows if row["split"] != "quarantine"]
    new_ids = {row["sourceId"] for row in acquisition.CATALOG["sources"]}
    blockers = ["inherited-source-use-decisions-remain-open", "input-corpus-budget-and-scale-evidence-pending", "full-training-budget-and-mixture-pilots-pending"]
    for domain in ("language", "code", "electronics", "math"):
        for split, minimum in (("train", 2), ("validation", 1)):
            families = {row["familyId"] for row in included if row["domain"] == domain and row["split"] == split}
            if len(families) < minimum:
                blockers.append("insufficient-independent-" + split + "-families:" + domain)
    score = {
        "candidateDocuments": len(rows), "newSourceFiles": sum(row["sourceId"] in new_ids for row in rows), "newSourceFamilies": len(new_ids),
        "newSourceUseDecisionsRecorded": len(rights["sources"]), "newFilesApprovedAsSourceInputs": sum(row["sourceId"] in new_ids for row in rows),
        "splitCounts": {split: sum(row["split"] == split for row in rows) for split in corpus.SPLITS},
        "newSplitCounts": {split: sum(row["split"] == split and row["sourceId"] in new_ids for row in rows) for split in corpus.SPLITS},
        "allCandidateAccounting": mixture.accounting(rows), "includedAccounting": mixture.accounting(included),
        "trainingUniqueProxyTokens": sum(row["tokens"] for row in included if row["split"] == "train"),
        "validationUniqueProxyTokens": sum(row["tokens"] for row in included if row["split"] == "validation"),
        "inheritedCorpusQuarantinesPreserved": sum(row["inheritedExclusion"] for row in rows), "inheritedDomainExclusionsPreserved": len(inherited),
        "trainingAllowed": False, "releasedGen2TrainingTokens": 0, "actualTrainingExposures": 0,
        "blockers": blockers, "generalLanguageBreadth": "Technical English only; no broad conversational or general-knowledge corpus adequacy claim",
        "tokenCountMethod": "exact owned v1.1.0 BPE proxy; no new Gen2 fitting",
    }
    payloads = {"sources.json": acquisition.data(sources), "scorecard.json": acquisition.data(score), "leakage.json": acquisition.data(leakage),
                "inherited-exclusions.json": acquisition.data(inherited), "decisions.jsonl": b"".join(acquisition.data(row) for row in decisions)}
    for split in corpus.SPLITS:
        payloads[split + ".jsonl"] = b"".join(acquisition.data(row) for row in rows if row["split"] == split)
    manifest = {"schemaVersion": 1, "releaseKind": "pretraining-candidate-only", "taskId": "LLM-TASK-023", "trainingAllowed": False,
                "sourceFingerprints": fingerprints(), "policy": acquisition.POLICY, "protectedRegistry": guard_binding,
                "inputManifests": [{"path": (PRIOR / "manifest.json").relative_to(AI).as_posix(), "sha256": sha((PRIOR / "manifest.json").read_bytes())},
                                   {"path": rights["acquisitionPath"] + "/manifest.json", "sha256": rights["acquisitionManifestSha256"]},
                                   {"path": admission.DECISION_PATH.relative_to(AI).as_posix(), "sha256": sha(admission.DECISION_PATH.read_bytes())}],
                "files": [{"path": name, "sha256": sha(raw), "bytes": len(raw)} for name, raw in sorted(payloads.items())]}
    manifest["contentId"] = sha(canonical(manifest))
    return payloads, manifest, score


def build():
    with build_lock(OUTPUT / ".build.lock"):
        payloads, manifest, score = compute()
        target = OUTPUT / manifest["contentId"]
        for name, raw in payloads.items():
            write_immutable(target / name, raw)
        write_immutable(target / "manifest.json", acquisition.data(manifest))
    return target, score


def verify(path, *, recompute=False):
    path = Path(path).resolve()
    manifest = read(path / "manifest.json")
    if manifest["contentId"] != path.name or sha(canonical({key: value for key, value in manifest.items() if key != "contentId"})) != path.name:
        raise ValueError("Expanded corpus identity changed")
    if manifest["sourceFingerprints"] != fingerprints() or manifest["trainingAllowed"] is not False or manifest["policy"] != acquisition.POLICY:
        raise ValueError("Expanded corpus implementation or admission boundary changed")
    for row in manifest["files"]:
        target = (path / row["path"]).resolve()
        if target.parent != path or target.is_symlink() or len(target.read_bytes()) != row["bytes"] or sha(target.read_bytes()) != row["sha256"]:
            raise ValueError("Expanded corpus shard changed")
    for row in manifest["inputManifests"]:
        target = (AI / row["path"]).resolve()
        if not target.is_relative_to(AI) or sha(target.read_bytes()) != row["sha256"]:
            raise ValueError("Expanded corpus input binding changed")
    admission.verify()
    corpus.verify(PRIOR, recompute=False)
    if recompute:
        payloads, expected, _ = compute()
        if expected != manifest or any((path / name).read_bytes() != raw for name, raw in payloads.items()):
            raise ValueError("Expanded corpus does not reproduce")
    return manifest


def require_use(path, usage):
    verify(path, recompute=False)
    if usage != "corpus-candidate-review":
        raise ValueError("EXPANDED_CORPUS_RELEASE_GATES_UNMET")
    return {"usage": usage, "trainingAllowed": False}
