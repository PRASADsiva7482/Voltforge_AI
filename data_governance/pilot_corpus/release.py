"""Reproducible input admission; source rights never imply model/run approval."""
from collections import Counter
import json
from pathlib import Path

from data_governance.ingestion.normalization import normalize
from data_governance.ingestion.pipeline import build_lock, write_immutable
from data_governance.pretraining import corpus, indexed, mixture
from data_governance.source_expansion import acquisition, admission, candidate
from data_governance.splitting.signatures import canonical, sha
from model.tokenizer import VoltForgeTokenizer
from . import supplement

AI = candidate.AI
ROOT = Path(__file__).resolve().parent
POLICY = json.loads((ROOT / "policy.v1.json").read_text(encoding="utf-8"))
PRIOR = AI / "corpus/pretraining-candidates/source-expansion-v1/733a4f704e7433f3d19f1f3c95e6109eceaf3f838be56acf6efff27322081f30"
OUTPUT = AI / "corpus/pilot-input/v1"
RESOURCE_OUTPUT = AI / "corpus/pilot-input/resources-v1"
DECISION = ROOT / "source-decisions.v1.json"
REVOCATIONS = AI / "data_governance/pilot-input-revocations.v1.json"
SPLITS = corpus.SPLITS
FILE_NAMES = {"sources.json", "scorecard.json", "mixture.json", "leakage.json", "reservations.json", "reference-checks.json", "inherited-exclusions.json", "decisions.jsonl", *(split + ".jsonl" for split in SPLITS)}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def data(value):
    return (canonical(value) + "\n").encode("utf-8")


def binding(path):
    path = Path(path).resolve()
    return {"path": path.relative_to(AI).as_posix(), "sha256": sha(path.read_bytes()), "bytes": path.stat().st_size}


def fingerprints():
    paths = sorted(ROOT.glob("*.py")) + sorted(ROOT.glob("*.json"))
    paths += [AI / "tools/release_pilot_corpus.py", AI / "data_governance/ingestion/pipeline.py"]
    return [binding(path) for path in sorted(paths)]


def input_bindings():
    return [binding(path) for path in (PRIOR / "manifest.json", corpus.ACQUISITION / "manifest.json", corpus.DOMAIN / "manifest.json", admission.DECISION_PATH)]


def safe_file(root, name):
    if not isinstance(name, str) or not name or name.startswith("/") or Path(name).is_absolute() or "\\" in name or ":" in name or ".." in Path(name).parts:
        raise ValueError("Unsafe release path")
    target = root / name
    if any(item.is_symlink() or getattr(item, "is_junction", lambda: False)() for item in (target, *target.parents) if item != root.parent):
        raise ValueError("Linked release path")
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Release path escaped root")
    return target


def rows_at(path):
    return [json.loads(line) for split in SPLITS for line in (path / (split + ".jsonl")).read_text(encoding="utf-8").splitlines()]


def owned_rows():
    rows, receipts = [], []
    for example in supplement.render():
        receipts.append(supplement.verify_example(example))
        family = example["family"]
        text, normalization = normalize(example["text"].encode("utf-8"), "text/x-code")
        identity = supplement.SOURCE + ":" + family
        rows.append({"id": identity, "sourceId": supplement.SOURCE, "domain": example["domain"], "language": "en", "familyId": identity,
                     "lineage": {"kind": "owned-synthetic", "repositoryId": "Voltforge_AI/data_governance/pilot_corpus", "sourceFamilyId": identity, "documentFamilyId": identity, "templateFamilyId": identity},
                     "text": text, "rawSha256": sha(example["text"]), "normalizedSha256": sha(text), "normalization": normalization,
                     "reservedSplit": supplement.RESERVATIONS[family], "split": supplement.RESERVATIONS[family], "inheritedExclusion": False,
                     "representation": "technical-prose-with-unexpanded-rst-markup", "sourceEvidence": {"generator": binding(ROOT / "supplement.py"), "calculationReceiptSha256": sha(canonical(receipts[-1]))}, "trainingAllowed": False})
    return rows, receipts


def source_rows():
    """Full upstream verification, without recomputing historical token counts."""
    candidate.verify(PRIOR, recompute=False)
    corpus.domain.verify(corpus.DOMAIN, recompute=True)
    # The candidate verifier checks v2 acquisition, full selected-file MIT grants,
    # new repository/tree evidence, and all upstream manifest/code bindings.
    rows = rows_at(PRIOR)
    extra, receipts = owned_rows()
    return rows + extra, receipts


def usage_for(split):
    return {"train": ["tokenizer-fitting-input", "bounded-pilot-pretraining-input", "corpus-review"], "validation": ["validation-input", "corpus-review"], "test": ["test-input", "corpus-review"], "quarantine": ["corpus-review"]}[split]


def source_evidence(rows):
    """Exact per-document permissions derived only from separately verified sources."""
    source_defs = {row["sourceId"]: row for row in read(PRIOR / "sources.json")}
    old_packet = corpus.acquisition.verify(corpus.ACQUISITION)
    decision, packet = admission.verify()
    old_mit = {row["sourceId"] for row in old_packet["plan"]["catalog"]["sources"] if row["licenseId"] == "MIT"}
    new_sources = {row["sourceId"] for row in decision["sources"]}
    owned = corpus.domain.POLICY["sourceId"]
    approved = []
    for row in sorted(rows, key=lambda item: item["id"]):
        split = row["split"]
        if split == "quarantine":
            continue
        source_id = row["sourceId"]
        if source_id in old_mit:
            file = next((item for item in old_packet["files"] if item["kind"] == "candidate-source" and item["sourceId"] == source_id and item["sha256"] == row["rawSha256"]), None)
            if not file or not file["licenseReview"]["completeMITGrantChecked"]:
                raise ValueError("Inherited selected-file MIT grant missing")
            evidence = {"basis": "retained complete MIT grant, copyright and SPDX in each selected file plus root LICENSE.md", "manifest": binding(corpus.ACQUISITION / "manifest.json"), "file": file}
        elif source_id == owned:
            evidence = {"basis": "original project-owned deterministic reference generator and independent calculation/compilation receipts; only 17 previously surviving records", "manifest": binding(corpus.DOMAIN / "manifest.json"), "recordEvidence": row["sourceEvidence"]}
        elif source_id in new_sources:
            source_use = "validation-input" if split == "validation" else "pretraining-input"
            if source_use not in next(item for item in decision["sources"] if item["sourceId"] == source_id)["allowedUses"]:
                raise ValueError("Expanded source role not approved")
            file = next((item for item in packet["files"] if item["kind"] == "candidate-source" and item["sourceId"] == source_id and item["sha256"] == row["rawSha256"]), None)
            if not file:
                raise ValueError("Expanded source bytes not approved")
            evidence = {"basis": "separate verified exact-byte source-input decision", "decision": binding(admission.DECISION_PATH), "file": file}
        elif source_id == supplement.SOURCE:
            evidence = {"basis": "original authored reference prose and independent rational-arithmetic checks; reserved validation only", "generator": binding(ROOT / "supplement.py"), "recordEvidence": row["sourceEvidence"]}
        else:
            raise ValueError("Source has no reviewed rights basis")
        approved.append({"documentId": row["id"], "sourceId": source_id, "rawSha256": row["rawSha256"], "normalizedSha256": row["normalizedSha256"], "reservedSplit": split,
                         "allowedUses": usage_for(split), "origin": source_defs.get(source_id, {}).get("origin", "Voltforge_AI/data_governance/pilot_corpus/supplement.py"),
                         "evidence": evidence, "privacy": "public maintainer technical material or original project references; no private user projects; heuristic secret scan required",
                         "retention": POLICY["retention"], "deletionProcedure": POLICY["deletionProcedure"]})
    return approved


def record_source_decisions():
    rows, _ = source_rows()
    value = {"schemaVersion": 1, "decisionId": "vf-pilot-exact-input-decisions-20260909-v1", "reviewer": {"kind": "coding-agent", "name": "Codex"},
             "authorizationBasis": "User explicitly authorized completing task 023. Exact retained permissive-license evidence and project-owned generation were reviewed; no separate owner signature or legal opinion is asserted.",
             "decision": "approved-exact-source-input-uses", "modelReleaseApproved": False, "trainingRunApproved": False,
             "inputBindings": input_bindings(), "documents": source_evidence(rows)}
    write_immutable(DECISION, data(value))
    return {"status": "recorded-exact-source-input-decisions", "documents": len(value["documents"]), "path": DECISION.relative_to(AI).as_posix()}


def verify_rights(rows):
    decision = read(DECISION)
    if decision["schemaVersion"] != 1 or decision["decision"] != "approved-exact-source-input-uses" or decision["modelReleaseApproved"] is not False or decision["trainingRunApproved"] is not False or decision["inputBindings"] != input_bindings():
        raise ValueError("Source decision binding or scope changed")
    if decision["documents"] != source_evidence(rows):
        raise ValueError("Missing, changed or unapproved source rights")
    return decision


def bounded_rows(rows):
    if len(rows) > POLICY["maximumDocuments"]:
        raise ValueError("Document budget exceeded")
    sizes = [len(row["text"].encode("utf-8")) for row in rows]
    if max(sizes, default=0) > POLICY["maximumNormalizedDocumentBytes"] or sum(sizes) > POLICY["maximumNormalizedCorpusBytes"]:
        raise ValueError("Normalized input byte budget exceeded")
    return sum(sizes)


def quality(rows, rights, leakage):
    bounded_rows(rows)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate corpus identity")
    included = [row for row in rows if row["split"] != "quarantine"]
    decisions = {row["documentId"]: row for row in rights["documents"]}
    if len(decisions) != len(rights["documents"]):
        raise ValueError("Duplicate source permission")
    for row in included:
        permission = decisions.get(row["id"])
        if not permission or any(permission.get(key) != row[key] for key in ("sourceId", "rawSha256", "normalizedSha256")) or permission["reservedSplit"] != row["split"] or permission["allowedUses"] != usage_for(row["split"]):
            raise ValueError("Missing source rights or changed reserved role")
        if row["inheritedExclusion"] or row["split"] != row["reservedSplit"] or sha(row["text"]) != row["normalizedSha256"]:
            raise ValueError("Contaminated, altered or readmitted input")
    if any(leakage.get(key, 1) != 0 for key in ("crossSplitDuplicateEdges", "protectedMatchesInIncluded", "crossSplitLineageKeys")):
        raise ValueError("Corpus contamination gate failed")
    coverage = {}
    for domain in POLICY["mixtureWeights"]:
        coverage[domain] = {}
        for split in ("train", "validation"):
            subset = [row for row in included if row["domain"] == domain and row["split"] == split]
            family_count = len({row["familyId"] for row in subset})
            minimum = POLICY["minimumIndependentTrainFamiliesPerDomain" if split == "train" else "minimumIndependentValidationFamiliesPerDomain"]
            total = sum(row["tokens"] for row in subset)
            if family_count < minimum or split == "validation" and total < POLICY["minimumValidationTokensPerDomain"]:
                raise ValueError("Insufficient independent " + split + " coverage: " + domain)
            coverage[domain][split] = {"families": family_count, "documents": len(subset), "uniqueProxyTokens": total}
    training = [row for row in included if row["split"] == "train"]
    total = sum(row["tokens"] for row in training)
    validation = sum(row["tokens"] for row in included if row["split"] == "validation")
    if not POLICY["minimumTrainUniqueProxyTokens"] <= total <= POLICY["maximumTrainUniqueProxyTokens"] or validation < POLICY["minimumValidationUniqueProxyTokens"]:
        raise ValueError("Provisional unique-token input budget unmet")
    plan = mixture.plan(training, POLICY["plannedProxyTokenExposures"], weights=POLICY["mixtureWeights"], maximum_passes=POLICY["maximumDocumentPasses"])
    if not plan["feasible"]:
        raise ValueError("Mixture exceeds repetition capacity")
    return coverage, plan


def compute():
    rows, receipts = source_rows()
    normalized_bytes = bounded_rows(rows)
    rights = verify_rights(rows)
    inherited = read(PRIOR / "inherited-exclusions.json")
    # Count every current input again, including the long FreeRTOS files. This
    # prevents inherited counters from standing in for measured tokenizer work.
    tokenizer = VoltForgeTokenizer()
    tokenizer.load(AI / "model/tokenizers/vfdlm-byte-bpe-v1.1.0")
    for row in rows:
        before = row["text"]
        text, _ = normalize(before.encode("utf-8"), "text/x-code")
        if text != before or sha(text) != row["normalizedSha256"]:
            raise ValueError("Input normalization changed")
        # Preserve the upstream exclusion *type* as well as its outcome. A
        # within-family duplicate removal does not become a protected-family
        # exclusion: that would incorrectly quarantine its retained siblings.
        # Recomputed decisions must still exclude every historical quarantine.
        row["tokens"] = len(tokenizer.encode(text, allowed_special=False))
    guards, guard_binding = corpus.splitting.protected_registry()
    original_owned = {row["recordId"]: row for line in (corpus.DOMAIN / "candidates.jsonl").read_text(encoding="utf-8").splitlines() if (row := json.loads(line))}
    descriptions = []
    for row in rows:
        described = dict(row)
        if row["sourceId"] == corpus.domain.POLICY["sourceId"]:
            described["_features"] = corpus.domain.descriptor(original_owned[row["id"]])["features"]
        descriptions.append({**corpus.describe(described), "inheritedExclusion": row["inheritedExclusion"]})
    decisions, leakage = indexed.partition(descriptions, guards)
    by_id = {row["documentId"]: row for row in decisions}
    for row in rows:
        row["split"] = by_id[row["id"]]["split"]
        row["trainingAllowed"] = False
    coverage, plan = quality(rows, rights, leakage["audit"])
    old_quarantines = {row["id"] for row in rows_at(PRIOR) if row["split"] == "quarantine"}
    if not old_quarantines.issubset({row["id"] for row in rows if row["split"] == "quarantine"}):
        raise ValueError("Historical quarantines changed")
    acquired_bytes = read(corpus.ACQUISITION / "manifest.json")["acquiredBytes"] + read(AI / admission.verify()[0]["acquisitionPath"] / "manifest.json")["acquiredBytes"]
    if acquired_bytes > POLICY["maximumRetainedAcquisitionBytes"]:
        raise ValueError("Retained acquisition exceeds input budget")
    included = [row for row in rows if row["split"] != "quarantine"]
    score = {"status": "passed-input-source-quality-split-budget-gates", "candidateDocuments": len(rows), "splitCounts": dict(Counter(row["split"] for row in rows)),
             "coverage": coverage, "includedAccounting": mixture.accounting(included), "allCandidateAccounting": mixture.accounting(rows),
             "trainingUniqueProxyTokens": sum(row["tokens"] for row in rows if row["split"] == "train"), "validationUniqueProxyTokens": sum(row["tokens"] for row in rows if row["split"] == "validation"),
             "normalizedInputBytes": normalized_bytes, "retainedAcquisitionBytes": acquired_bytes, "independentReferenceChecks": len(receipts), "protectedDescriptors": len(guards),
             "inheritedCorpusQuarantinesPreserved": len(old_quarantines), "inheritedDomainExclusionsPreserved": len(inherited),
             "tokenCountMethod": "exact owned Gen1 v1.1.0 BPE proxy on all input text; Gen2 recount belongs to task 024", "plannedProxyTokenExposures": plan["scheduledExposures"],
             "actualTrainingExposures": 0, "releasedGen2TrainingTokens": 0, "trainingRunApproved": False, "productionTrainingAllowed": False,
             "limits": ["Technical English and C; not broad conversational or general knowledge adequacy", "Math/electronics are small original verified references", "The 20000-exposure plan exercises input scheduling; no held-out model result is claimed", "Protected matching is lexical/structural, not a semantic contamination proof", "Production corpus budget and empirical mixtures remain mandatory in tasks 031/032"]}
    payloads = {"sources.json": data(rights), "scorecard.json": data(score), "mixture.json": data(plan), "leakage.json": data(leakage),
                "reservations.json": data({"inheritedManifest": binding(PRIOR / "manifest.json"), "newFamiliesReservedBeforeRender": supplement.RESERVATIONS}),
                "reference-checks.json": data(receipts), "inherited-exclusions.json": data(inherited), "decisions.jsonl": b"".join(data(row) for row in decisions)}
    for split in SPLITS:
        payloads[split + ".jsonl"] = b"".join(data(row) for row in sorted(rows, key=lambda item: item["id"]) if row["split"] == split)
    plan_binding = {"schemaVersion": 1, "taskId": "LLM-TASK-023", "releaseKind": "admitted-pilot-input-corpus", "inputUseAllowed": True, "allowedUses": POLICY["allowedUses"],
                    "trainingAllowed": False, "trainingRunApproved": False, "productionTrainingAllowed": False, "modelReleaseApproved": False,
                    "policy": POLICY, "sourceFingerprints": fingerprints(), "inputBindings": input_bindings(), "protectedRegistry": guard_binding,
                    "files": [{"path": name, "sha256": sha(raw), "bytes": len(raw)} for name, raw in sorted(payloads.items())]}
    return payloads, plan_binding, score


def verify_resources(path, plan):
    path = Path(path).resolve()
    if not path.is_relative_to(RESOURCE_OUTPUT.resolve()) or path.name != "measurement.json":
        raise ValueError("Resource receipt is outside the reviewed namespace")
    receipt = read(path)
    if receipt["contentId"] != path.parent.name or sha(canonical({key: value for key, value in receipt.items() if key != "contentId"})) != path.parent.name:
        raise ValueError("Resource receipt identity changed")
    if receipt["planSha256"] != sha(canonical(plan)) or receipt["scope"] != "fresh-process-full-input-normalization-proxy-tokenization-indexed-matching-and-quality-verification" or receipt["freshProcess"] is not True:
        raise ValueError("Resource measurement is not bound to these inputs")
    if type(receipt["elapsedSeconds"]) not in (float, int) or not 0 < receipt["elapsedSeconds"] <= POLICY["maximumPipelineSeconds"] or type(receipt["peakResidentBytes"]) is not int or not 0 < receipt["peakResidentBytes"] <= POLICY["maximumPeakResidentBytes"]:
        raise ValueError("Resource budget exceeded or unmeasured")
    return receipt


def manifest_for(plan, resource_path):
    verify_resources(resource_path, plan)
    manifest = {**plan, "resourceMeasurement": binding(Path(resource_path))}
    manifest["contentId"] = sha(canonical(manifest))
    return manifest


def build(resource_path):
    with build_lock(OUTPUT / ".build.lock"):
        payloads, plan, score = compute()
        manifest = manifest_for(plan, resource_path)
        target = OUTPUT / manifest["contentId"]
        for name, raw in payloads.items():
            write_immutable(target / name, raw)
        write_immutable(target / "manifest.json", data(manifest))
    return target, score


def verify(path, *, recompute=False):
    path = Path(path).resolve()
    if not path.is_relative_to(OUTPUT.resolve()) or path.parent != OUTPUT.resolve():
        raise ValueError("Corpus release is outside the admitted namespace")
    manifest = read(path / "manifest.json")
    if manifest["contentId"] != path.name or sha(canonical({key: value for key, value in manifest.items() if key != "contentId"})) != path.name:
        raise ValueError("Corpus release identity changed")
    if manifest["sourceFingerprints"] != fingerprints() or manifest["inputBindings"] != input_bindings() or manifest["policy"] != POLICY:
        raise ValueError("Corpus implementation, source or policy binding changed")
    if set(item["path"] for item in manifest["files"]) != FILE_NAMES or len(manifest["files"]) != len(FILE_NAMES):
        raise ValueError("Incomplete corpus shard inventory")
    for item in manifest["files"]:
        target = safe_file(path, item["path"])
        if target.stat().st_size != item["bytes"] or sha(target.read_bytes()) != item["sha256"]:
            raise ValueError("Corpus shard changed")
    resource_path = safe_file(AI, manifest["resourceMeasurement"]["path"])
    if binding(resource_path) != manifest["resourceMeasurement"]:
        raise ValueError("Resource measurement binding changed")
    plan = {key: value for key, value in manifest.items() if key not in ("contentId", "resourceMeasurement")}
    verify_resources(resource_path, plan)
    candidate.verify(PRIOR, recompute=False)
    if recompute:
        payloads, expected, _ = compute()
        if manifest_for(expected, resource_path) != manifest or any((path / name).read_bytes() != raw for name, raw in payloads.items()):
            raise ValueError("Corpus release does not reproduce")
    return manifest


def check_revocations(manifest):
    ledger = read(REVOCATIONS)
    if ledger.get("schemaVersion") != 1 or not isinstance(ledger.get("entries"), list):
        raise ValueError("Invalid input revocation ledger")
    source_ids = {row["sourceId"] for row in read(DECISION)["documents"]}
    for entry in ledger["entries"]:
        if set(entry) != {"kind", "id", "reason"} or entry["kind"] not in {"source", "corpus"} or not isinstance(entry["id"], str) or not entry["id"] or not isinstance(entry["reason"], str) or not entry["reason"]:
            raise ValueError("Invalid revocation entry")
        if entry["kind"] == "source" and entry["id"] in source_ids or entry["kind"] == "corpus" and entry["id"] == manifest["contentId"]:
            raise ValueError("CORPUS_INPUT_USE_REVOKED")


def require_use(path, usage, *, split=None):
    # Recompute before handing any data to a consumer. No caller approval flag,
    # transplanted manifest, mutated source, stale exclusion or token counter
    # can bypass these checks. No consumer iterator yields before this returns.
    if usage not in POLICY["allowedUses"]:
        raise ValueError("PILOT_INPUT_USE_NOT_APPROVED")
    roles = {"tokenizer-fitting-input": "train", "bounded-pilot-pretraining-input": "train", "validation-input": "validation", "test-input": "test"}
    expected = roles.get(usage)
    if expected is not None and split != expected or usage == "corpus-review" and split not in SPLITS:
        raise ValueError("RESERVED_SPLIT_USE_DENIED")
    manifest = verify(path, recompute=True)
    check_revocations(manifest)
    return {"manifest": manifest, "usage": usage, "split": split, "trainingRunApproved": False, "productionTrainingAllowed": False}


def read_inputs(path, usage, *, split):
    admission_receipt = require_use(path, usage, split=split)
    # Return the exact validated bytes as a tuple, not a later lazy disk read.
    # A concurrent edit after verification is caught by this second byte binding.
    manifest = admission_receipt["manifest"]
    raw = (Path(path) / (split + ".jsonl")).read_bytes()
    item = next(item for item in manifest["files"] if item["path"] == split + ".jsonl")
    if sha(raw) != item["sha256"] or len(raw) != item["bytes"]:
        raise ValueError("Corpus changed during input handoff")
    rows = tuple(json.loads(line) for line in raw.decode("utf-8").splitlines())
    if usage != "bounded-pilot-pretraining-input":
        return rows
    # This small scheduling experiment uses the declared proxy tokenizer. Gen2
    # training must recount under task 024 and obtain separate run acceptance.
    mixture_raw = (Path(path) / "mixture.json").read_bytes()
    mixture_item = next(item for item in manifest["files"] if item["path"] == "mixture.json")
    if sha(mixture_raw) != mixture_item["sha256"]:
        raise ValueError("Mixture changed during input handoff")
    schedule = json.loads(mixture_raw)["schedule"]
    expected = mixture.plan(list(rows), POLICY["plannedProxyTokenExposures"], weights=POLICY["mixtureWeights"], maximum_passes=POLICY["maximumDocumentPasses"])
    if schedule != expected["schedule"] or not expected["feasible"]:
        raise ValueError("Unapproved exposure schedule")
    tokenizer = VoltForgeTokenizer()
    tokenizer.load(AI / "model/tokenizers/vfdlm-byte-bpe-v1.1.0")
    by_id, cache, spans = {row["id"]: row for row in rows}, {}, []
    for item in schedule:
        identity = item["documentId"]
        if identity not in cache:
            cache[identity] = tokenizer.encode(by_id[identity]["text"], allowed_special=False)
        tokens = cache[identity][item["startToken"]:item["startToken"]+item["tokenCount"]]
        if len(tokens) != item["tokenCount"]:
            raise ValueError("Exposure slice exceeds verified input")
        spans.append({**item, "proxyTokenIds": tuple(tokens), "tokenizer": "vfdlm-byte-bpe-v1.1.0", "trainingRunApproved": False})
    return tuple(spans)
