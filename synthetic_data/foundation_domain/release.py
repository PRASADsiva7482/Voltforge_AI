"""Pre-reserved owned-source adapter and immutable, independently verified shards."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import unicodedata

from data_governance.ingestion.pipeline import write_immutable, build_lock
from data_governance.splitting.signatures import canonical, sha, features
from data_governance.splitting.partition import reserve_before_render, require_render_assignment, plan_partitions
from data_governance.splitting import release as partitions
from synthetic_data.verifiers import compiler_session, compiler_receipt, validate_compiler_receipt
from . import recipes, calculations, firmware, entities

ROOT = Path(__file__).resolve().parent
AI = ROOT.parents[1]
POLICY = json.loads((ROOT / "policy.v1.json").read_text(encoding="utf-8"))
FILES = ("__init__.py", "policy.v1.json", "recipes.py", "calculations.py", "firmware.py", "entities.py", "release.py")
SPLITS = ("train", "validation", "test", "quarantine")


def json_bytes(value): return (canonical(value) + "\n").encode("utf-8")


def lineage(family, board=None):
    # Queue math and queue firmware share an algorithm, even though authored as
    # separate problem/program recipes. Reserve their ancestry together.
    scenario = "reserved-slot-ring" if family in {"ring-occupancy", "serial-queue-drain"} else family
    return {"kind": "owned-synthetic", "sourceFamilyId": "owned-domain-v1:" + family,
            "documentFamilyId": "reference-project-v1:" + family, "repositoryId": "Voltforge_AI/synthetic_data/foundation_domain",
            "templateFamilyId": "owned-domain-v1:" + family, "scenarioFamilyId": "owned-domain-v1:" + scenario,
            "circuitFamilyId": "owned-domain-v1:" + scenario, "boardVariant": board}


def descriptors():
    return [lineage(name) for name in recipes.RECIPES] + [lineage(name, board) for name in firmware.PROGRAMS for board in firmware.BOARDS] + [lineage(row[0]) for row in entities.CASES]


def source_fingerprints():
    paths = {ROOT / name for name in FILES}
    paths.update(AI / name for name in ("synthetic_data/verifiers.py", "synthetic_data/toolchains.v1.json", "electronics_corpus/store.py", "electronics_corpus/catalog.v1.json"))
    for folder in ("data_governance/splitting", "data_governance/ingestion"):
        paths.update(path for path in (AI / folder).iterdir() if path.suffix in {".py", ".json"})
    catalog = json.loads((AI / "electronics_corpus/catalog.v1.json").read_text(encoding="utf-8"))
    paths.update(AI / "electronics_corpus" / row["path"] for row in catalog["packs"])
    return [{"path": path.relative_to(AI).as_posix(), "sha256": sha(path.read_bytes())} for path in sorted(paths)]


def reserve(output_root):
    reservation = reserve_before_render(descriptors())
    envelope = {"reservation": reservation, "sourceFingerprints": source_fingerprints(), "sourcePolicy": POLICY}
    identity = sha(canonical(envelope))
    path = output_root / "reservations" / (identity + ".json")
    write_immutable(path, json_bytes(envelope))
    return envelope, path


def render(reservation, compile_receipts, environment):
    rows, calculation_review = [], []
    reserved = {sha(canonical(row["lineage"])): row["split"] for row in reservation["assignments"]}
    environment_sha = sha(canonical(environment))

    def add(family, variant, kind, domain, question, answer, assumptions, evidence, board=None, **payload):
        ancestry = lineage(family, board)
        split = reserved[sha(canonical(ancestry))]
        assignment = require_render_assignment(reservation, ancestry, split)
        row = {"schemaVersion": 1, "recordId": f"vf-g2-domain-v1:{family}:{variant}", "familyId": family, "kind": kind,
               "domain": domain, "question": question, "answer": answer, "projectAssumptions": assumptions,
               "lineage": ancestry, "reservedSplit": split, "reservationDocumentId": assignment["documentId"],
               "sourceId": POLICY["sourceId"], "sourceEvidence": evidence, "trainingAllowed": False, **payload}
        text = "\n\n".join([question, answer, assumptions] + [payload[k] for k in ("brokenSource", "fixedSource") if k in payload])
        row["normalizedText"] = unicodedata.normalize("NFC", text.replace("\r\n", "\n"))
        rows.append(row)

    for family, (domain, project, question, explanation, unit, variants) in recipes.RECIPES.items():
        for index, parameters in enumerate(variants):
            value = recipes.answer(family, parameters)
            check = calculations.receipt(family, parameters, value, unit)
            check = {"familyId": family, "variant": index, "parameters": parameters, **check}
            check["receiptId"] = sha(canonical(check))
            calculation_review.append(check)
            add(family, str(index), "calculation", domain, question + " Supplied values: " + canonical(parameters),
                f"Result: {value:.12g} {unit}. " + explanation,
                "Authored reference project: " + project + ". Components are ideal mathematical elements with supplied values; no unspecified manufacturer rating is inferred.",
                {"recipePath": "synthetic_data/foundation_domain/recipes.py", "calculationReceiptId": check["receiptId"]},
                parameters=parameters, numericAnswer=value, unit=unit)
    expected_case_ids = {case["caseId"] for family in firmware.PROGRAMS for board in firmware.BOARDS for case in firmware.cases(family, board)}
    if set(compile_receipts) != expected_case_ids: raise ValueError("Missing or extra exact compiler cases")
    for family, (_source, _call, _broken, explanation, assumptions) in firmware.PROGRAMS.items():
        for board in firmware.BOARDS:
            cases = firmware.cases(family, board)
            checks = []
            for case in cases:
                stored = compile_receipts[case["caseId"]]
                if stored["environmentSha256"] != environment_sha: raise ValueError("Compiler environment binding mismatch")
                checks.append(validate_compiler_receipt(case, stored["receipt"]))
            target = firmware.exact_board(board)
            add(family, board, "compiler-repair-project", "firmware-debug",
                f"For {target['name']} ({target['variant']}), diagnose the supplied broken {family} reference program and provide its fixed source. Exact target: {target['fqbn']}.",
                explanation, assumptions, {"board": target, "compilerReceiptIds": [check["receiptId"] for check in checks], "environmentSha256": environment_sha},
                board=board, brokenSource=cases[0]["source"], fixedSource=cases[1]["source"], hardwareExecutionVerified=False)
    for family, kind, query, question, answer, expected in entities.CASES:
        check = entities.evidence(kind, query)
        if check["status"] != expected or check["genericRatingsKnown"]: raise ValueError("Exact entity uncertainty control failed")
        add(family, "0", "entity-uncertainty", "entity-grounding", question, answer,
            "Only the supplied label is available; no exact manufacturer variant or electrical measurement was provided.", check)
    return sorted(rows, key=lambda row: row["recordId"]), calculation_review


def descriptor(row):
    segments = [row["question"], row["answer"], row["projectAssumptions"]] + [row[key] for key in ("brokenSource", "fixedSource") if key in row]
    return {"id": row["recordId"], "recordId": row["recordId"], "lineage": row["lineage"],
            "rawSha256": sha(json_bytes(row)), "normalizedSha256": sha(row["normalizedText"]), "features": features(segments)}


def apply_reserved_splits(rows, guards):
    docs = [descriptor(row) for row in rows]
    plan = plan_partitions(docs, protected=guards)
    by_id = {row["recordId"]: row for row in rows}
    components = defaultdict(list)
    for decision in plan["assignments"]: components[decision["componentId"]].append(decision)
    # Post-render matching may connect previously unrelated recipes. Never
    # reassign them after seeing content: quarantine a cross-reservation bridge.
    for members in components.values():
        splits = {by_id[row["documentId"]]["reservedSplit"] for row in members}
        for decision in members:
            if len(splits) > 1:
                decision.update(split="quarantine", reason="post-render-cross-reservation-bridge")
            elif decision["split"] != "quarantine": decision["split"] = next(iter(splits))
            decision["reservedSplit"] = by_id[decision["documentId"]]["reservedSplit"]
    audit = partitions.cross_split_audit(docs, plan["assignments"], guards)
    return plan["assignments"], {"audit": audit, "protectedMatches": plan["protectedMatches"], "duplicateMatches": plan["duplicateMatches"]}


def compute(envelope, receipts, environment):
    if envelope != {"reservation": reserve_before_render(descriptors()), "sourceFingerprints": source_fingerprints(), "sourcePolicy": POLICY}:
        raise ValueError("Pre-render reservation/source policy mismatch")
    previous = AI / POLICY["previousPartitionManifest"]
    partitions.verify(previous.parent, recompute=True)
    guards, guard_evidence = partitions.protected_registry()
    rows, calculation_review = render(envelope["reservation"], receipts, environment)
    decisions, leakage = apply_reserved_splits(rows, guards)
    assignments = {row["documentId"]: row for row in decisions}
    included = [row for row in rows if assignments[row["recordId"]]["split"] != "quarantine"]
    coverage = {"candidateRecords": len(rows), "includedRecords": len(included),
                "candidateFamilies": len({row["familyId"] for row in rows}), "includedFamilies": len({row["familyId"] for row in included}),
                "candidateDomainCounts": dict(Counter(row["domain"] for row in rows)), "includedDomainCounts": dict(Counter(row["domain"] for row in included)),
                "splitCounts": {split: sum(row["split"] == split for row in decisions) for split in SPLITS},
                "reasonCounts": dict(Counter(row["reason"] for row in decisions)),
                "calculationReceipts": len(calculation_review), "calculationFamilies": len(recipes.RECIPES),
                "calculationNegativeControlsRejected": 3 * len(calculation_review),
                "firmwareFamilies": len(firmware.PROGRAMS), "compileExecutions": len(receipts),
                "compilerSuccesses": sum(row["receipt"]["observedSuccess"] for row in receipts.values()),
                "expectedCompilerFailures": sum(not row["receipt"]["observedSuccess"] for row in receipts.values()),
                "exactTargets": sorted({row["receipt"]["fqbn"] for row in receipts.values()}),
                "independentHumanReviewCompleted": False, "hardwareExecutionVerified": False,
                "trainingAllowed": False, "releasedGen2TrainingTokens": 0,
                "leakageAudit": leakage["audit"], "semanticLeakageCompletenessClaimed": False}
    coverage["domainsWithoutIncludedRecords"] = sorted(set(POLICY["requiredDomains"]) - set(coverage["includedDomainCounts"]))
    coverage["domainCorpusReady"] = not coverage["domainsWithoutIncludedRecords"]
    # Candidate coverage and eligibility are different acceptance facts. Keep
    # rejected compiler examples and their receipts as quarantine evidence;
    # never weaken the frozen matcher to make every domain survive this pilot.
    if set(POLICY["requiredDomains"]) - set(coverage["candidateDomainCounts"]): raise ValueError("Required authored domain is missing")
    if coverage["includedRecords"] == 0: raise ValueError("No independent domain records survive")
    if coverage["firmwareFamilies"] < POLICY["requiredFirmwareFamilies"] or coverage["exactTargets"] != sorted(POLICY["requiredExactTargets"]): raise ValueError("Incomplete firmware coverage")
    payloads = {"reservation.json": json_bytes(envelope), "source-manifest.json": json_bytes({"policy": POLICY, "files": envelope["sourceFingerprints"]}),
                "environment.json": json_bytes(environment), "compiler-receipts.json": json_bytes(receipts), "coverage.json": json_bytes(coverage),
                "calculation-review.jsonl": b"".join(json_bytes(row) for row in calculation_review), "decisions.jsonl": b"".join(json_bytes(row) for row in decisions),
                "leakage.json": json_bytes(leakage), "candidates.jsonl": b"".join(json_bytes(row) for row in rows)}
    for split in SPLITS:
        payloads[split + ".jsonl"] = b"".join(json_bytes(row) for row in rows if assignments[row["recordId"]]["split"] == split)
    basis = {"schemaVersion": 1, "releaseId": POLICY["id"], "sourceFingerprints": envelope["sourceFingerprints"],
             "previousPartitionManifest": {"path": POLICY["previousPartitionManifest"], "sha256": sha(previous.read_bytes())},
             "protectedRegistry": guard_evidence, "trainingAllowed": False, "hardwareExecutionVerified": False,
             "files": [{"path": name, "sha256": sha(data), "bytes": len(data)} for name, data in sorted(payloads.items())]}
    return payloads, basis | {"contentId": sha(canonical(basis))}, coverage


def build(output_root=None):
    output_root = Path(output_root or AI / "corpus/owned-domain/v1").resolve()
    with build_lock(AI / "corpus/.work/owned-domain-build.lock"):
        envelope, reservation_path = reserve(output_root)
        environment = firmware.build_environment()
        environment_sha = sha(canonical(environment))
        cache = AI / "corpus/.work/owned-domain-compiles" / environment_sha
        receipts = {}
        with compiler_session():
            for family in firmware.PROGRAMS:
                for board in firmware.BOARDS:
                    for case in firmware.cases(family, board):
                        path = cache / (sha(canonical(case)) + ".json")
                        if path.exists():
                            stored = json.loads(path.read_text(encoding="utf-8"))
                            if stored["environmentSha256"] != environment_sha: raise ValueError("Cached compiler environment mismatch")
                            validate_compiler_receipt(case, stored["receipt"])
                        else:
                            print("Compiling " + case["caseId"], flush=True)
                            stored = {"environmentSha256": environment_sha, "receipt": compiler_receipt(case)}
                            write_immutable(path, json_bytes(stored))
                        receipts[case["caseId"]] = stored
        if firmware.build_environment() != environment: raise ValueError("Compiler environment changed during build")
        payloads, manifest, coverage = compute(envelope, receipts, environment)
        target = output_root / manifest["contentId"]
        for name, data in payloads.items(): write_immutable(target / name, data)
        write_immutable(target / "manifest.json", json_bytes(manifest))
        return {"status": "passed", "releasePath": str(target), "manifestSha256": sha((target / "manifest.json").read_bytes()), "reservationPath": str(reservation_path), "coverage": coverage}


def verify(path, *, recompute=False, recompile=False):
    path = Path(path).resolve()
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    basis = {key: value for key, value in manifest.items() if key != "contentId"}
    if manifest.get("trainingAllowed") is not False or sha(canonical(basis)) != manifest["contentId"] or path.name != manifest["contentId"]: raise ValueError("Invalid domain release identity")
    if manifest["sourceFingerprints"] != source_fingerprints(): raise ValueError("Bound source implementation changed")
    for entry in manifest["files"]:
        target = (path / entry["path"]).resolve()
        if target.parent != path or sha(target.read_bytes()) != entry["sha256"] or target.stat().st_size != entry["bytes"]: raise ValueError("Domain release bytes changed")
    read = lambda name: json.loads((path / name).read_text(encoding="utf-8"))
    coverage = read("coverage.json")
    if recompute or recompile:
        environment = firmware.build_environment()
        if read("environment.json") != environment: raise ValueError("Exact local build environment differs")
        receipts = read("compiler-receipts.json")
        if recompile:
            with compiler_session():
                for family in firmware.PROGRAMS:
                    for board in firmware.BOARDS:
                        for case in firmware.cases(family, board):
                            fresh = compiler_receipt(case)
                            if fresh["receiptId"] != receipts[case["caseId"]]["receipt"]["receiptId"]: raise ValueError("Fresh compile identity changed")
        payloads, expected, coverage = compute(read("reservation.json"), receipts, environment)
        if expected != manifest or any((path / name).read_bytes() != data for name, data in payloads.items()): raise ValueError("Domain release does not reproduce")
    return {"status": "passed", "releasePath": str(path), "manifestSha256": sha((path / "manifest.json").read_bytes()), "recomputed": recompute or recompile, "freshCompileExecutions": len(firmware.PROGRAMS) * len(firmware.BOARDS) * 2 if recompile else 0, "coverage": coverage}


def require_use(path, use):
    result = verify(path, recompute=True)
    if use not in POLICY["useAllowed"]: raise ValueError("TRAINING_CORPUS_RELEASE_REQUIRED")
    return result
