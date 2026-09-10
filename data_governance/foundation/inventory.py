"""Read-only source/use decisions and reproducible, content-free corpus accounting."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from data_governance import governance
from evaluation.foundation.suite import canonical, read, sha, check_training_candidates, verify_suite

AI = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent
REGISTRY = AI / "data_governance/source-registry.v1.json"
LOCKED_SOURCES = ("policy.v1.json", "proposed-sources.v1.json", "external-evidence.v1.json", "inventory.py")


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_path(relative, root=AI):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ":" in relative:
        raise ValueError("Inventory path must be relative")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Inventory path escapes its root")
    return path


def descriptor(path):
    return {"path": path.relative_to(AI).as_posix(), "sha256": file_hash(path), "bytes": path.stat().st_size}


def source_decisions(source, *, root=AI):
    policy = read(ROOT / "policy.v1.json")
    missing = [key for key in policy["requiredPermissionFields"] if not source.get(key)]
    if source.get("revision") in {"unknown", "per-request"}:
        missing.append("immutable-revision")
    if not re.fullmatch(r"[0-9a-f]{64}", str(source.get("checksum", ""))):
        missing.append("immutable-checksum")
    origin = source.get("origin", {})
    if not isinstance(origin, dict) or not origin.get("path") or not origin.get("evidence") or origin.get("owner") in {None, "unknown"}:
        missing.append("verified-origin")
    license_record = source.get("license", {})
    if not isinstance(license_record, dict) or not license_record.get("identifier") or not license_record.get("evidence"):
        missing.append("license-evidence")
    if source.get("kind") in policy["deniedKinds"]:
        missing.append("excluded-source-kind")
    results = {}
    for usage in policy["uses"]:
        # Source/library availability is separate from explicit permission for each use.
        result = governance.source_approval_decision(source, usage, verify_checksum=True, source_root=root)
        reasons = sorted(set(result["reasons"] + missing))
        results[usage] = {"allowed": not reasons, "reasons": reasons}
    return results


def payload_text(record):
    return canonical({key: record.get(key) for key in ("task", "input", "output")})


def summarize_pool(records, heldout_ids, tokenizer, *, heldout_records=(), foundation_checker=check_training_candidates):
    heldout_texts = {sha(payload_text(record).encode()) for record in (*records, *heldout_records) if record["recordId"] in heldout_ids}
    candidates, excluded = [], Counter()
    seen = set()
    for record in records:
        fingerprint = sha(payload_text(record).encode())
        if record["recordId"] in heldout_ids or fingerprint in heldout_texts:
            excluded["historical-held-out"] += 1
        elif fingerprint in seen:
            excluded["exact-payload-duplicate"] += 1
        else:
            seen.add(fingerprint)
            candidates.append(record)
    collisions = set()
    try:
        foundation_checker(candidates)
    except ValueError as error:
        prefix = "FOUNDATION_EVALUATION_LEAKAGE: "
        if not str(error).startswith(prefix):
            raise
        matches = json.loads(str(error)[len(prefix):])
        collisions = {item["record"] for item in matches}
        if any(type(index) is not int or not 0 <= index < len(candidates) for index in collisions):
            raise ValueError("Invalid foundation leakage result")
    rows = []
    for index, record in enumerate(candidates):
        if index in collisions:
            excluded["foundation-evaluation-leakage"] += 1
            continue
        text = payload_text(record)
        rows.append({"recordId": record["recordId"], "payloadSha256": sha(text.encode()), "task": record["task"],
                     "utf8Bytes": len(text.encode()), "proxyTokens": len(tokenizer.encode(text))})
    return {"inputRecords": len(records), "uniqueEligibleRecords": len(rows), "proxyTokens": sum(row["proxyTokens"] for row in rows),
            "utf8Bytes": sum(row["utf8Bytes"] for row in rows), "excluded": dict(excluded), "taskCounts": dict(Counter(row["task"] for row in rows)),
            "recordFingerprints": rows, "semanticDeduplicationClaimed": False, "gen2ReleaseReadyTokens": 0}


def token_gap(tokens, policy):
    if type(tokens) is not int or tokens < 0:
        raise ValueError("Unique token inventory must be a nonnegative integer")
    rows = []
    for stage, targets in policy["planningExposures"].items():
        for target in targets:
            rows.append({"stage": stage, "plannedExposures": target, "availableExistingPoolProxyTokens": tokens,
                         "additionalUniqueProxyTokensAtOnePass": max(0, target - tokens),
                         "passesOverExistingPoolToReachExposures": target / tokens if tokens else None,
                         "gen2ReleasedTokens": 0, "gen2ReleaseTokenGap": target,
                         "repeatSensitivity": [{"passes": repeat, "uniqueTokensUnchanged": tokens, "exposuresIfRepeated": tokens * repeat,
                                                "remainingExposures": max(0, target - tokens * repeat)} for repeat in policy["repeatSensitivity"]]})
    return rows


def build_inventory():
    from electronics_corpus.store import ElectronicsCorpus
    from model.tokenizer import VoltForgeTokenizer
    from synthetic_data.immutable_release import verify_all_releases, verify_current_release, resolve_tokenizer_shard, ImmutableReleaseError
    from task_schema.schema import validate_task_record
    from tools.verify_foundation_evaluation_binding import verify as verify_binding
    policy = read(ROOT / "policy.v1.json")
    verify_binding()
    registry = governance.load_source_registry(REGISTRY)
    governance.audit_current_corpora(write=False)
    source_rows = []
    for original in registry["sources"]:
        source = deepcopy(original)
        source["acquisitionMethod"] = "Existing repository source or retained historical input; verify its original registry evidence and bytes."
        uses = source_decisions(source)
        source_rows.append({"sourceId": source["sourceId"], "name": source["name"], "kind": source["kind"],
                           "permission": source, "permissionReceiptSha256": sha(canonical(original).encode()),
                           "authority": {**descriptor(REGISTRY), "selector": "sourceId=" + source["sourceId"]},
                           "useDecisions": uses, "approvalInheritedFromExistingRegistry": True,
                           "gen2CorpusReleased": False, "newRightsGranted": False})
    source_map = {row["sourceId"]: row for row in source_rows}
    if len(source_map) != len(source_rows):
        raise ValueError("Duplicate source identity")
    manifests = [read(path) for path in sorted((AI / "data_governance/manifests").glob("*.manifest.json"))]
    if len({row["path"] for row in manifests}) != len(manifests):
        raise ValueError("Duplicate shard manifest path")
    units, records = [], []
    for manifest in manifests:
        path = safe_path(manifest["path"])
        manifest_path = governance.default_manifest_path(path)
        matching_bytes = path.is_file() and file_hash(path) == manifest["sha256"]
        admitted = False
        reason = "source-or-shard-not-approved"
        try:
            if not matching_bytes:
                raise ValueError("SHARD_BYTES_MISMATCH")
            if not all(source_map.get(item["sourceId"], {}).get("useDecisions", {}).get("training", {}).get("allowed") for item in manifest["sources"]):
                raise ValueError("SOURCE_PERMISSION_NOT_ACCEPTED")
            governance.require_approved_shard(path)
            admitted = True
            reason = "existing-source-and-shard-permissions-verified"
        except (governance.DataGovernanceError, ValueError) as error:
            reason = getattr(error, "code", str(error))
        unit = {"unitId": manifest["shardId"], "kind": "task-shard", "path": manifest["path"],
                "recordCount": manifest["recordCount"], "bytes": path.stat().st_size if path.is_file() else None,
                "sha256": file_hash(path) if path.is_file() else None, "manifest": descriptor(manifest_path),
                "sourceIds": [row["sourceId"] for row in manifest["sources"]], "bytesMatchManifest": matching_bytes,
                "permissionStatus": "approved-existing-shard" if admitted else "quarantined", "reason": reason,
                "countedInExistingPool": admitted, "gen2ReleaseApproved": False, "rawContentStoredInInventory": False}
        units.append(unit)
        if admitted:
            shard_records = [validate_task_record(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(shard_records) != unit["recordCount"]:
                raise ValueError("Approved shard count mismatch")
            records.extend(shard_records)
    corpus = ElectronicsCorpus()
    curated = source_map[corpus.catalog["source"]["sourceId"]]
    if not curated["useDecisions"]["runtime-retrieval"]["allowed"] or not curated["useDecisions"]["training"]["allowed"]:
        raise ValueError("Curated source permission is unavailable")
    for pack in corpus.catalog["packs"]:
        path = AI / "electronics_corpus" / pack["path"]
        units.append({"unitId": pack["packId"], "kind": "curated-fact-pack", **descriptor(path), "recordCount": pack["recordCount"],
                      "sourceIds": [curated["sourceId"]], "permissionStatus": "approved-upstream-facts",
                      "manifest": descriptor(AI / "electronics_corpus/catalog.v1.json"), "countedInExistingPool": False,
                      "gen2ReleaseApproved": False, "reason": "Source permission verified; upstream facts are not added again to the derived synthetic training pool."})
    discovered = {path.relative_to(AI).as_posix() for pattern in policy["contentRoots"] for path in AI.glob(pattern) if path.is_file()}
    declared = {row["path"] for row in units}
    for relative in sorted(discovered - declared):
        path = safe_path(relative)
        units.append({"unitId": "unknown-" + file_hash(path)[:20], "kind": "unmanifested-content", **descriptor(path),
                      "recordCount": None, "sourceIds": [], "permissionStatus": "quarantined", "countedInExistingPool": False,
                      "gen2ReleaseApproved": False, "reason": "No checked source permission or matching governed manifest; zero token credit."})
    if declared - discovered:
        raise ValueError("A declared corpus unit is outside discovered data roots")
    current = verify_current_release()
    retained = []
    for release in verify_all_releases():
        for item in release["manifest"]["shards"]:
            retained.append({"releaseId": release["releaseId"], "releasePath": release["releasePath"],
                             "sourcePath": item["sourcePath"], "sha256": item["sha256"], "recordCount": item["recordCount"],
                             "isCurrent": release["releaseId"] == current["releaseId"], "countedAsAdditionalUniqueData": False})
    tokenizer = VoltForgeTokenizer()
    tokenizer_dir = AI / policy["tokens"]["tokenizerPath"]
    tokenizer.load(tokenizer_dir)
    heldout_ids, tokenizer_history = set(), []
    for path in sorted((AI / "model/tokenizers").glob("*/tokenizer_manifest.json")):
        manifest = read(path)
        split = manifest["lineage"]["split"]
        heldout_ids.update(split["evaluationRecordIds"])
        unresolved = []
        for item in manifest["lineage"]["shards"]:
            try:
                resolve_tokenizer_shard(item)
            except ImmutableReleaseError:
                unresolved.append(item["shardId"])
        tokenizer_history.append({"version": manifest["version"], "manifest": descriptor(path),
                                  "declaredTrainingRecords": split["trainingRecordCount"], "evaluationRecordCount": len(split["evaluationRecordIds"]),
                                  "unresolvedHistoricalShardIds": unresolved, "historicalBytesSubstituted": False})
    # Detect relabelled copies of held-out content across both retained releases.
    historical_heldout = []
    for release in verify_all_releases():
        for item in release["manifest"]["shards"]:
            path = safe_path(release["releasePath"] + "/content/" + item["sourcePath"])
            historical_heldout.extend(record for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
                                     for record in [json.loads(line)] if record["recordId"] in heldout_ids)
    pool = summarize_pool(records, heldout_ids, tokenizer, heldout_records=historical_heldout)
    pool["currentApprovedShardRecords"] = len(records)
    pool["retainedHeldOutGuardRecords"] = len(historical_heldout)
    proposals = read(ROOT / "proposed-sources.v1.json")["sources"]
    evidence = read(ROOT / "external-evidence.v1.json")
    if len(evidence["records"]) != len(proposals) or len({row["sourceId"] for row in proposals}) != len(proposals) or {row["sourceId"] for row in evidence["records"]} != {row["sourceId"] for row in proposals}:
        raise ValueError("Proposed source permission observations incomplete")
    for row in proposals:
        if row["acquired"] is not False or row["revision"] is not None or row["contentSha256"] is not None or any(row[key] is not False for key in ("trainingAllowed", "runtimeRetrievalAllowed", "redistributionAllowed")):
            raise ValueError("A proposed source cannot acquire permission through this inventory")
        observation = next(item for item in evidence["records"] if item["sourceId"] == row["sourceId"])
        if observation["url"] != row["licenseObservation"]["primaryUrl"] or observation["summarySha256"] != sha(row["licenseObservation"]["summary"].encode()):
            raise ValueError("Permission observation does not match its proposed source")
    excluded = [{"sourceId": name, "kind": kind, "trainingAllowed": False, "contentRead": False, "recordCount": None, "reason": reason}
                for name, kind, reason in (
                    ("vf-excluded-private-projects-chats", "private-user-project", "No consented Gen2 training-source release; do not inspect private projects or conversations."),
                    ("vf-excluded-secrets", "secret", "Credentials and service configuration are never a corpus."),
                    ("vf-excluded-teacher-outputs", "teacher-model-output", "No external model-output source decision; synthetic data defaults to owned deterministic authorship."),
                    ("vf-excluded-evaluation-material", "evaluation-material", "Task-018 fixtures and all historical held-out records are excluded from training and tokenizer fitting."))]
    inputs = [descriptor(AI / path) for path in (
        "data_governance/source-registry.v1.json", "data_governance/policy.v1.json", "data_governance/governance.py",
        "electronics_corpus/catalog.v1.json", "synthetic_data/releases/CURRENT.json", "model/tokenizer.py",
        "evaluation/foundation/manifest.v1.json", "foundation/evaluation-binding.v1.json")]
    inputs += [descriptor(path) for path in sorted((AI / "data_governance/manifests").glob("*.manifest.json"))]
    inputs += [descriptor(path) for path in sorted(tokenizer_dir.iterdir()) if path.is_file()]
    inputs += [descriptor(path) for path in sorted((AI / "model/tokenizers").glob("*/tokenizer_manifest.json"))]
    for source in source_rows:
        for relative in (source["permission"]["origin"].get("path"), source["permission"]["preprocessing"].get("path")):
            if relative and safe_path(relative).is_file():
                inputs.append(descriptor(safe_path(relative)))
    for release in verify_all_releases():
        inputs.append(descriptor(safe_path(release["releasePath"] + "/release-manifest.json")))
        for item in release["manifest"]["shards"]:
            inputs.append(descriptor(safe_path(release["releasePath"] + "/content/" + item["sourcePath"])))
    inputs = sorted({row["path"]: row for row in inputs}.values(), key=lambda row: row["path"])
    return {"schemaVersion": 1, "inventoryId": "vf-g2-source-inventory-v1", "taskId": "LLM-TASK-019", "status": "inventory-complete-no-gen2-corpus-release",
            "sources": source_rows, "proposals": proposals, "excludedSources": excluded, "units": units, "retainedCopies": retained,
            "sourceCounts": {"existing": len(source_rows), "approvedForExistingTrainingUse": sum(row["useDecisions"]["training"]["allowed"] for row in source_rows),
                             "notApprovedForTraining": sum(not row["useDecisions"]["training"]["allowed"] for row in source_rows), "proposedNotAdmitted": len(proposals), "excludedClasses": len(excluded)},
            "unitCounts": dict(Counter(row["permissionStatus"] for row in units)), "existingPool": pool,
            "tokenizer": {"path": policy["tokens"]["tokenizerPath"], "version": tokenizer.version, "gen2ExactCounts": False},
            "tokenizerHistory": tokenizer_history, "tokenGap": token_gap(pool["proxyTokens"], policy), "inputs": inputs,
            "foundationSuite": verify_suite(), "networkAccessed": False, "rawContentStored": False,
            "gen2TrainingAllowed": False, "modelReleaseApproved": False,
            "limits": ["Existing permission records are inherited repository declarations, not new rights grants or external legal opinions.",
                       "The existing pool is structured synthetic task data, not a diverse base-pretraining corpus. Upstream facts and retained copies add no independent token credit.",
                       "Proxy counts use the existing Gen1 tokenizer. The owned Gen2 tokenizer, family splits, semantic deduplication and normalized corpus release are later tasks.",
                       "Historical tokenizer v1.0.0 has unresolved source bytes; its known evaluation IDs remain excluded and no current bytes are substituted.",
                       "External sources are proposals only; permission observations and webpage hashes do not establish acquired corpus bytes or training approval.",
                       "Before training: evaluation-only custody, source/use review, task-020 ingestion and task-021 split/leakage gates remain mandatory."]}


def freeze():
    if (ROOT / "manifest.v1.json").exists() or (ROOT / "inventory.v1.json").exists():
        raise ValueError("Frozen inventory exists; create a new version, never overwrite")
    result = build_inventory()
    output = ROOT / "inventory.v1.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    manifest = {"schemaVersion": 1, "inventoryId": result["inventoryId"], "freezeStatus": "locked",
                "sources": [{"path": name, "sha256": file_hash(ROOT / name)} for name in LOCKED_SOURCES],
                "inventorySha256": file_hash(output), "inputs": result["inputs"], "trainingAllowed": False}
    (ROOT / "manifest.v1.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return verify()


def verify(*, recompute=False):
    verify_suite()
    manifest = read(ROOT / "manifest.v1.json")
    if (manifest.get("inventoryId"), manifest.get("freezeStatus"), manifest.get("trainingAllowed")) != ("vf-g2-source-inventory-v1", "locked", False):
        raise ValueError("Invalid frozen inventory identity")
    if len(manifest["sources"]) != len(LOCKED_SOURCES) or {row["path"] for row in manifest["sources"]} != set(LOCKED_SOURCES):
        raise ValueError("Missing frozen inventory source")
    for row in manifest["sources"]:
        if file_hash(safe_path(row["path"], ROOT)) != row["sha256"]:
            raise ValueError("Frozen inventory source hash mismatch")
    if file_hash(ROOT / "inventory.v1.json") != manifest["inventorySha256"]:
        raise ValueError("Frozen inventory receipt hash mismatch")
    for row in manifest["inputs"]:
        if file_hash(safe_path(row["path"])) != row["sha256"]:
            raise ValueError("Inventory input changed; create a new reviewed inventory version: " + row["path"])
    result = read(ROOT / "inventory.v1.json")
    for row in result["units"]:
        if file_hash(safe_path(row["path"])) != row["sha256"]:
            raise ValueError("Inventoried corpus bytes changed")
    if recompute and canonical(build_inventory()) != canonical(result):
        raise ValueError("Inventory or token accounting is not reproducible")
    return {"status": "passed", "inventorySha256": manifest["inventorySha256"], "sourceCounts": result["sourceCounts"],
            "unitCounts": result["unitCounts"], "uniqueEligibleRecords": result["existingPool"]["uniqueEligibleRecords"],
            "existingPoolProxyTokens": result["existingPool"]["proxyTokens"], "gen2ReleaseReadyTokens": 0,
            "recomputed": recompute, "trainingAllowed": False, "networkAccessed": False}


def require_source_permission(source_id, usage):
    """Task-020 staging prerequisite only; no corpus or training-run admission."""
    verify()
    if usage not in read(ROOT / "policy.v1.json")["uses"]:
        raise ValueError("SOURCE_USE_DENIED: unknown use")
    result = read(ROOT / "inventory.v1.json")
    source = next((row for row in result["sources"] if row["sourceId"] == source_id), None)
    if source is None or not source_decisions(source["permission"])[usage]["allowed"]:
        raise ValueError("SOURCE_USE_DENIED: source permission is absent or invalid")
    return {"sourceId": source_id, "usage": usage, "permissionReceiptSha256": source["permissionReceiptSha256"],
            "sourceRevision": source["permission"]["revision"], "sourceSha256": source["permission"]["checksum"],
            "sourcePermissionVerified": True, "gen2CorpusReleaseRequired": True, "trainingRunAllowed": False}
