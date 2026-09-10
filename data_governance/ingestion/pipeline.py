"""Bounded offline ingestion with transactional resume and immutable shard output."""
from __future__ import annotations

import base64
from collections import Counter
from contextlib import closing, contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sqlite3
import tempfile
import time
import tracemalloc
import unicodedata

from data_governance.foundation import inventory as sources
from .normalization import POLICY, ROOT, Quarantine, normalize, reject_secrets

AI = ROOT.parents[1]
FIXTURES = AI / "tests/fixtures/ingestion/v1"
CODE_FILES = ("__init__.py", "normalization.py", "pdf_worker.py", "pipeline.py", "policy.v1.json")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def exclusive_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def parser_identity():
    import pypdf
    if importlib.metadata.version("pypdf") != POLICY["pypdfVersion"]:
        raise ValueError("Pinned corpus parser is unavailable")
    dependency = Path(pypdf.__file__).parent
    return {"version": POLICY["parserVersion"], "python": platform.python_version(), "unicode": unicodedata.unidata_version,
            "pypdf": POLICY["pypdfVersion"], "implementation": [{"path": name, "sha256": sources.file_hash(ROOT / name)} for name in CODE_FILES],
            "pypdfCodeSha256": digest(canonical([{ "path": path.relative_to(dependency).as_posix(), "sha256": sources.file_hash(path)}
                                                 for path in sorted(dependency.rglob("*.py"))]).encode()),
            "requirementsSha256": sources.file_hash(AI / "requirements-corpus.txt")}


def approved_plan():
    checked = sources.verify()
    inventory = read_json(sources.ROOT / "inventory.v1.json")
    units = []
    permissions = {}
    for unit in sorted(inventory["units"], key=lambda row: row["path"]):
        if unit["permissionStatus"] != "approved-existing-shard":
            continue
        for identity in unit["sourceIds"]:
            if identity not in permissions:
                permissions[identity] = sources.require_source_permission(identity, "training")
        units.append({"path": unit["path"], "sha256": unit["sha256"], "bytes": unit["bytes"], "unitId": unit["unitId"],
                      "sourceIds": unit["sourceIds"], "inputManifest": unit["manifest"],
                      "format": "jsonl", "mediaType": "application/x-vf-task+json", "encoding": "utf-8"})
    return {"schemaVersion": 1, "mode": "source-approved-staging", "inventorySha256": checked["inventorySha256"],
            "permissions": permissions, "units": units, "parser": parser_identity(), "trainingAllowed": False,
            "eligibleRecords": inventory["existingPool"]["recordFingerprints"]}


def fixture_plan(root=FIXTURES):
    catalog = read_json(root / "catalog.json")
    units = []
    for item in sorted(catalog["files"], key=lambda row: row["path"]):
        path = sources.safe_path(item["path"], root)
        if sources.file_hash(path) != item["sha256"]:
            raise ValueError("Fixture checksum mismatch")
        units.append({**item, "bytes": path.stat().st_size, "unitId": "fixture-" + digest(item["path"].encode()),
                      "sourceIds": ["vf-ingestion-test-fixtures-only"]})
    return {"schemaVersion": 1, "mode": "fixtures-only", "catalogSha256": sources.file_hash(root / "catalog.json"),
            "permissions": {}, "units": units, "parser": parser_identity(), "trainingAllowed": False, "eligibleRecords": []}


def validate_plan(plan, input_root):
    # No caller-supplied source ID or approved:true flag can authorize new bytes.
    if plan.get("mode") == "source-approved-staging":
        if input_root.resolve() != AI.resolve() or plan != approved_plan():
            raise ValueError("Source admission plan changed")
    elif plan.get("mode") == "fixtures-only":
        if plan != fixture_plan(input_root):
            raise ValueError("Fixture plan changed")
    else:
        raise ValueError("Unknown ingestion authority")
    if plan["trainingAllowed"] is not False:
        raise ValueError("Ingestion cannot authorize training")
    if len({row["path"] for row in plan["units"]}) != len(plan["units"]):
        raise ValueError("Duplicate input path")
    for unit in plan["units"]:
        path = sources.safe_path(unit["path"], input_root)
        if path.stat().st_size != unit["bytes"] or sources.file_hash(path) != unit["sha256"]:
            raise ValueError("Input bytes changed")
        if unit["format"] not in {"jsonl", "document"}:
            raise ValueError("Unknown input format")


def documents(path, unit):
    """Stream lines with a hard byte bound, hashing oversized lines without storing them."""
    limit = POLICY["maxPdfBytes"] if unit["mediaType"] == "application/pdf" else POLICY["maxDocumentBytes"]
    with path.open("rb") as stream:
        if unit["format"] == "document":
            if unit["bytes"] > limit:
                yield "document", None, unit["sha256"], unit["bytes"]
            else:
                raw = stream.read(limit + 1)
                yield "document", raw, digest(raw), len(raw)
            return
        number = 0
        while True:
            raw = stream.readline(limit + 1)
            if not raw:
                return
            number += 1
            size, fingerprint = len(raw), hashlib.sha256(raw)
            if len(raw) > limit:
                while not raw.endswith(b"\n"):
                    raw = stream.readline(limit + 1)
                    size += len(raw)
                    fingerprint.update(raw)
                    if not raw:
                        break
                raw = None
            yield "line:" + str(number), raw, fingerprint.hexdigest(), size


@contextmanager
def build_lock(path):
    """OS-released lock permits resume after process death, including on Windows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise ValueError("Ingestion build is already locked") from None
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def row_signature(raw_sha, raw, normalized, rejection):
    return digest(canonical([raw_sha, raw, normalized, rejection]).encode())


def stage_document(raw, raw_sha, unit, locator, plan, eligible):
    lineage = {"sourceIds": unit["sourceIds"], "inputPath": unit["path"], "inputSha256": unit["sha256"],
               "unitId": unit["unitId"], "locator": locator, "rawSha256": raw_sha,
               "permissionReceipts": [plan["permissions"][key] for key in unit["sourceIds"] if key in plan["permissions"]]}
    document_id = digest(canonical(lineage).encode())
    rejection = {"documentId": document_id, "lineage": lineage, "reason": None, "rawContentStored": False}
    try:
        if raw is None:
            raise Quarantine("document-too-large")
        extraction_bytes = raw
        if unit["mediaType"] == "application/x-vf-task+json":
            try:
                record = json.loads(raw.decode("utf-8"))
            except (UnicodeError, ValueError):
                raise Quarantine("invalid-task-json") from None
            # Scan the entire input, including metadata omitted from normalized text.
            reject_secrets(canonical(record))
            if not isinstance(record, dict):
                raise Quarantine("invalid-task-json")
            payload = sources.payload_text(record)
            if eligible.get(record.get("recordId")) != digest(payload.encode()):
                raise Quarantine("not-in-inventoried-training-pool")
            extraction_bytes = payload.encode()
            lineage["recordId"] = record["recordId"]
        text, quality = normalize(extraction_bytes, unit["mediaType"], unit["encoding"])
        raw_record = {"documentId": document_id, "lineage": lineage, "mediaType": unit["mediaType"],
                      "encoding": unit["encoding"], "rawBase64": base64.b64encode(raw).decode("ascii")}
        normalized = {"documentId": document_id, "lineage": lineage, "text": text, "textSha256": digest(text.encode()),
                      "parserVersion": POLICY["parserVersion"], "quality": quality,
                      "fixtureOnly": plan["mode"] == "fixtures-only", "trainingAllowed": False}
        return raw_record, normalized, None
    except Quarantine as error:
        rejection["reason"] = str(error)
        return None, None, rejection


def write_immutable(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("Immutable output collision")
        return
    # Readers only accept files listed by the final manifest. Publish via a
    # complete, fsynced temporary file; a crash cannot leave a partial shard.
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pending-", delete=False) as stream:
        temp = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    # Build lock serializes publication; never overwrite existing release bytes.
    if path.exists():
        raise ValueError("Immutable output appeared during publication")
    temp.rename(path)


def export_shards(connection, output):
    files = []
    for field, folder in (("raw", "raw"), ("normalized", "normalized"), ("rejection", "quarantine")):
        batch, size, number = [], 0, 0

        def flush():
            nonlocal batch, size, number
            if not batch:
                return
            data = b"".join(batch)
            name = f"{folder}/{number:06d}-{digest(data)}.jsonl"
            write_immutable(output / name, data)
            files.append({"path": name, "sha256": digest(data), "bytes": len(data), "records": len(batch), "kind": folder})
            batch, size, number = [], 0, number + 1

        for (value,) in connection.execute(f"SELECT {field} FROM documents WHERE {field} IS NOT NULL ORDER BY seq"):
            data = value.encode() + b"\n"
            if batch and (len(batch) >= POLICY["shardRecords"] or size + len(data) > POLICY["shardBytes"]):
                flush()
            batch.append(data)
            size += len(data)
        flush()
    return files


def build(plan, input_root, output_root, work_root, *, stop_after=None):
    input_root, output_root, work_root = map(Path, (input_root, output_root, work_root))
    started = time.perf_counter()
    tracemalloc.start()
    try:
        validate_plan(plan, input_root)
        plan_sha = digest(canonical(plan).encode())
        output = output_root / plan_sha
        if (output / "manifest.json").exists():
            checked = verify(output, plan=plan)
            return {"status": "already-complete", "releasePath": str(output), **checked}
        with build_lock(work_root / (plan_sha + ".lock")), closing(sqlite3.connect(work_root / (plan_sha + ".sqlite3"))) as db:
            db.execute(f"PRAGMA cache_size=-{POLICY['sqliteCacheKiB']}")
            db.execute("PRAGMA temp_store=FILE")
            db.execute("PRAGMA synchronous=FULL")
            db.execute("CREATE TABLE IF NOT EXISTS documents (seq INTEGER PRIMARY KEY, raw_sha TEXT, raw TEXT, normalized TEXT, rejection TEXT, signature TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS seen (text_sha TEXT PRIMARY KEY, document_id TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS binding (plan_sha TEXT PRIMARY KEY)")
            db.execute("INSERT OR IGNORE INTO binding VALUES (?)", (plan_sha,))
            if list(db.execute("SELECT plan_sha FROM binding")) != [(plan_sha,)]:
                raise ValueError("Resume journal plan mismatch")
            # The duplicate index is derived state. Rebuild it from checksum-
            # checked committed rows rather than trusting a stale side table.
            db.execute("DELETE FROM seen")
            previous = 0
            for seq, raw_sha, raw_value, normalized, rejection, signature in db.execute("SELECT * FROM documents ORDER BY seq"):
                if seq != previous + 1 or signature != row_signature(raw_sha, raw_value, normalized, rejection):
                    raise ValueError("Resume journal checksum or sequence mismatch")
                if normalized:
                    value = json.loads(normalized)
                    db.execute("INSERT INTO seen VALUES (?,?)", (value["textSha256"], value["documentId"]))
                previous = seq
            db.commit()
            eligible = {row["recordId"]: row["payloadSha256"] for row in plan["eligibleRecords"]}
            sequence, resumed, new = 0, 0, 0
            for unit in plan["units"]:
                path = sources.safe_path(unit["path"], input_root)
                for locator, raw, raw_sha, _ in documents(path, unit):
                    sequence += 1
                    old = db.execute("SELECT raw_sha,raw,normalized,rejection,signature FROM documents WHERE seq=?", (sequence,)).fetchone()
                    if old is not None:
                        if old[0] != raw_sha or row_signature(*old[:4]) != old[4]:
                            raise ValueError("Resume journal checksum mismatch")
                        resumed += 1
                        continue
                    raw_record, norm, rejection = stage_document(raw, raw_sha, unit, locator, plan, eligible)
                    if norm is not None:
                        prior = db.execute("SELECT document_id FROM seen WHERE text_sha=?", (norm["textSha256"],)).fetchone()
                        if prior:
                            rejection = {"documentId": norm["documentId"], "lineage": norm["lineage"], "reason": "exact-normalized-duplicate",
                                         "duplicateOf": prior[0], "rawContentStored": False}
                            raw_record, norm = None, None
                        else:
                            db.execute("INSERT INTO seen VALUES (?,?)", (norm["textSha256"], norm["documentId"]))
                    values = [canonical(row) if row is not None else None for row in (raw_record, norm, rejection)]
                    db.execute("INSERT INTO documents VALUES (?,?,?,?,?,?)", (sequence, raw_sha, *values, row_signature(raw_sha, *values)))
                    db.commit()
                    new += 1
                    if stop_after is not None and new >= stop_after:
                        return {"status": "interrupted-for-resume-test", "newDocuments": new, "resumedDocuments": resumed}
                # Inputs must remain unchanged throughout the streaming pass.
                if sources.file_hash(path) != unit["sha256"]:
                    raise ValueError("Input changed during ingestion")
            if db.execute("SELECT COUNT(*) FROM documents").fetchone()[0] != sequence:
                raise ValueError("Unexpected trailing resume journal records")
            validate_plan(plan, input_root)
            files = export_shards(db, output)
            reasons, quality = Counter(), Counter()
            for normalized, rejection in db.execute("SELECT normalized,rejection FROM documents ORDER BY seq"):
                if rejection:
                    reasons[json.loads(rejection)["reason"]] += 1
                else:
                    for key, value in json.loads(normalized)["quality"].items():
                        if type(value) is int:
                            quality[key] += value
            accepted = sum(row["records"] for row in files if row["kind"] == "normalized")
            manifest = {"schemaVersion": 1, "planSha256": plan_sha, "plan": plan, "status": "immutable-ingestion-staging",
                        "trainingAllowed": False, "gen2CorpusReleased": False, "networkAccessed": False,
                        "inputDocuments": sequence, "acceptedDocuments": accepted, "quarantinedDocuments": sequence - accepted,
                        "quarantineReasons": dict(sorted(reasons.items())), "qualityCounters": dict(sorted(quality.items())), "files": files,
                        "limits": ["Exact deduplication only; near-duplicate/family splits remain task 021.",
                                   "No Gen2 tokenizer/token count, corpus admission, training or runtime change.",
                                   "PDF reading-order/formula/OCR review is source-specific; no general extraction accuracy claim.",
                                   "Secret patterns are conservative heuristics, not comprehensive privacy clearance."]}
            write_immutable(output / "manifest.json", (canonical(manifest) + "\n").encode())
            checked = verify(output, plan=plan)
            elapsed = time.perf_counter() - started
            return {"status": "completed", "releasePath": str(output), **checked, "newDocuments": new, "resumedDocuments": resumed,
                    "performance": {"elapsedSeconds": elapsed, "inputBytes": sum(row["bytes"] for row in plan["units"]),
                                    "documentsPerSecond": sequence / elapsed, "pythonPeakAllocatedBytes": tracemalloc.get_traced_memory()[1],
                                    "memoryScope": "Parent Python traced allocations; excludes OS/native memory and PDF child processes.",
                                    "sqliteCacheKiB": POLICY["sqliteCacheKiB"], "pdfInputLimitBytes": POLICY["maxPdfBytes"]}}
    finally:
        tracemalloc.stop()


def verify(output, *, plan=None, recompute=False, input_root=None):
    output = Path(output)
    manifest = read_json(output / "manifest.json")
    bound = manifest["plan"]
    if manifest.get("trainingAllowed") is not False or manifest.get("gen2CorpusReleased") is not False or bound.get("trainingAllowed") is not False:
        raise ValueError("Ingestion release cannot authorize training")
    if manifest["planSha256"] != digest(canonical(bound).encode()) or output.name != manifest["planSha256"]:
        raise ValueError("Manifest plan hash mismatch")
    if plan is not None and bound != plan:
        raise ValueError("Release plan changed")
    counts = Counter()
    declared = {"manifest.json"}
    for row in manifest["files"]:
        path = sources.safe_path(row["path"], output)
        if row["path"] in declared or path.stat().st_size != row["bytes"] or sources.file_hash(path) != row["sha256"]:
            raise ValueError("Immutable shard hash mismatch")
        declared.add(row["path"])
        with path.open("rb") as stream:
            number = sum(1 for _ in stream)
        if number != row["records"] or row["kind"] not in {"raw", "normalized", "quarantine"}:
            raise ValueError("Shard count or kind mismatch")
        counts[row["kind"]] += number
    actual = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file() and not path.name.startswith(".pending-")}
    if actual != declared or counts["raw"] != counts["normalized"] or counts["normalized"] != manifest["acceptedDocuments"] or counts["quarantine"] != manifest["quarantinedDocuments"]:
        raise ValueError("Release files or totals mismatch")
    if counts["raw"] + counts["quarantine"] != manifest["inputDocuments"] or sum(manifest["quarantineReasons"].values()) != counts["quarantine"]:
        raise ValueError("Manifest counters mismatch")
    if recompute:
        root = Path(input_root) if input_root else AI if bound["mode"] == "source-approved-staging" else FIXTURES
        with tempfile.TemporaryDirectory(prefix="vf-ingestion-recompute-") as temp:
            rebuilt = build(bound, root, Path(temp) / "output", Path(temp) / "work")
            if (Path(rebuilt["releasePath"]) / "manifest.json").read_bytes() != (output / "manifest.json").read_bytes():
                raise ValueError("Ingestion is not byte-reproducible")
    return {"verification": "passed", "manifestSha256": sources.file_hash(output / "manifest.json"),
            "acceptedDocuments": manifest["acceptedDocuments"], "quarantinedDocuments": manifest["quarantinedDocuments"],
            "recomputed": recompute, "trainingAllowed": False}
