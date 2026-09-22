"""Bounded, pinned source acquisition. No redirects, crawling or code execution."""
import json
from pathlib import Path, PurePosixPath
import re
import urllib.request

from data_governance.ingestion.normalization import decode, reject_secrets, Quarantine
from data_governance.ingestion.pipeline import write_immutable, build_lock
from data_governance.splitting.signatures import canonical, sha

ROOT = Path(__file__).resolve().parent
AI = ROOT.parents[1]
POLICY = json.loads((ROOT / "policy.v1.json").read_text(encoding="utf-8"))
CATALOG = json.loads((ROOT / "sources.v1.json").read_text(encoding="utf-8"))


def data(value): return (canonical(value) + "\n").encode("utf-8")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl): raise ValueError("Acquisition redirect rejected")


def url(source, path):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", source["repository"]) or not re.fullmatch(r"[a-f0-9]{40}", source["revision"]): raise ValueError("Source is not pinned to an exact repository commit")
    if not isinstance(path, str) or not re.fullmatch(r"[A-Za-z0-9_./-]+", path) or PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts: raise ValueError("Unsafe source path")
    return "https://raw.githubusercontent.com/" + source["repository"] + "/" + source["revision"] + "/" + path


def fetch(address, maximum):
    request = urllib.request.Request(address, headers={"User-Agent": "VoltForge-explicit-corpus-acquisition-v1", "Accept-Encoding": "identity"})
    with urllib.request.build_opener(NoRedirect()).open(request, timeout=45) as response:
        if response.status != 200 or response.geturl() != address: raise ValueError("Unexpected source response")
        raw = response.read(maximum + 1)
    if len(raw) > maximum: raise ValueError("Source exceeds acquisition byte limit")
    return raw


def review_license(source, license_bytes, raw, path):
    license_text = decode(license_bytes, "utf-8")
    text = decode(raw, "utf-8")
    if source["licenseRequiredText"] not in license_text: raise ValueError("Pinned source license did not match reviewed proposal")
    conflicts = re.search(r"(?i)(all rights reserved|non[- ]commercial|no derivatives|SPDX-License-Identifier:\s*(?!MIT\b)[A-Za-z0-9.-]+|licensed under.*(?:GPL|Apache|BSD))", text)
    if conflicts: raise Quarantine("selected-file-license-exception-review-required")
    if source["licenseId"] == "MIT" and ("Permission is hereby granted, free of charge" not in text[:5000] or "Copyright" not in text[:5000]): raise Quarantine("selected-file-MIT-notice-missing")
    return {"licenseId": source["licenseId"], "licenseSha256": sha(license_bytes), "selectedFileSha256": sha(raw),
            "path": path, "scope": source["licenseScope"], "exceptionScan": "passed-bounded-lexical-review-not-legal-clearance",
            "trainingApproved": False, "noticesPreserved": True}


def acquire(output_root=None, *, fetcher=fetch):
    output_root = Path(output_root or AI / "corpus/acquisition/v1").resolve()
    plan = {"catalog": CATALOG, "policySha256": sha((ROOT / "policy.v1.json").read_bytes()), "acquirerSha256": sha(Path(__file__).read_bytes())}
    with build_lock(AI / "corpus/.work/pretraining-acquisition.lock"):
        plan_id = sha(canonical(plan))
        write_immutable(output_root / "plans" / (plan_id + ".json"), data(plan))
        cache = AI / "corpus/.work/pretraining-downloads" / plan_id
        total, files, denied = 0, [], []
        payloads = {}

        def obtain(source, name):
            nonlocal total
            address = url(source, name)
            cached = cache / (sha(address) + ".raw")
            raw = cached.read_bytes() if cached.exists() else fetcher(address, POLICY["maximumDocumentBytes"])
            if len(raw) > POLICY["maximumDocumentBytes"]: raise ValueError("Source exceeds acquisition byte limit")
            total += len(raw)
            if total > POLICY["maximumTotalAcquisitionBytes"]: raise ValueError("Total acquisition byte limit exceeded")
            try: reject_secrets(decode(raw, "utf-8"))
            except Quarantine as error:
                error.raw_sha256 = sha(raw)
                raise
            write_immutable(cached, raw)
            return raw, address

        for source in CATALOG["sources"]:
            license_bytes, license_url = obtain(source, source["licensePath"])
            license_path = source["sourceId"] + "/" + source["licensePath"]
            payloads[license_path] = license_bytes
            files.append({"sourceId": source["sourceId"], "kind": "license-evidence", "path": license_path, "upstreamPath": source["licensePath"], "url": license_url, "sha256": sha(license_bytes), "bytes": len(license_bytes)})
            for name in source["paths"]:
                print("Acquiring " + source["sourceId"] + ":" + name, flush=True)
                try: raw, address = obtain(source, name)
                except Quarantine as error:
                    denied.append({"sourceId": source["sourceId"], "path": name, "sha256": error.raw_sha256, "reason": str(error)})
                    continue
                try: license_review = review_license(source, license_bytes, raw, name)
                except Quarantine as error:
                    denied.append({"sourceId": source["sourceId"], "path": name, "sha256": sha(raw), "reason": str(error)})
                    continue
                path = source["sourceId"] + "/" + name
                payloads[path] = raw
                files.append({"sourceId": source["sourceId"], "kind": "candidate-source", "path": path, "upstreamPath": name, "url": address,
                              "sha256": sha(raw), "bytes": len(raw), "licenseReview": license_review})
        manifest = {"schemaVersion": 1, "plan": plan, "sourceAdmissionStatus": "candidate-review-only", "trainingAllowed": False,
                    "files": sorted(files, key=lambda x: x["path"]), "denied": denied, "acquiredBytes": total,
                    "networkPolicy": "explicit-pinned-source-download-no-live-retrieval-import", "codeExecuted": False}
        manifest["contentId"] = sha(canonical(manifest))
        target = output_root / manifest["contentId"]
        for name, raw in payloads.items(): write_immutable(target / name, raw)
        write_immutable(target / "manifest.json", data(manifest))
        return {"status": "passed-candidate-acquisition", "releasePath": str(target), "manifestSha256": sha((target / "manifest.json").read_bytes()),
                "sourceFiles": sum(row["kind"] == "candidate-source" for row in files), "deniedFiles": len(denied), "acquiredBytes": total, "trainingAllowed": False}


def verify(path):
    path = Path(path).resolve()
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest["contentId"] != path.name or sha(canonical({k: v for k, v in manifest.items() if k != "contentId"})) != path.name: raise ValueError("Acquisition identity changed")
    if manifest["trainingAllowed"] is not False or manifest["plan"]["catalog"] != CATALOG or manifest["plan"]["policySha256"] != sha((ROOT / "policy.v1.json").read_bytes()) or manifest["plan"]["acquirerSha256"] != sha(Path(__file__).read_bytes()): raise ValueError("Acquisition plan changed")
    sources = {s["sourceId"]: s for s in CATALOG["sources"]}
    seen = set()
    for row in manifest["files"]:
        target = (path / row["path"]).resolve()
        if not target.is_relative_to(path) or row["path"] in seen: raise ValueError("Unsafe/duplicate acquired path")
        seen.add(row["path"])
        raw = target.read_bytes()
        if sha(raw) != row["sha256"] or len(raw) != row["bytes"]: raise ValueError("Acquired source bytes changed")
        source = sources[row["sourceId"]]
        if row["url"] != url(source, row["upstreamPath"]): raise ValueError("Source URL/revision mismatch")
        if row["kind"] == "candidate-source":
            if row["upstreamPath"] not in source["paths"]: raise ValueError("Unselected file admitted")
            expected = review_license(source, (path / source["sourceId"] / source["licensePath"]).read_bytes(), raw, row["upstreamPath"])
            if expected != row["licenseReview"]: raise ValueError("Source license evidence changed")
    return manifest
