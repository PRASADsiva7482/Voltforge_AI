"""Acquire only declared Git blobs; retain evidence without granting corpus use."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import urllib.request

from data_governance.ingestion.normalization import decode, reject_secrets, Quarantine
from data_governance.ingestion.pipeline import build_lock, write_immutable
from data_governance.splitting.signatures import canonical, sha

ROOT = Path(__file__).resolve().parent
AI = ROOT.parents[1]
CATALOG = json.loads((ROOT / "catalog.v1.json").read_text(encoding="utf-8"))
POLICY = json.loads((ROOT / "policy.v1.json").read_text(encoding="utf-8"))
OUTPUT = AI / "corpus/source-expansion/acquisition-v1"
MIT_TERMS = (
    "permission is hereby granted, free of charge, to any person obtaining a copy",
    "of this software and associated documentation files",
    "to deal in the software without restriction",
    "including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies",
    "subject to the following conditions",
    "the above copyright notice and this permission notice shall be included in all copies or substantial portions of the software",
    "the software is provided",
    "without warranty of any kind, express or implied",
    "including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement",
    "in no event shall the authors or copyright holders be liable for any claim, damages or other liability",
    "whether in an action of contract, tort or otherwise",
    "arising from, out of or in connection with the software or the use or other dealings in the software",
)
MIT_GRANT = '''Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.'''


def data(value):
    return (canonical(value) + "\n").encode("utf-8")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Source expansion redirects are forbidden")


def fetch(address, maximum):
    request = urllib.request.Request(address, headers={"User-Agent": "VoltForge-pinned-source-expansion-v1", "Accept-Encoding": "identity"})
    with urllib.request.build_opener(NoRedirect()).open(request, timeout=30) as response:
        if response.status != 200 or response.geturl() != address:
            raise ValueError("Unexpected source expansion response")
        raw = response.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError("Source expansion byte limit exceeded")
    return raw


def safe_path(name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.\-/]+", name) or PurePosixPath(name).is_absolute() or any(part in {"..", "."} for part in name.split("/")) or "//" in name:
        raise ValueError("Unsafe selected source path")
    return name


def address(source, name=None):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", source["repository"]) or not re.fullmatch(r"[0-9a-f]{40}", source["revision"]):
        raise ValueError("Source expansion requires a pinned repository commit")
    if name is None:
        return f"https://api.github.com/repos/{source['repository']}/git/trees/{source['revision']}"
    return f"https://raw.githubusercontent.com/{source['repository']}/{source['revision']}/{safe_path(name)}"


def blob_sha(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()


def required_directories(source):
    result = {""}
    for name in [source["licensePath"], *source["scopeEvidencePaths"], *source["selectedPaths"]]:
        parts = safe_path(name).split("/")[:-1]
        result.update("/".join(parts[:end]) for end in range(1, len(parts) + 1))
    return sorted(result, key=lambda value: (value.count("/") + bool(value), value))


def verify_tree(source, tree):
    if tree.get("truncated") is not False or tree.get("sha") != source["revision"]:
        raise ValueError("Source tree is incomplete or revision changed")
    directories = required_directories(source)
    records = tree.get("directoryTrees", [])
    if [row["directory"] for row in records] != directories:
        raise ValueError("Incomplete ancestor directory evidence")
    expanded = {}
    flat = []
    for record in records:
        directory = record["directory"]
        expected_sha = source["revision"]
        if directory:
            parent, _, base = directory.rpartition("/")
            entries = {item["path"]: item for item in expanded[parent]["tree"]}
            entry = entries.get(base)
            if not entry or entry["type"] != "tree" or entry["mode"] != "040000":
                raise ValueError("Source ancestor is not a Git directory")
            expected_sha = entry["sha"]
        body = record["body"]
        if not re.fullmatch(r"[0-9a-f]{40}", expected_sha) or body.get("sha") != expected_sha or body.get("truncated") is not False:
            raise ValueError("Pinned directory tree identity changed")
        if len({row["path"] for row in body["tree"]}) != len(body["tree"]):
            raise ValueError("Duplicate directory entries")
        for item in body["tree"]:
            if "/" in item["path"] or item["path"] in {".", ".."}:
                raise ValueError("Malformed Git directory entry")
            flat.append({**item, "path": directory + "/" + item["path"] if directory else item["path"]})
        expanded[directory] = body
    if sorted(flat, key=lambda item: item["path"]) != tree.get("tree"):
        raise ValueError("Flattened tree differs from retained ancestor evidence")
    scope_license_paths(source, tree)


def obtain_tree(source, obtain):
    records, expanded, flat = [], {}, []
    for directory in required_directories(source):
        revision = source["revision"]
        if directory:
            parent, _, base = directory.rpartition("/")
            entry = next((row for row in expanded[parent]["tree"] if row["path"] == base), None)
            if not entry or entry["type"] != "tree" or entry["mode"] != "040000":
                raise ValueError("Source ancestor is not a regular Git directory")
            revision = entry["sha"]
        raw = obtain(f"https://api.github.com/repos/{source['repository']}/git/trees/{revision}", POLICY["maximumMetadataBytesPerRepository"])
        body = json.loads(raw)
        expanded[directory] = body
        records.append({"directory": directory, "body": body})
        flat.extend({**row, "path": directory + "/" + row["path"] if directory else row["path"]} for row in body["tree"])
    tree = {"schemaVersion": 1, "sha": source["revision"], "truncated": False, "directoryTrees": records, "tree": sorted(flat, key=lambda row: row["path"])}
    verify_tree(source, tree)
    raw = data(tree)
    if len(raw) > POLICY["maximumMetadataBytesPerRepository"]:
        raise ValueError("Combined ancestor evidence exceeds metadata budget")
    return raw, tree


def flattened(text):
    return " ".join(text.casefold().replace("*", " ").split())


def verify_mit(license_bytes):
    text = flattened(decode(license_bytes, "utf-8")).replace("non-infringement", "noninfringement")
    marker = "permission is hereby granted"
    if "copyright" not in text or marker not in text or text[text.index(marker):] != flattened(MIT_GRANT):
        raise ValueError("Complete MIT notice does not match the reviewed source policy")
    if re.search(r"non[- ]commercial|no derivatives|may not.*(?:train|learning)|permission.*required.*(?:ai|llm)", text):
        raise ValueError("Additional license restrictions require source-specific review")


def scope_license_paths(source, tree):
    selected = source["selectedPaths"]
    found = set()
    for item in tree["tree"]:
        name = item["path"]
        base = PurePosixPath(name).name.casefold()
        if not base.startswith(("license", "copying", "notice")):
            continue
        parent = PurePosixPath(name).parent.as_posix()
        if parent == "." or any(path.startswith(parent + "/") for path in selected):
            found.add(name)
    covered = {source["licensePath"], *source["scopeEvidencePaths"]}
    if found - covered:
        raise ValueError("Unreviewed ancestor license/notice applies to selected files")
    return sorted(found)


def review_selected(source, name, raw, license_raw):
    verify_mit(license_raw)
    text = decode(raw, "utf-8")
    reject_secrets(text)
    # These signals request review rather than inferring permission from a root
    # label. Do not interpret remote source comments as execution instructions.
    if re.search(r"(?i)(non[- ]commercial|all rights reserved|no derivatives|SPDX-License-Identifier:\s*(?!MIT\b)[A-Za-z0-9.-]+|licensed under[^\n]*(?:GPL|CC-BY|Creative Commons)|permission[^\n]*required[^\n]*(?:LLM|training))", text):
        raise Quarantine("selected-file-license-exception-review-required")
    return {"licenseId": "MIT", "licenseSha256": sha(license_raw), "selectedFileSha256": sha(raw),
            "completeGrantChecked": True, "selectedFileExceptionScan": "passed-bounded-review",
            "noticesRetained": True, "sourceApproval": "pending-recorded-packet-review", "trainingAllowed": False}


def source_fingerprints():
    paths = [ROOT / "catalog.v1.json", ROOT / "policy.v1.json", Path(__file__),
             AI / "data_governance/ingestion/normalization.py", AI / "data_governance/ingestion/pipeline.py"]
    return [{"path": path.relative_to(AI).as_posix(), "sha256": sha(path.read_bytes())} for path in paths]


def acquire(*, fetcher=fetch, output_root=None, cache_root=None):
    if sum(len(source["selectedPaths"]) for source in CATALOG["sources"]) > POLICY["maximumSelectedFiles"]:
        raise ValueError("Too many selected source files")
    plan = {"catalog": CATALOG, "sourceFingerprints": source_fingerprints()}
    plan_id = sha(canonical(plan))
    root = Path(output_root or OUTPUT).resolve()
    cache = Path(cache_root or AI / "corpus/.work/source-expansion" / plan_id)
    total = 0
    payloads, files, denied = {}, [], []

    def obtain(url, maximum):
        nonlocal total
        path = cache / (sha(url) + ".raw")
        raw = path.read_bytes() if path.exists() else fetcher(url, maximum)
        total += len(raw)
        if len(raw) > maximum or total > POLICY["maximumAcquisitionBytes"]:
            raise ValueError("Source expansion aggregate byte limit exceeded")
        reject_secrets(decode(raw, "utf-8"))
        write_immutable(path, raw)
        return raw

    with build_lock(root / ".build.lock"):
        for source in CATALOG["sources"]:
            print("Acquiring " + source["repository"], flush=True)
            tree_raw, tree = obtain_tree(source, obtain)
            entries = {row["path"]: row for row in tree["tree"]}
            if len(entries) != len(tree["tree"]):
                raise ValueError("Duplicate source tree paths")
            tree_name = source["sourceId"] + "/tree.json"
            payloads[tree_name] = tree_raw
            files.append({"sourceId": source["sourceId"], "kind": "tree-evidence", "path": tree_name, "url": address(source), "sha256": sha(tree_raw), "bytes": len(tree_raw)})
            source_files = {}
            planned = list(dict.fromkeys([source["licensePath"], *source["scopeEvidencePaths"], *source["selectedPaths"]]))
            for name in planned:
                entry = entries.get(name)
                if not entry or entry["type"] != "blob" or entry["mode"] != "100644":
                    raise ValueError("Selected source is missing or not a regular Git blob")
                raw = obtain(address(source, name), POLICY["maximumFileBytes"])
                if blob_sha(raw) != entry["sha"] or len(raw) != entry["size"]:
                    raise ValueError("Selected bytes do not match pinned Git blob")
                source_files[name] = raw
            license_raw = source_files[source["licensePath"]]
            verify_mit(license_raw)
            for name in planned:
                raw = source_files[name]
                selected = name in source["selectedPaths"]
                try:
                    review = review_selected(source, name, raw, license_raw) if selected else None
                except Quarantine as error:
                    denied.append({"sourceId": source["sourceId"], "path": name, "sha256": sha(raw), "reason": str(error)})
                    continue
                path = source["sourceId"] + "/" + name
                payloads[path] = raw
                files.append({"sourceId": source["sourceId"], "kind": "candidate-source" if selected else "permission-evidence", "upstreamPath": name,
                              "path": path, "url": address(source, name), "gitBlobSha1": entries[name]["sha"], "sha256": sha(raw), "bytes": len(raw), "review": review})
        manifest = {"schemaVersion": 1, "releaseKind": "source-expansion-review-packet", "plan": plan, "files": sorted(files, key=lambda row: row["path"]),
                    "denied": denied, "acquiredBytes": total, "trainingAllowed": False, "sourceApprovalInherited": False, "codeExecuted": False}
        manifest["contentId"] = sha(canonical(manifest))
        target = root / manifest["contentId"]
        for name, raw in payloads.items():
            write_immutable(target / name, raw)
        write_immutable(target / "manifest.json", data(manifest))
    return target


def verify(path):
    path = Path(path).resolve()
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest["contentId"] != path.name or sha(canonical({key: value for key, value in manifest.items() if key != "contentId"})) != path.name:
        raise ValueError("Expansion acquisition identity changed")
    if manifest["plan"] != {"catalog": CATALOG, "sourceFingerprints": source_fingerprints()} or manifest["trainingAllowed"] is not False:
        raise ValueError("Expansion acquisition plan changed")
    rows = manifest["files"]
    if len({row["path"] for row in rows}) != len(rows):
        raise ValueError("Duplicate acquired evidence path")
    source_map = {source["sourceId"]: source for source in CATALOG["sources"]}
    for row in rows:
        target = (path / safe_path(row["path"])).resolve()
        if not target.is_relative_to(path) or target.is_symlink():
            raise ValueError("Unsafe acquired evidence path")
        raw = target.read_bytes()
        if len(raw) != row["bytes"] or sha(raw) != row["sha256"]:
            raise ValueError("Acquired evidence bytes changed")
        source = source_map[row["sourceId"]]
        if row["kind"] == "tree-evidence":
            tree = json.loads(raw)
            if tree.get("truncated") is not False or tree.get("sha") != source["revision"] or row["url"] != address(source):
                raise ValueError("Pinned tree identity changed")
            verify_tree(source, tree)
        else:
            if row["url"] != address(source, row["upstreamPath"]) or blob_sha(raw) != row["gitBlobSha1"]:
                raise ValueError("Pinned file identity changed")
            tree = json.loads((path / source["sourceId"] / "tree.json").read_text(encoding="utf-8"))
            entry = next((item for item in tree["tree"] if item["path"] == row["upstreamPath"]), None)
            expected_kind = "candidate-source" if row["upstreamPath"] in source["selectedPaths"] else "permission-evidence"
            if not entry or entry["type"] != "blob" or entry["mode"] != "100644" or entry["sha"] != row["gitBlobSha1"] or entry["size"] != len(raw) or row["kind"] != expected_kind:
                raise ValueError("Acquired file differs from declared Git inventory or role")
            if row["kind"] == "candidate-source":
                expected = review_selected(source, row["upstreamPath"], raw, (path / source["sourceId"] / source["licensePath"]).read_bytes())
                if expected != row["review"]:
                    raise ValueError("Selected-file review changed")
    for source in CATALOG["sources"]:
        expected = {source["licensePath"], *source["scopeEvidencePaths"], *source["selectedPaths"]}
        actual = [row["upstreamPath"] for row in rows if row["sourceId"] == source["sourceId"] and row["kind"] != "tree-evidence"] + [row["path"] for row in manifest["denied"] if row["sourceId"] == source["sourceId"]]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError("Incomplete selected/denied source inventory")
    return manifest
