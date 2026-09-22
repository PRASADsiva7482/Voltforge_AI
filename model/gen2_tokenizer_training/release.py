"""Immutable fitted candidates and the independently selected 0.1.0 release."""
import time
from pathlib import Path
from data_governance.ingestion.pipeline import build_lock, write_immutable
from model.gen2_tokenizer.codec import ByteBPE
from model.gen2_tokenizer.contract import canonical, sha, require, contract_document, template_document
from model.gen2_tokenizer.release import read_json
from .corpus import AI, ROOT, POLICY, reference_lineage
from .fitting import fit

CANDIDATES = AI / "model/tokenizers/gen2/candidates/v1"
RELEASES = AI / "model/tokenizers/gen2/releases"
VERSION_RECORD = RELEASES / "versions/0.1.0.json"
COMPARISONS = AI / "model/tokenizers/gen2/comparisons/v1"
FILES = ("contract.json", "template.json", "vocab.json", "merges.json")


def data(value):
    return canonical(value) + b"\n"


def binding(path):
    path = Path(path).resolve()
    return {"path": path.relative_to(AI).as_posix(), "sha256": sha(path.read_bytes()), "bytes": path.stat().st_size}


def fingerprints():
    names = [*sorted(ROOT.glob("*.py")), *sorted(ROOT.glob("*.json")), AI / "context_compiler/gen2_release.py", AI / "tools/train_gen2_tokenizer.py"]
    names += [AI / name for name in ("model/tokenizer.py", "model/gen2_tokenizer/codec.py", "model/gen2_tokenizer/contract.py", "model/gen2_tokenizer/release.py", "model/gen2_tokenizer/fixtures.py", "context_compiler/gen2.py", "model/gen1/config.py", "model/gen1/model.py", "data_governance/ingestion/pipeline.py", "tools/release_pilot_corpus.py")]
    return [binding(path) for path in sorted(names)]


def payloads(tokenizer):
    return {"contract.json": data(contract_document()), "template.json": data(template_document()), "vocab.json": data(tokenizer.vocab_document()), "merges.json": data(tokenizer.merges)}


def byte_binding(tokenizer):
    value = ByteBPE.binding(tokenizer)
    return {**value, "version": POLICY["version"], "releaseKind": "owned-corpus-tokenizer", "corpusManifestSha256": POLICY["corpusManifestSha256"]}


def identity(body):
    return {**body, "contentId": sha(canonical(body))}


def publish_bytes(root, body, blobs):
    manifest = identity(body)
    target = root / manifest["contentId"]
    for name, raw in sorted(blobs.items()):
        write_immutable(target / name, raw)
    write_immutable(target / "manifest.json", data(manifest))
    return target


def validate_measurement(value):
    require(set(value) == {"fitSeconds", "pipelineSeconds", "peakResidentBytes", "freshProcess", "trainingDocuments", "trainingBytes"}, "GEN2_FIT_MEASUREMENT_INVALID")
    require(value["freshProcess"] is True and type(value["fitSeconds"]) in (int, float) and 0 < value["fitSeconds"] <= POLICY["maximumFitSeconds"], "GEN2_FITTING_TIME_BUDGET")
    require(type(value["pipelineSeconds"]) in (int, float) and value["fitSeconds"] <= value["pipelineSeconds"] <= POLICY["maximumFitSeconds"]+120, "GEN2_FITTING_TIME_BUDGET")
    require(type(value["peakResidentBytes"]) is int and 0 < value["peakResidentBytes"] <= POLICY["maximumPeakResidentBytes"], "GEN2_FITTING_MEMORY_BUDGET")


def candidate_body(tokenizer, lineage, measurement):
    blobs = {**payloads(tokenizer), "measurement.json": data(measurement)}
    validate_measurement(measurement)
    require(measurement["trainingDocuments"] == lineage["documents"] and measurement["trainingBytes"] == lineage["bytes"], "GEN2_FIT_MEASUREMENT_LINEAGE_CHANGED")
    body = {"schemaVersion": 1, "taskId": "LLM-TASK-024", "releaseKind": "fitted-tokenizer-candidate", "version": "0.1.0-candidate.1", "policy": POLICY,
            "sourceFingerprints": fingerprints(), "corpusLineage": lineage, "binding": byte_binding(tokenizer), "targetVocabSize": tokenizer.target_vocab_size, "actualVocabSize": tokenizer.vocab_size,
            "targetReached": tokenizer.target_vocab_size == tokenizer.vocab_size, "servingAllowed": False, "trainingRunApproved": False,
            "files": [{"path": name, "sha256": sha(raw), "bytes": len(raw)} for name, raw in sorted(blobs.items())]}
    return body, blobs


def build_candidate(target):
    from tools.release_pilot_corpus import peak_resident
    started = time.perf_counter()
    tokenizer, lineage, seconds = fit(target)
    measurement = {"fitSeconds": seconds, "pipelineSeconds": time.perf_counter()-started, "peakResidentBytes": peak_resident(), "freshProcess": True, "trainingDocuments": lineage["documents"], "trainingBytes": lineage["bytes"]}
    body, blobs = candidate_body(tokenizer, lineage, measurement)
    with build_lock(CANDIDATES / ".build.lock"):
        path = publish_bytes(CANDIDATES, body, blobs)
    return path, measurement


def directory(path, root):
    path = Path(path).absolute()
    require(not any(item.is_symlink() or getattr(item, "is_junction", lambda: False)() for item in (path, *path.parents)), "GEN2_RELEASE_LINK_DENIED")
    require(path.resolve().parent == root.resolve(), "GEN2_RELEASE_NAMESPACE_INVALID")
    return path.resolve()


def check_manifest(path):
    manifest = read_json(path / "manifest.json", 1_048_576)
    require(manifest.get("contentId") == path.name and identity({key: value for key, value in manifest.items() if key != "contentId"}) == manifest, "GEN2_RELEASE_IDENTITY_CHANGED")
    return manifest


def check_blobs(path, blobs):
    require({item.name for item in path.iterdir()} == {*blobs, "manifest.json"}, "GEN2_RELEASE_FILE_INVENTORY_CHANGED")
    for name, raw in blobs.items():
        read_json(path / name)
        require((path / name).read_bytes() == raw, "GEN2_RELEASE_BYTES_CHANGED")


def verify_candidate(path, *, recompute=False):
    path = directory(path, CANDIDATES)
    manifest = check_manifest(path)
    tokenizer = ByteBPE(read_json(path / "merges.json"), target_vocab_size=manifest["targetVocabSize"])
    require(tokenizer.target_vocab_size in POLICY["candidateVocabularyTargets"], "GEN2_FITTING_TARGET_INVALID")
    measured = read_json(path / "measurement.json")
    body, blobs = candidate_body(tokenizer, reference_lineage(), measured)
    require(identity(body) == manifest, "GEN2_CANDIDATE_CONTRACT_CHANGED")
    check_blobs(path, blobs)
    if recompute:
        expected, _, _ = fit(tokenizer.target_vocab_size)
        require(expected.merges == tokenizer.merges, "GEN2_CANDIDATE_RETRAIN_MISMATCH")
    return tokenizer, manifest


def candidates():
    paths = sorted(CANDIDATES.glob("*/manifest.json"))
    require(len(paths) == 2, "GEN2_EXACT_CANDIDATE_PAIR_REQUIRED")
    result = [(*verify_candidate(path.parent), path.parent) for path in paths]
    require(sorted(tokenizer.target_vocab_size for tokenizer, _, _ in result) == POLICY["candidateVocabularyTargets"], "GEN2_EXACT_CANDIDATE_PAIR_REQUIRED")
    return sorted(result, key=lambda item: item[0].target_vocab_size)


def final_body(candidate_manifest, candidate_path, comparison_path, selected):
    return {"schemaVersion": 1, "taskId": "LLM-TASK-024", "releaseKind": "owned-tokenizer-release", "version": POLICY["version"], "binding": candidate_manifest["binding"],
            "targetVocabSize": candidate_manifest["targetVocabSize"], "actualVocabSize": candidate_manifest["actualVocabSize"], "corpusLineage": candidate_manifest["corpusLineage"],
            "candidateManifest": binding(candidate_path / "manifest.json"), "comparison": binding(comparison_path / "report.json"), "selection": selected,
            "sourceFingerprints": fingerprints(), "servingAllowed": False, "trainingRunApproved": False, "tokenizerInputUseAllowed": True, "retention": POLICY["retention"],
            "files": [item for item in candidate_manifest["files"] if item["path"] in FILES]}


def publish(comparison_path):
    from .evaluation import verify_comparison
    comparison_path = Path(comparison_path).resolve()
    report = verify_comparison(comparison_path, recompute=True)
    selected = next(item for item in report["candidates"] if item["targetVocabSize"] == report["selection"]["targetVocabSize"])
    candidate_path = AI / selected["manifest"]["path"]
    tokenizer, candidate_manifest = verify_candidate(candidate_path.parent)
    body = final_body(candidate_manifest, candidate_path.parent, comparison_path, report["selection"])
    with build_lock(RELEASES / ".build.lock"):
        path = publish_bytes(RELEASES, body, payloads(tokenizer))
        write_immutable(VERSION_RECORD, data({"tokenizerId": POLICY["tokenizerId"], "version": POLICY["version"], "releaseManifest": binding(path / "manifest.json")}))
    return path


class ReleasedTokenizer(ByteBPE):
    def __init__(self, merges, target, release_id):
        super().__init__(merges, target_vocab_size=target)
        self.release_id = release_id

    def binding(self):
        return {**byte_binding(self), "releaseId": self.release_id}


def load(path, *, recompute=False):
    from .evaluation import verify_comparison
    path = directory(path, RELEASES)
    manifest = check_manifest(path)
    version = read_json(VERSION_RECORD)
    require(version == {"tokenizerId": POLICY["tokenizerId"], "version": POLICY["version"], "releaseManifest": binding(path / "manifest.json")}, "GEN2_RELEASE_VERSION_NOT_REGISTERED")
    candidate_path = AI / manifest["candidateManifest"]["path"]
    require(binding(candidate_path) == manifest["candidateManifest"], "GEN2_RELEASE_CANDIDATE_CHANGED")
    tokenizer, candidate_manifest = verify_candidate(candidate_path.parent, recompute=recompute)
    comparison_path = AI / manifest["comparison"]["path"]
    require(binding(comparison_path) == manifest["comparison"], "GEN2_RELEASE_COMPARISON_CHANGED")
    report = verify_comparison(comparison_path.parent, recompute=recompute)
    require(report["selection"]["targetVocabSize"] == tokenizer.target_vocab_size, "GEN2_RELEASE_SELECTION_CHANGED")
    require(identity(final_body(candidate_manifest, candidate_path.parent, comparison_path.parent, report["selection"])) == manifest, "GEN2_RELEASE_CONTRACT_CHANGED")
    check_blobs(path, payloads(tokenizer))
    return ReleasedTokenizer(tokenizer.merges, tokenizer.target_vocab_size, manifest["contentId"]), manifest
