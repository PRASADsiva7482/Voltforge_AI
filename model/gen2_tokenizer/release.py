"""Content-addressed, write-once fixture artifacts. Production admission is closed."""
import json
from pathlib import Path

from data_governance.ingestion.pipeline import build_lock, write_immutable
from .codec import ByteBPE
from .contract import canonical, contract_document, require, sha, template_document, TokenizerError
from .fixtures import fit_fixture, lineage

AI = Path(__file__).resolve().parents[2]
ROOT = AI / "model/tokenizers/fixtures/gen2/v1"
FILENAMES = ("contract.json", "template.json", "merges.json", "vocab.json")
SOURCES = (
    "model/gen2_tokenizer/__init__.py", "model/gen2_tokenizer/contract.py",
    "model/gen2_tokenizer/codec.py", "model/gen2_tokenizer/fixtures.py",
    "model/gen2_tokenizer/release.py", "model/gen2_tokenizer/comparison.py",
    "context_compiler/gen2.py", "model/tokenizer.py", "foundation/contract.v1.json",
    "data_governance/ingestion/pipeline.py",
)


def fingerprints():
    return [{"path": name, "sha256": sha((AI / name).read_bytes())} for name in SOURCES]


def _unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "GEN2_ARTIFACT_DUPLICATE_KEY")
        result[key] = value
    return result


def read_json(path, maximum_bytes=33_554_432):
    try:
        require(path.is_file() and not path.is_symlink() and path.stat().st_size <= maximum_bytes, "GEN2_ARTIFACT_SIZE_OR_PATH_INVALID")
        with path.open("rb") as stream:
            raw = stream.read(maximum_bytes + 1)
        require(len(raw) <= maximum_bytes, "GEN2_ARTIFACT_SIZE_OR_PATH_INVALID")
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique, parse_constant=lambda _: (_ for _ in ()).throw(TokenizerError("GEN2_ARTIFACT_NUMBER_INVALID")))
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError):
        raise TokenizerError("GEN2_ARTIFACT_UNREADABLE") from None


def documents(tokenizer):
    payloads = {
        "contract.json": canonical(contract_document()) + b"\n",
        "template.json": canonical(template_document()) + b"\n",
        "merges.json": canonical(tokenizer.merges) + b"\n",
        "vocab.json": canonical(tokenizer.vocab_document()) + b"\n",
    }
    manifest = {
        "schemaVersion": 1, "taskId": "LLM-TASK-024", "releaseKind": "tokenizer-implementation-fixture-only",
        "trainingAllowed": False, "servingAllowed": False, "corpusRelease": None,
        "binding": tokenizer.binding(), "targetVocabSize": tokenizer.target_vocab_size,
        "actualVocabSize": tokenizer.vocab_size, "targetReached": tokenizer.vocab_size == tokenizer.target_vocab_size,
        "lineage": lineage(), "minimumPairFrequency": 2, "sourceFingerprints": fingerprints(),
        "files": [{"path": name, "sha256": sha(raw), "bytes": len(raw)} for name, raw in sorted(payloads.items())],
    }
    manifest["contentId"] = sha(canonical(manifest))
    return payloads, manifest


def build_fixture(target_vocab_size, *, output_root=None):
    tokenizer = fit_fixture(target_vocab_size)
    payloads, manifest = documents(tokenizer)
    root = Path(output_root or ROOT).resolve()
    with build_lock(root / ".build.lock"):
        target = root / manifest["contentId"]
        for name, raw in payloads.items():
            write_immutable(target / name, raw)
        # Manifest last: incomplete writes are never accepted as a release.
        write_immutable(target / "manifest.json", canonical(manifest) + b"\n")
    return target


def verify_fixture(path, *, recompute=False):
    path = Path(path).resolve()
    manifest = read_json(path / "manifest.json", 1_048_576)
    require(isinstance(manifest, dict), "GEN2_MANIFEST_INVALID")
    require(manifest.get("contentId") == path.name and sha(canonical({key: value for key, value in manifest.items() if key != "contentId"})) == path.name, "GEN2_MANIFEST_IDENTITY_CHANGED")
    # No arbitrary manifest paths are traversed. Rebuild all expected metadata,
    # including exact contract, lineage, byte inventory and admission denial.
    merges = read_json(path / "merges.json")
    require(isinstance(merges, list), "GEN2_MERGES_INVALID")
    tokenizer = ByteBPE(merges, target_vocab_size=manifest.get("targetVocabSize"))
    payloads, expected = documents(tokenizer)
    require(canonical(expected) == canonical(manifest), "GEN2_MANIFEST_CONTRACT_CHANGED")
    for name in FILENAMES:
        value = read_json(path / name)
        require(canonical(value) + b"\n" == payloads[name], "GEN2_ARTIFACT_CONTRACT_CHANGED")
        require((path / name).read_bytes() == payloads[name], "GEN2_ARTIFACT_BYTES_CHANGED")
    if recompute:
        fitted = fit_fixture(tokenizer.target_vocab_size)
        require(fitted.merges == tokenizer.merges, "GEN2_FIXTURE_DOES_NOT_REPRODUCE")
    return tokenizer, manifest


def require_corpus_fitting(path):
    """Reject before opening any corpus shard or fitting any token.

    Negative admission never needs to parse quarantined data. Positive flags in
    user-authored JSON cannot supply rights, split or approval evidence. Task
    023's future admitted format needs an independently verified adapter here.
    """
    manifest = read_json(Path(path).resolve() / "manifest.json", 1_048_576)
    require(isinstance(manifest, dict), "GEN2_CORPUS_MANIFEST_INVALID")
    require(manifest.get("trainingAllowed") is True and manifest.get("releaseKind") != "pretraining-candidate-only", "GEN2_CORPUS_NOT_ADMITTED")
    raise TokenizerError("GEN2_CORPUS_ADMISSION_ADAPTER_MISSING")
