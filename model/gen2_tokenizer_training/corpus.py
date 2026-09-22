"""Only task023's verified train handoff can supply merge-fitting bytes."""
from pathlib import Path
import json
from data_governance.pilot_corpus import release as inputs
from model.gen2_tokenizer.contract import canonical, sha, require

AI = inputs.AI
ROOT = Path(__file__).resolve().parent
POLICY = inputs.read(ROOT / "policy.v1.json")
CORPUS = AI / POLICY["corpusPath"]


def fitting_inputs(path=CORPUS):
    path = Path(path).resolve()
    require(path == CORPUS.resolve(), "GEN2_TRAINING_CORPUS_NOT_SELECTED")
    require(sha((path / "manifest.json").read_bytes()) == POLICY["corpusManifestSha256"], "GEN2_TRAINING_CORPUS_BINDING_CHANGED")
    rows = inputs.read_inputs(path, "tokenizer-fitting-input", split="train")
    require(0 < len(rows) <= POLICY["maximumTrainDocuments"], "GEN2_FITTING_DOCUMENT_BUDGET")
    raw = tuple(row["text"].encode("utf-8", "surrogatepass") for row in rows)
    require(sum(map(len, raw)) <= POLICY["maximumTrainBytes"], "GEN2_FITTING_BYTE_BUDGET")
    require(all(row["split"] == "train" and not row["inheritedExclusion"] for row in rows), "GEN2_HELDOUT_FITTING_DENIED")
    lineage = lineage_for(rows, raw)
    return raw, lineage


def lineage_for(rows, raw):
    return {"corpusManifest": inputs.binding(CORPUS / "manifest.json"), "split": "train", "shard": inputs.binding(CORPUS / "train.jsonl"),
               "fittingDocuments": [{"documentId": row["id"], "sourceId": row["sourceId"], "sha256": sha(value), "bytes": len(value)} for row, value in zip(rows, raw, strict=True)],
               "documents": len(rows), "bytes": sum(map(len, raw)), "validationOrTestFitted": False, "pretrainedTokenizerUsed": False}


def validation_inputs():
    # Only a frozen pair of candidate artifacts may invoke this in evaluation.
    return inputs.read_inputs(CORPUS, "validation-input", split="validation")


def reference_lineage():
    # Exact pinned manifest plus all shard/source hashes are sufficient for
    # read-only lineage checks. Every new fit still recomputes input admission.
    require(sha((CORPUS / "manifest.json").read_bytes()) == POLICY["corpusManifestSha256"], "GEN2_TRAINING_CORPUS_BINDING_CHANGED")
    manifest = inputs.verify(CORPUS, recompute=False)
    inputs.check_revocations(manifest)
    raw_shard = (CORPUS / "train.jsonl").read_bytes()
    item = next(row for row in manifest["files"] if row["path"] == "train.jsonl")
    require(sha(raw_shard) == item["sha256"], "GEN2_TRAINING_SHARD_CHANGED")
    rows = tuple(json.loads(line) for line in raw_shard.decode("utf-8").splitlines())
    return lineage_for(rows, tuple(row["text"].encode("utf-8", "surrogatepass") for row in rows))
