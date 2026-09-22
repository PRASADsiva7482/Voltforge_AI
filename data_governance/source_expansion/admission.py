"""Verify the recorded exact-byte source decision; never accept caller approval flags."""
import json
from pathlib import Path
from . import acquisition

DECISION_PATH = Path(__file__).with_name("admission.v1.json")


def verify():
    decision = json.loads(DECISION_PATH.read_text(encoding="utf-8"))
    path = (acquisition.AI / decision["acquisitionPath"]).resolve()
    if not path.is_relative_to(acquisition.OUTPUT.resolve()):
        raise ValueError("Source decision is outside the declared acquisition namespace")
    if acquisition.sha((path / "manifest.json").read_bytes()) != decision["acquisitionManifestSha256"]:
        raise ValueError("Source decision evidence changed")
    packet = acquisition.verify(path)
    if decision["decision"] != "approved-exact-source-input-uses" or decision["corpusTrainingAllowed"] is not False or decision["modelReleaseApproved"] is not False or not decision["conditions"]:
        raise ValueError("Source decision scope changed")
    source_map = {row["sourceId"]: row for row in decision["sources"]}
    if len(source_map) != len(decision["sources"]) or set(source_map) != {row["sourceId"] for row in acquisition.CATALOG["sources"]}:
        raise ValueError("Source decision inventory changed")
    for source in acquisition.CATALOG["sources"]:
        uses = ["corpus-candidate-review", "validation-input"] if source["reservedSplit"] == "validation" else ["corpus-candidate-review", "tokenizer-fitting-input", "pretraining-input"]
        if source_map[source["sourceId"]]["allowedUses"] != uses:
            raise ValueError("Source decision violates reserved family use")
        actual = {row["upstreamPath"] for row in packet["files"] if row["sourceId"] == source["sourceId"] and row["kind"] == "candidate-source"}
        if actual != set(source["selectedPaths"]):
            raise ValueError("Recorded source review requires every selected file")
    return decision, packet


def require_source_input(source_id, name, usage):
    decision, packet = verify()
    source = next((row for row in decision["sources"] if row["sourceId"] == source_id), None)
    if not source or usage not in source["allowedUses"]:
        raise ValueError("SOURCE_INPUT_USE_NOT_APPROVED")
    row = next((row for row in packet["files"] if row["sourceId"] == source_id and row.get("upstreamPath") == name and row["kind"] == "candidate-source"), None)
    if row is None:
        raise ValueError("SOURCE_INPUT_BYTES_NOT_APPROVED")
    return row
