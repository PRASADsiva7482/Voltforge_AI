"""Generate the machine-readable VFAI-001 legacy artifact inventory."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np


AI_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = AI_ROOT / "model"
DEFAULT_OUTPUT = MODEL_ROOT / "artifact_inventory.json"
SCAN_ROOTS = (MODEL_ROOT / "artifacts", MODEL_ROOT / "configs")
EXTRA_PATHS = (
    MODEL_ROOT / "legacy" / "retired_scale_experiment" / "architecture_config.yaml",
)


CLASSIFICATIONS: dict[str, tuple[str, list[str]]] = {
    "model/artifacts/model_weights.npz": (
        "random-untrained",
        [
            "Tensor shapes match model/artifacts/config.json.",
            "model/train.py computes forward-pass loss but performs no optimizer or weight update before saving this file.",
            "The reported loss is therefore not evidence of trained weights.",
        ],
    ),
    "model/artifacts/model_meta.json": (
        "misleading-unverified-metadata",
        [
            "Claims TRAINED status for model_weights.npz.",
            "The producing model/train.py script does not update model weights.",
            "Metadata does not embed the model_config expected by the legacy inference loader.",
        ],
    ),
    "model/artifacts/config.json": (
        "legacy-random-experiment-config",
        ["Describes the tiny random-untrained model_weights.npz experiment."],
    ),
    "model/artifacts/model_1b_meta.json": (
        "legacy-architecture-simulation-metadata",
        [
            "Claims a 1.036B model and metrics without a corresponding weight checkpoint.",
            "The producing train_1b.py path simulates optimization/evaluation rather than training a release artifact.",
        ],
    ),
    "model/artifacts/voltforge_1b_config.json": (
        "legacy-architecture-manifest-without-weights",
        ["Declares intended export names and parameter count, but none of the declared 1B weight files exist."],
    ),
    "model/legacy/retired_scale_experiment/architecture_config.yaml": (
        "legacy-unexecuted-scale-config",
        ["A training architecture plan only; no compatible release checkpoint exists."],
    ),
    "model/configs/small.yaml": (
        "legacy-experimental-config",
        ["Describes a 6-layer experiment but is not bound to an immutable tokenizer/checkpoint manifest."],
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(AI_ROOT).as_posix()


def classify(path: Path) -> tuple[str, list[str]]:
    rel = relative(path)
    if rel in CLASSIFICATIONS:
        return CLASSIFICATIONS[rel]
    name = path.name
    if name.startswith("checkpoint_") and path.suffix == ".npz":
        return (
            "orphaned-incompatible-checkpoint",
            [
                "Checkpoint tensors imply vocabulary 6,023, width 256, and 6 layers.",
                "The retained tokenizer has vocabulary 4,096 and the retained config describes width 64 and 2 layers.",
                "No matching immutable tokenizer/config or validated metric record is present.",
            ],
        )
    if name in {"vocab.json", "merges.json", "tokenizer_config.json"}:
        return (
            "legacy-unapproved-tokenizer",
            [
                "Compatible with the tiny random-untrained model_weights.npz shape contract.",
                "Not bound to an approved immutable model manifest.",
            ],
        )
    if path.suffix == ".jsonl":
        return (
            "legacy-unreviewed-dataset",
            ["Provenance, licensing, leakage, deduplication, and label verification have not passed Gen1 data gates."],
        )
    return ("legacy-unclassified", ["Retained for audit; not approved for production loading."])


def json_details(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        return {"parseStatus": "invalid", "error": type(error).__name__}
    details: dict[str, Any] = {"parseStatus": "valid", "jsonType": type(value).__name__}
    if isinstance(value, dict):
        details["keys"] = sorted(value)
        claims = {
            key: value[key]
            for key in (
                "status",
                "engine",
                "model_name",
                "parameters",
                "total_parameters",
                "loss",
                "best_loss",
                "perplexity",
                "total_samples",
                "trained_at",
            )
            if key in value
        }
        if claims:
            details["claimedValues"] = claims
        if path.name == "vocab.json":
            ids = list(value.values())
            details.update(
                {
                    "vocabEntries": len(value),
                    "uniqueIds": len(set(ids)) if all(isinstance(item, int) for item in ids) else None,
                    "minimumId": min(ids) if ids and all(isinstance(item, int) for item in ids) else None,
                    "maximumId": max(ids) if ids and all(isinstance(item, int) for item in ids) else None,
                }
            )
    elif isinstance(value, list):
        details["entries"] = len(value)
    return details


def jsonl_details(path: Path) -> dict[str, Any]:
    records = 0
    invalid = 0
    key_counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                records += 1
                if isinstance(record, dict):
                    key_counts.update(record.keys())
            except (json.JSONDecodeError, UnicodeError):
                invalid += 1
    return {
        "records": records,
        "invalidRecords": invalid,
        "observedKeys": sorted(key_counts),
    }


def npz_details(path: Path) -> dict[str, Any]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            tensors = []
            parameter_count = 0
            for name in sorted(archive.files):
                tensor = archive[name]
                count = int(math.prod(tensor.shape))
                parameter_count += count
                tensors.append(
                    {
                        "name": name,
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                        "parameters": count,
                    }
                )
    except Exception as error:
        return {"parseStatus": "invalid", "error": type(error).__name__}

    by_name = {tensor["name"]: tensor for tensor in tensors}
    embedding = by_name.get("wte")
    layer_indexes = {
        int(match.group(1))
        for tensor in tensors
        if (match := re.match(r"l(\d+)_", tensor["name"]))
    }
    signature: dict[str, Any] = {}
    if embedding and len(embedding["shape"]) == 2:
        signature["vocabSize"] = embedding["shape"][0]
        signature["dModel"] = embedding["shape"][1]
    if layer_indexes:
        signature["nLayers"] = max(layer_indexes) + 1
    gate = by_name.get("l0_Wgate")
    if gate and len(gate["shape"]) == 2:
        signature["dFf"] = gate["shape"][1]
    return {
        "parseStatus": "valid",
        "tensorCount": len(tensors),
        "parameterCount": parameter_count,
        "inferredModelSignature": signature,
        "tensors": tensors,
    }


def inspect(path: Path) -> dict[str, Any]:
    classification, evidence = classify(path)
    stat = path.stat()
    if path.suffix == ".npz":
        kind = "model-checkpoint"
        details = npz_details(path)
    elif path.suffix == ".jsonl":
        kind = "training-dataset"
        details = jsonl_details(path)
    elif path.name in {"vocab.json", "merges.json", "tokenizer_config.json"}:
        kind = "tokenizer-artifact"
        details = json_details(path)
    elif path.suffix in {".yaml", ".yml"} or "config" in path.name:
        kind = "model-config"
        details = json_details(path) if path.suffix == ".json" else {"parseStatus": "not-parsed-yaml"}
    else:
        kind = "model-metadata"
        details = json_details(path) if path.suffix == ".json" else {}
    return {
        "path": relative(path),
        "kind": kind,
        "classification": classification,
        "productionEligible": False,
        "sizeBytes": stat.st_size,
        "modifiedAtUtc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "sha256": sha256(path),
        "evidence": evidence,
        "details": details,
    }


def build_inventory() -> dict[str, Any]:
    paths = sorted({
        *(
            path
            for root in SCAN_ROOTS
            if root.exists()
            for path in root.rglob("*")
            if path.is_file()
        ),
        *(path for path in EXTRA_PATHS if path.is_file()),
    })
    entries = [inspect(path) for path in paths]
    classifications = Counter(entry["classification"] for entry in entries)
    return {
        "schemaVersion": 1,
        "inventoryId": "vfai-001-legacy-artifact-inventory",
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "productionRegistry": "model/registry/active_model.json",
        "productionPolicy": {
            "legacyDirectoryScannedForActivation": False,
            "randomInitializationAllowedAtRuntime": False,
            "partialCheckpointLoadingAllowed": False,
            "checksumValidationRequired": True,
            "activeApprovedArtifact": None,
            "reasonCode": "NO_APPROVED_MODEL_ARTIFACT",
        },
        "summary": {
            "fileCount": len(entries),
            "totalBytes": sum(entry["sizeBytes"] for entry in entries),
            "productionEligibleCount": 0,
            "classificationCounts": dict(sorted(classifications.items())),
        },
        "productionLoadingPaths": [
            {
                "entryPoint": "model/infer.py:VoltForgeInferenceEngine",
                "resolver": "model/artifact_registry.py:resolve_active_artifact",
                "registry": "model/registry/active_model.json",
                "legacyDirectoryFallback": False,
            }
        ],
        "nonProductionLegacyPaths": [
            {
                "path": "model/train.py",
                "classification": "legacy-random-experiment-writer",
                "note": "Writes model_weights.npz without optimizer updates."
            },
            {
                "path": "model/train_chunks.py",
                "classification": "legacy-experimental-trainer",
                "note": "Produced orphaned checkpoints without immutable tokenizer/config/run manifests."
            },
            {
                "path": "model/legacy/retired_scale_experiment/simulated_training.py",
                "classification": "legacy-simulated-training-path",
                "note": "Does not produce a validated trained 1B checkpoint."
            },
            {
                "path": "model/reasoning_llm.py",
                "classification": "legacy-template-reasoning-path",
                "note": "Directly loads the legacy tokenizer for non-production template reasoning."
            }
        ],
        "artifacts": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    inventory = build_inventory()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "fileCount": inventory["summary"]["fileCount"],
                "totalBytes": inventory["summary"]["totalBytes"],
            }
        )
    )


if __name__ == "__main__":
    main()
