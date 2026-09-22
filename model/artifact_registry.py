"""Fail-closed registry for production VoltForge model artifacts.

Production inference must resolve one explicitly approved artifact through
``model/registry/active_model.json``.  The legacy ``model/artifacts`` directory
is intentionally never scanned as a source of an active model.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from model.identity import empty_identity_health, parse_artifact_id
from model.tokenizer import SPECIAL_TOKEN_TO_ID, TOKENIZER_ALGORITHM


MODEL_ROOT = Path(__file__).resolve().parent
DEFAULT_REGISTRY_PATH = MODEL_ROOT / "registry" / "active_model.json"
SUPPORTED_SCHEMA_VERSIONS = frozenset({1, 2})
SUPPORTED_RUNTIME = "numpy-transformer-v1"
REQUIRED_FILES = (
    "config",
    "metadata",
    "weights",
    "vocab",
    "merges",
    "tokenizerConfig",
    "tokenizerManifest",
)


class ArtifactRegistryError(RuntimeError):
    """A precise, public-safe model artifact validation failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ResolvedArtifact:
    artifact_id: str
    runtime: str
    root: Path
    files: Mapping[str, Path]
    config: Mapping[str, Any]
    parameter_count: int
    quantization: str
    checksum: str
    release_status: str
    data_lineage: Mapping[str, Any]
    registry_path: Path

    def health(self) -> dict[str, Any]:
        health = self.identity.health_fields()
        health.update({
            "ready": True,
            "state": "ready",
            "code": "MODEL_ARTIFACT_READY",
            "message": "An approved local VoltForge model artifact passed validation.",
            "runtime": self.runtime,
            "parameterCount": self.parameter_count,
            "contextLength": int(self.config["context_length"]),
            "vocabSize": int(self.config["vocab_size"]),
            "quantization": self.quantization,
            "checksum": self.checksum,
            "releaseStatus": self.release_status,
            "dataSourceCount": len(self.data_lineage["sourceIds"]),
            "trainingShardCount": len(self.data_lineage["shardIds"]),
            "registryPath": str(self.registry_path),
        })
        return health

    @property
    def identity(self):
        return parse_artifact_id(self.artifact_id)


def _read_json(path: Path, code: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ArtifactRegistryError(code, f"Required artifact file is missing: {path.name}") from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactRegistryError(code, f"Artifact JSON is invalid: {path.name}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_beneath(base: Path, relative: str, code: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ArtifactRegistryError(code, "Artifact manifest paths must be relative.")
    resolved = (base / candidate).resolve()
    try:
        resolved.relative_to(base.resolve())
    except ValueError as error:
        raise ArtifactRegistryError(code, "Artifact manifest path escapes its approved root.") from error
    return resolved


def expected_numpy_tensor_shapes(config: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    try:
        vocab_size = int(config["vocab_size"])
        context_length = int(config["context_length"])
        d_model = int(config["d_model"])
        n_heads = int(config["n_heads"])
        n_layers = int(config["n_layers"])
        d_ff = int(config["d_ff"])
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactRegistryError(
            "MODEL_CONFIG_INVALID",
            "Model config must contain integer vocab_size, context_length, d_model, n_heads, n_layers, and d_ff values.",
        ) from error

    if min(vocab_size, context_length, d_model, n_heads, n_layers, d_ff) <= 0:
        raise ArtifactRegistryError("MODEL_CONFIG_INVALID", "Model dimensions must be positive integers.")
    if d_model % n_heads != 0:
        raise ArtifactRegistryError(
            "MODEL_CONFIG_INVALID", "Model d_model must be divisible by n_heads."
        )

    shapes: dict[str, tuple[int, ...]] = {"wte": (vocab_size, d_model)}
    for layer in range(n_layers):
        prefix = f"l{layer}_"
        shapes.update(
            {
                f"{prefix}rms1": (d_model,),
                f"{prefix}Wq": (d_model, d_model),
                f"{prefix}Wk": (d_model, d_model),
                f"{prefix}Wv": (d_model, d_model),
                f"{prefix}Wo": (d_model, d_model),
                f"{prefix}rms2": (d_model,),
                f"{prefix}Wgate": (d_model, d_ff),
                f"{prefix}Wup": (d_model, d_ff),
                f"{prefix}Wdown": (d_ff, d_model),
            }
        )
    shapes["rms_f"] = (d_model,)
    shapes["lm_head"] = (d_model, vocab_size)
    return shapes


def _validate_tokenizer(
    files: Mapping[str, Path],
    config: Mapping[str, Any],
    data_lineage: Mapping[str, Any],
) -> None:
    vocab = _read_json(files["vocab"], "TOKENIZER_VOCAB_INVALID")
    merges = _read_json(files["merges"], "TOKENIZER_MERGES_INVALID")
    tokenizer_config = _read_json(files["tokenizerConfig"], "TOKENIZER_CONFIG_INVALID")
    tokenizer_manifest = _read_json(files["tokenizerManifest"], "TOKENIZER_MANIFEST_INVALID")
    if not isinstance(vocab, dict) or not vocab:
        raise ArtifactRegistryError("TOKENIZER_VOCAB_INVALID", "Tokenizer vocabulary must be a non-empty object.")
    ids = list(vocab.values())
    if any(not isinstance(token_id, int) for token_id in ids) or len(ids) != len(set(ids)):
        raise ArtifactRegistryError(
            "TOKENIZER_VOCAB_INVALID", "Tokenizer IDs must be unique integers."
        )
    if not isinstance(merges, list):
        raise ArtifactRegistryError("TOKENIZER_MERGES_INVALID", "Tokenizer merges must be a JSON array.")
    if not isinstance(tokenizer_manifest, dict):
        raise ArtifactRegistryError("TOKENIZER_MANIFEST_INVALID", "Tokenizer manifest must be an object.")
    declared_artifact_hash = tokenizer_manifest.get("artifactSha256")
    manifest_payload = {
        key: value for key, value in tokenizer_manifest.items() if key != "artifactSha256"
    }
    actual_artifact_hash = hashlib.sha256(
        json.dumps(
            manifest_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    if declared_artifact_hash != actual_artifact_hash:
        raise ArtifactRegistryError(
            "TOKENIZER_MANIFEST_CHECKSUM_MISMATCH", "Tokenizer manifest integrity checksum is invalid."
        )
    if (
        tokenizer_manifest.get("releaseStatus") != "approved"
        or tokenizer_manifest.get("algorithm") != TOKENIZER_ALGORITHM
        or tokenizer_config.get("algorithm") != TOKENIZER_ALGORITHM
    ):
        raise ArtifactRegistryError(
            "TOKENIZER_NOT_APPROVED", "Model artifacts require an approved VoltForge byte-BPE tokenizer."
        )
    if tokenizer_config.get("specialTokenIds") != SPECIAL_TOKEN_TO_ID:
        raise ArtifactRegistryError(
            "TOKENIZER_SPECIAL_CONTRACT_MISMATCH", "Tokenizer special-token IDs are incompatible."
        )
    tokenizer_lineage = tokenizer_manifest.get("lineage")
    if not isinstance(tokenizer_lineage, dict) or (
        sorted(tokenizer_lineage.get("sourceIds") or []) != sorted(data_lineage["sourceIds"])
        or sorted(tokenizer_lineage.get("shardIds") or []) != sorted(data_lineage["shardIds"])
    ):
        raise ArtifactRegistryError(
            "TOKENIZER_LINEAGE_MISMATCH",
            "Model and tokenizer must declare the same governed source and shard lineage.",
        )
    internal_files = tokenizer_manifest.get("files")
    if not isinstance(internal_files, dict):
        raise ArtifactRegistryError("TOKENIZER_MANIFEST_INVALID", "Tokenizer file descriptors are missing.")
    for manifest_key, resolved_key in (
        ("vocab", "vocab"),
        ("merges", "merges"),
        ("config", "tokenizerConfig"),
    ):
        descriptor = internal_files.get(manifest_key)
        if not isinstance(descriptor, dict) or descriptor.get("sha256") != _sha256(files[resolved_key]):
            raise ArtifactRegistryError(
                "TOKENIZER_FILE_CHECKSUM_MISMATCH",
                f"Tokenizer manifest does not bind {manifest_key} to the model artifact.",
            )
    try:
        declared_tokenizer_size = int(tokenizer_config["vocab_size"])
        declared_model_size = int(config["vocab_size"])
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactRegistryError(
            "TOKENIZER_CONFIG_INVALID", "Tokenizer and model vocabulary sizes must be declared."
        ) from error
    if len(vocab) != declared_tokenizer_size or len(vocab) != declared_model_size:
        raise ArtifactRegistryError(
            "TOKENIZER_VOCAB_MISMATCH",
            f"Tokenizer has {len(vocab)} entries, tokenizer config declares {declared_tokenizer_size}, and model config declares {declared_model_size}.",
        )


def _validate_weights(weights_path: Path, config: Mapping[str, Any]) -> int:
    expected = expected_numpy_tensor_shapes(config)
    try:
        with np.load(weights_path, allow_pickle=False) as weights:
            actual_names = set(weights.files)
            expected_names = set(expected)
            if actual_names != expected_names:
                missing = sorted(expected_names - actual_names)
                unexpected = sorted(actual_names - expected_names)
                raise ArtifactRegistryError(
                    "WEIGHT_TENSORS_MISMATCH",
                    f"Weight tensor set is incompatible; missing={missing}, unexpected={unexpected}.",
                )
            for name, shape in expected.items():
                tensor = weights[name]
                if tuple(tensor.shape) != shape:
                    raise ArtifactRegistryError(
                        "WEIGHT_SHAPE_MISMATCH",
                        f"Weight tensor {name} has shape {list(tensor.shape)}; expected {list(shape)}.",
                    )
                if tensor.dtype != np.float32:
                    raise ArtifactRegistryError(
                        "WEIGHT_DTYPE_MISMATCH",
                        f"Weight tensor {name} has dtype {tensor.dtype}; expected float32 for {SUPPORTED_RUNTIME}.",
                    )
    except ArtifactRegistryError:
        raise
    except (OSError, ValueError) as error:
        raise ArtifactRegistryError(
            "WEIGHTS_INVALID", "Model weights are not a readable, non-pickled NPZ artifact."
        ) from error
    return sum(math.prod(shape) for shape in expected.values())


def resolve_active_artifact(registry_path: str | Path | None = None) -> ResolvedArtifact:
    registry = Path(registry_path or DEFAULT_REGISTRY_PATH).resolve()
    if not registry.is_file():
        raise ArtifactRegistryError(
            "MODEL_REGISTRY_NOT_FOUND", f"Local model registry is missing: {registry}"
        )
    registry_data = _read_json(registry, "MODEL_REGISTRY_INVALID")
    if not isinstance(registry_data, dict):
        raise ArtifactRegistryError("MODEL_REGISTRY_INVALID", "Model registry must be a JSON object.")
    schema_version = registry_data.get("schemaVersion")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ArtifactRegistryError(
            "MODEL_REGISTRY_SCHEMA_UNSUPPORTED",
            "Model registry schemaVersion must be one of: 1, 2.",
        )
    if schema_version == 2:
        # A schema-2 registry is trusted only after its Ed25519 signature and
        # state invariants pass.  Import lazily so the legacy NumPy loader stays
        # independent from the future Gen1 runtime implementation.
        from model.registry_manager import RegistryManagerError, verify_registry

        try:
            registry_data = verify_registry(
                registry,
                trust_store_path=registry.parent / "trust" / "trusted-keys.json",
            )
        except RegistryManagerError as error:
            raise ArtifactRegistryError(error.code, error.message) from error

    active_id = registry_data.get("activeArtifactId")
    if not isinstance(active_id, str) or not active_id.strip():
        reason = str(registry_data.get("reason") or "No local model artifact has passed release approval.")
        raise ArtifactRegistryError("NO_APPROVED_MODEL_ARTIFACT", reason)
    try:
        parse_artifact_id(active_id)
    except ValueError as error:
        raise ArtifactRegistryError("MODEL_ARTIFACT_ID_INVALID", str(error)) from error

    artifacts = registry_data.get("artifacts")
    if not isinstance(artifacts, list):
        raise ArtifactRegistryError("MODEL_REGISTRY_INVALID", "Model registry artifacts must be an array.")
    entry = next(
        (candidate for candidate in artifacts if isinstance(candidate, dict) and candidate.get("artifactId") == active_id),
        None,
    )
    if entry is None:
        raise ArtifactRegistryError(
            "ACTIVE_MODEL_ARTIFACT_NOT_FOUND",
            f"Active artifact {active_id!r} is not present in the registry.",
        )
    if entry.get("releaseStatus") != "approved":
        raise ArtifactRegistryError(
            "MODEL_ARTIFACT_NOT_APPROVED",
            f"Active artifact {active_id!r} is not release-approved.",
        )
    runtime = entry.get("runtime")
    if runtime != SUPPORTED_RUNTIME:
        raise ArtifactRegistryError(
            "MODEL_RUNTIME_UNSUPPORTED", f"Artifact runtime {runtime!r} is not supported."
        )
    data_lineage = entry.get("dataLineage")
    if not isinstance(data_lineage, dict):
        raise ArtifactRegistryError(
            "MODEL_DATA_LINEAGE_MISSING",
            "An approved artifact must declare governed sourceIds and shardIds.",
        )
    source_ids = data_lineage.get("sourceIds")
    shard_ids = data_lineage.get("shardIds")
    if (
        not isinstance(source_ids, list)
        or not source_ids
        or any(not isinstance(value, str) or not value.startswith("vf-src-") for value in source_ids)
        or len(source_ids) != len(set(source_ids))
        or not isinstance(shard_ids, list)
        or not shard_ids
        or any(not isinstance(value, str) or not value.startswith("vf-shard-") for value in shard_ids)
        or len(shard_ids) != len(set(shard_ids))
    ):
        raise ArtifactRegistryError(
            "MODEL_DATA_LINEAGE_INVALID",
            "Approved artifact dataLineage must contain unique, non-empty VoltForge sourceIds and shardIds.",
        )
    quantization = entry.get("quantization")
    if quantization != "fp32":
        raise ArtifactRegistryError(
            "MODEL_QUANTIZATION_UNSUPPORTED",
            f"Artifact quantization {quantization!r} is not supported by {SUPPORTED_RUNTIME}.",
        )
    artifact_checksum = entry.get("checksum")
    if not isinstance(artifact_checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", artifact_checksum):
        raise ArtifactRegistryError(
            "MODEL_ARTIFACT_CHECKSUM_INVALID",
            "Artifact checksum must be a lowercase SHA-256 value.",
        )
    model_root = registry.parent.parent.resolve()
    root_value = entry.get("root")
    if not isinstance(root_value, str) or not root_value:
        raise ArtifactRegistryError("MODEL_ARTIFACT_ROOT_INVALID", "Artifact root is missing.")
    root = (registry.parent / root_value).resolve()
    try:
        root.relative_to(model_root)
    except ValueError as error:
        raise ArtifactRegistryError(
            "MODEL_ARTIFACT_ROOT_INVALID", "Artifact root must remain beneath the model directory."
        ) from error
    if not root.is_dir():
        raise ArtifactRegistryError(
            "MODEL_ARTIFACT_ROOT_MISSING", f"Approved artifact directory is missing: {root.name}"
        )

    manifest_files = entry.get("files")
    if not isinstance(manifest_files, dict):
        raise ArtifactRegistryError("MODEL_ARTIFACT_FILES_INVALID", "Artifact files manifest is missing.")
    resolved_files: dict[str, Path] = {}
    for key in REQUIRED_FILES:
        file_entry = manifest_files.get(key)
        if not isinstance(file_entry, dict):
            raise ArtifactRegistryError(
                "MODEL_ARTIFACT_FILE_UNDECLARED", f"Required artifact file {key!r} is not declared."
            )
        relative_path = file_entry.get("path")
        declared_hash = file_entry.get("sha256")
        if not isinstance(relative_path, str) or not isinstance(declared_hash, str):
            raise ArtifactRegistryError(
                "MODEL_ARTIFACT_FILE_UNDECLARED",
                f"Artifact file {key!r} must declare path and sha256.",
            )
        file_path = _resolve_beneath(root, relative_path, "MODEL_ARTIFACT_PATH_INVALID")
        if not file_path.is_file():
            raise ArtifactRegistryError(
                "MODEL_ARTIFACT_FILE_MISSING", f"Required artifact file is missing: {relative_path}"
            )
        actual_hash = _sha256(file_path)
        if actual_hash != declared_hash.lower():
            raise ArtifactRegistryError(
                "MODEL_ARTIFACT_CHECKSUM_MISMATCH",
                f"Checksum mismatch for artifact file {relative_path}.",
            )
        resolved_files[key] = file_path

    config = _read_json(resolved_files["config"], "MODEL_CONFIG_INVALID")
    if not isinstance(config, dict):
        raise ArtifactRegistryError("MODEL_CONFIG_INVALID", "Model config must be a JSON object.")
    metadata = _read_json(resolved_files["metadata"], "MODEL_METADATA_INVALID")
    if (
        not isinstance(metadata, dict)
        or metadata.get("artifactId") != active_id
        or metadata.get("familySlug") != "vfdlm"
    ):
        raise ArtifactRegistryError(
            "MODEL_METADATA_MISMATCH",
            "Model metadata identity does not match the active VFDLM registry entry.",
        )
    _validate_tokenizer(resolved_files, config, data_lineage)
    actual_parameter_count = _validate_weights(resolved_files["weights"], config)
    declared_parameter_count = entry.get("parameterCount")
    if not isinstance(declared_parameter_count, int) or declared_parameter_count != actual_parameter_count:
        raise ArtifactRegistryError(
            "MODEL_PARAMETER_COUNT_MISMATCH",
            f"Manifest declares {declared_parameter_count!r} parameters; tensors contain {actual_parameter_count}.",
        )
    weights_checksum = manifest_files["weights"]["sha256"].lower()
    if artifact_checksum != weights_checksum:
        raise ArtifactRegistryError(
            "MODEL_ARTIFACT_CHECKSUM_MISMATCH",
            "Artifact checksum does not match the declared model weights checksum.",
        )

    return ResolvedArtifact(
        artifact_id=active_id,
        runtime=runtime,
        root=root,
        files=resolved_files,
        config=config,
        parameter_count=actual_parameter_count,
        quantization=quantization,
        checksum=artifact_checksum,
        release_status="approved",
        data_lineage=data_lineage,
        registry_path=registry,
    )


def get_artifact_health(registry_path: str | Path | None = None) -> dict[str, Any]:
    registry = Path(registry_path or DEFAULT_REGISTRY_PATH).resolve()
    try:
        return resolve_active_artifact(registry).health()
    except ArtifactRegistryError as error:
        health = empty_identity_health()
        health.update({
            "ready": False,
            "state": "unavailable",
            "code": error.code,
            "message": error.message,
            "parameterCount": None,
            "contextLength": None,
            "vocabSize": None,
            "quantization": None,
            "checksum": None,
            "releaseStatus": "none" if error.code == "NO_APPROVED_MODEL_ARTIFACT" else "invalid",
            "registryPath": str(registry),
        })
        if error.code == "NO_APPROVED_MODEL_ARTIFACT":
            try:
                registry_data = _read_json(registry, "MODEL_REGISTRY_INVALID")
                if isinstance(registry_data, dict) and registry_data.get("schemaVersion") == 2:
                    from model.registry_manager import RegistryManagerError, verify_registry

                    try:
                        verified_registry = verify_registry(
                            registry,
                            trust_store_path=registry.parent / "trust" / "trusted-keys.json",
                        )
                    except RegistryManagerError:
                        verified_registry = None
                    if verified_registry is not None:
                        catalog = [
                            {
                                "artifactId": item["artifactId"],
                                "releaseStatus": item["releaseStatus"],
                                "activationEligible": item["activationEligible"],
                                "runtime": item["runtime"],
                                "parameterCount": item["parameterCount"],
                                "contextLength": item["contextLength"],
                                "quantization": item["quantization"],
                                "manifestSha256": item["manifestSha256"],
                            }
                            for item in verified_registry["artifacts"]
                        ]
                        health.update(
                            {
                                "registrySchemaVersion": verified_registry["schemaVersion"],
                                "registryRevision": verified_registry["revision"],
                                "registrySha256": verified_registry["registrySha256"],
                                "catalogArtifactCount": len(catalog),
                                "catalogArtifacts": catalog,
                            }
                        )
            except ArtifactRegistryError:
                pass
        return health
    except Exception:
        health = empty_identity_health()
        health.update({
            "ready": False,
            "state": "failed",
            "code": "MODEL_ARTIFACT_VALIDATION_FAILED",
            "message": "Local model artifact validation failed unexpectedly.",
            "parameterCount": None,
            "contextLength": None,
            "vocabSize": None,
            "quantization": None,
            "checksum": None,
            "releaseStatus": "invalid",
            "registryPath": str(registry),
        })
        return health
