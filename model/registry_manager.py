"""Signed, fail-closed lifecycle management for VoltForge Gen1 artifacts.

The production inference loader remains intentionally separate until VFAI-017.
This module owns package integrity, compatibility validation, atomic registry
switches, rollback, and recovery.  It never loads model weights or prompt data.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import base64
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Callable, Iterator, Mapping

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
except ImportError:  # pragma: no cover - exercised only in a broken toolchain
    serialization = None
    Ed25519PrivateKey = None
    Ed25519PublicKey = None

from model.identity import parse_artifact_id


MODEL_ROOT = Path(__file__).resolve().parent
DEFAULT_REGISTRY_PATH = MODEL_ROOT / "registry" / "active_model.json"
DEFAULT_TRUST_STORE_PATH = MODEL_ROOT / "registry" / "trust" / "trusted-keys.json"
REGISTRY_SCHEMA_VERSION = 2
ARTIFACT_SCHEMA_VERSION = 2
SIGNATURE_ALGORITHM = "Ed25519"
SUPPORTED_RUNTIME = "pytorch-gen1-v1"
SUPPORTED_RUNTIME_INTERFACE = 1
SUPPORTED_ARCHITECTURE = "vfdlm-gen1-decoder-v1"
SUPPORTED_CHECKPOINT_FORMAT = "pytorch-weights-only-state-dict-v1"
SUPPORTED_TOKENIZER_ID = "vfdlm-byte-bpe"
SUPPORTED_TOKENIZER_CONTRACT = "1.0.0"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class RegistryManagerError(RuntimeError):
    """Precise public-safe package or registry validation failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class RuntimeContract:
    runtime: str
    interface_version: int
    python_abi: str
    torch_version: str
    architecture_id: str = SUPPORTED_ARCHITECTURE
    checkpoint_format: str = SUPPORTED_CHECKPOINT_FORMAT
    tokenizer_id: str = SUPPORTED_TOKENIZER_ID
    tokenizer_contract: str = SUPPORTED_TOKENIZER_CONTRACT

    @classmethod
    def current(cls) -> "RuntimeContract":
        try:
            torch_version = importlib.metadata.version("torch")
        except importlib.metadata.PackageNotFoundError:
            torch_version = "unavailable"
        python_abi = f"cp{os.sys.version_info.major}{os.sys.version_info.minor}"
        return cls(
            runtime=SUPPORTED_RUNTIME,
            interface_version=SUPPORTED_RUNTIME_INTERFACE,
            python_abi=python_abi,
            torch_version=torch_version,
        )

    def as_compatibility(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "runtime": self.runtime,
            "runtimeInterfaceVersion": self.interface_version,
            "pythonAbi": self.python_abi,
            "torchVersion": self.torch_version,
            "architectureId": self.architecture_id,
            "checkpointFormat": self.checkpoint_format,
            "tokenizerId": self.tokenizer_id,
            "tokenizerContractVersion": self.tokenizer_contract,
        }


@dataclass(frozen=True)
class VerifiedArtifact:
    artifact_id: str
    root: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    manifest_file_sha256: str
    release_status: str
    activation_eligible: bool
    runtime: str
    parameter_count: int
    context_length: int
    quantization: str
    package_bytes: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def json_file_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except (FileNotFoundError, OSError) as error:
        raise RegistryManagerError(
            "REGISTRY_FILE_UNREADABLE", f"Required file is unreadable: {path.name}."
        ) from error
    return digest.hexdigest()


def read_json(path: Path, code: str = "REGISTRY_JSON_INVALID") -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RegistryManagerError(code, f"Required JSON file is missing: {path.name}.") from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RegistryManagerError(code, f"JSON file is invalid: {path.name}.") from error


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_file_bytes(value))


def _require_crypto() -> None:
    if Ed25519PrivateKey is None or Ed25519PublicKey is None or serialization is None:
        raise RegistryManagerError(
            "REGISTRY_CRYPTO_UNAVAILABLE",
            "Ed25519 support is unavailable; install the pinned registry dependency.",
        )


def _document_digest(document: Mapping[str, Any], digest_field: str) -> str:
    payload = deepcopy(dict(document))
    payload.pop("signature", None)
    payload.pop(digest_field, None)
    return sha256_bytes(canonical_json_bytes(payload))


def initialize_signing_key(
    private_key_path: Path,
    trust_store_path: Path = DEFAULT_TRUST_STORE_PATH,
    *,
    key_id: str,
) -> dict[str, Any]:
    """Create or validate one local Ed25519 key and its public trust record."""

    _require_crypto()
    private_key_path = private_key_path.resolve()
    trust_store_path = trust_store_path.resolve()
    if not private_key_path.exists() and trust_store_path.exists():
        raise RegistryManagerError(
            "REGISTRY_PRIVATE_KEY_MISSING",
            "The trusted signing key has no local private-key backup; refusing silent key rotation.",
        )
    if private_key_path.exists():
        try:
            private_key = serialization.load_pem_private_key(
                private_key_path.read_bytes(), password=None
            )
        except (OSError, ValueError, TypeError) as error:
            raise RegistryManagerError(
                "REGISTRY_PRIVATE_KEY_INVALID", "The local registry signing key is invalid."
            ) from error
        if not isinstance(private_key, Ed25519PrivateKey):
            raise RegistryManagerError(
                "REGISTRY_PRIVATE_KEY_INVALID", "The registry signing key must be Ed25519."
            )
    else:
        private_key_path.parent.mkdir(parents=True, exist_ok=True)
        private_key = Ed25519PrivateKey.generate()
        private_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        try:
            with private_key_path.open("xb") as handle:
                handle.write(private_bytes)
            try:
                private_key_path.chmod(0o600)
            except OSError:
                pass
        except FileExistsError:
            return initialize_signing_key(
                private_key_path, trust_store_path, key_id=key_id
            )

    public_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    record = {
        "keyId": key_id,
        "algorithm": SIGNATURE_ALGORITHM,
        "publicKeyBase64": base64.b64encode(public_bytes).decode("ascii"),
        "publicKeySha256": sha256_bytes(public_bytes),
        "purpose": "VoltForge local artifact and registry signing",
    }
    if trust_store_path.exists():
        trust_store = read_json(trust_store_path, "REGISTRY_TRUST_STORE_INVALID")
        keys = trust_store.get("keys") if isinstance(trust_store, dict) else None
        existing = next(
            (item for item in keys or [] if isinstance(item, dict) and item.get("keyId") == key_id),
            None,
        )
        if existing != record:
            raise RegistryManagerError(
                "REGISTRY_TRUST_KEY_MISMATCH",
                f"Trust record for signing key {key_id!r} does not match the private key.",
            )
    else:
        write_json(
            trust_store_path,
            {
                "schemaVersion": 1,
                "trustPolicy": "explicit-local-ed25519-keys-only",
                "keys": [record],
            },
        )
    return record


def _load_private_key(path: Path):
    _require_crypto()
    try:
        private_key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    except (FileNotFoundError, OSError, ValueError, TypeError) as error:
        raise RegistryManagerError(
            "REGISTRY_PRIVATE_KEY_INVALID", "The local registry signing key is missing or invalid."
        ) from error
    if not isinstance(private_key, Ed25519PrivateKey):
        raise RegistryManagerError(
            "REGISTRY_PRIVATE_KEY_INVALID", "The registry signing key must be Ed25519."
        )
    return private_key


def sign_document(
    document: Mapping[str, Any],
    *,
    digest_field: str,
    private_key_path: Path,
    key_id: str,
) -> dict[str, Any]:
    private_key = _load_private_key(private_key_path.resolve())
    signed = deepcopy(dict(document))
    signed.pop("signature", None)
    signed[digest_field] = _document_digest(signed, digest_field)
    payload = canonical_json_bytes(signed)
    public_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    signed["signature"] = {
        "algorithm": SIGNATURE_ALGORITHM,
        "keyId": key_id,
        "publicKeySha256": sha256_bytes(public_bytes),
        "signatureBase64": base64.b64encode(private_key.sign(payload)).decode("ascii"),
    }
    return signed


def verify_signed_document(
    document: Mapping[str, Any],
    *,
    digest_field: str,
    trust_store_path: Path = DEFAULT_TRUST_STORE_PATH,
    invalid_code: str,
) -> str:
    _require_crypto()
    if not isinstance(document, dict):
        raise RegistryManagerError(invalid_code, "Signed document must be a JSON object.")
    digest = document.get(digest_field)
    if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
        raise RegistryManagerError(invalid_code, f"{digest_field} must be a lowercase SHA-256 value.")
    if _document_digest(document, digest_field) != digest:
        raise RegistryManagerError(invalid_code, f"{digest_field} does not match the signed content.")

    signature = document.get("signature")
    if not isinstance(signature, dict) or signature.get("algorithm") != SIGNATURE_ALGORITHM:
        raise RegistryManagerError(invalid_code, "A trusted Ed25519 signature is required.")
    key_id = signature.get("keyId")
    encoded_signature = signature.get("signatureBase64")
    if not isinstance(key_id, str) or not isinstance(encoded_signature, str):
        raise RegistryManagerError(invalid_code, "Signature key ID or value is missing.")

    trust_store = read_json(trust_store_path.resolve(), "REGISTRY_TRUST_STORE_INVALID")
    keys = trust_store.get("keys") if isinstance(trust_store, dict) else None
    if not isinstance(keys, list):
        raise RegistryManagerError("REGISTRY_TRUST_STORE_INVALID", "Trust store keys must be an array.")
    key_record = next(
        (item for item in keys if isinstance(item, dict) and item.get("keyId") == key_id),
        None,
    )
    if key_record is None or key_record.get("algorithm") != SIGNATURE_ALGORITHM:
        raise RegistryManagerError(invalid_code, f"Signing key {key_id!r} is not trusted.")
    try:
        public_bytes = base64.b64decode(key_record["publicKeyBase64"], validate=True)
        signature_bytes = base64.b64decode(encoded_signature, validate=True)
    except (KeyError, TypeError, ValueError) as error:
        raise RegistryManagerError(invalid_code, "Signature or trust-key encoding is invalid.") from error
    public_fingerprint = sha256_bytes(public_bytes)
    if (
        key_record.get("publicKeySha256") != public_fingerprint
        or signature.get("publicKeySha256") != public_fingerprint
    ):
        raise RegistryManagerError(invalid_code, "Signature public-key fingerprint does not match trust.")

    unsigned = deepcopy(dict(document))
    unsigned.pop("signature", None)
    try:
        Ed25519PublicKey.from_public_bytes(public_bytes).verify(
            signature_bytes, canonical_json_bytes(unsigned)
        )
    except (ValueError, TypeError) as error:
        raise RegistryManagerError(invalid_code, "Ed25519 signature verification failed.") from error
    except Exception as error:  # cryptography raises InvalidSignature from a versioned module
        raise RegistryManagerError(invalid_code, "Ed25519 signature verification failed.") from error
    return digest


def _safe_relative_path(value: Any, code: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise RegistryManagerError(code, "Manifest paths must be non-empty POSIX relative paths.")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise RegistryManagerError(code, "Manifest paths must remain beneath their declared root.")
    return path


def _resolve_beneath(base: Path, value: Any, code: str) -> Path:
    relative = _safe_relative_path(value, code)
    resolved = base.joinpath(*relative.parts).resolve()
    try:
        resolved.relative_to(base.resolve())
    except ValueError as error:
        raise RegistryManagerError(code, "Manifest path escapes its declared root.") from error
    return resolved


def validate_compatibility(
    compatibility: Mapping[str, Any], runtime_contract: RuntimeContract | None = None
) -> None:
    contract = runtime_contract or RuntimeContract.current()
    expected = contract.as_compatibility()
    mismatches = [
        key for key, expected_value in expected.items() if compatibility.get(key) != expected_value
    ]
    if mismatches:
        joined = ", ".join(sorted(mismatches))
        raise RegistryManagerError(
            "MODEL_ARTIFACT_INCOMPATIBLE",
            f"Artifact compatibility does not match this runtime: {joined}.",
        )


def verify_artifact_directory(
    root: Path,
    *,
    trust_store_path: Path = DEFAULT_TRUST_STORE_PATH,
    runtime_contract: RuntimeContract | None = None,
    require_activation: bool = False,
) -> VerifiedArtifact:
    root = root.resolve()
    if not root.is_dir():
        raise RegistryManagerError(
            "MODEL_ARTIFACT_ROOT_MISSING", f"Artifact directory is missing: {root.name}."
        )
    manifest_path = root / "artifact-manifest.json"
    manifest = read_json(manifest_path, "MODEL_ARTIFACT_MANIFEST_INVALID")
    if not isinstance(manifest, dict) or manifest.get("schemaVersion") != ARTIFACT_SCHEMA_VERSION:
        raise RegistryManagerError(
            "MODEL_ARTIFACT_SCHEMA_UNSUPPORTED",
            f"Artifact schemaVersion must be {ARTIFACT_SCHEMA_VERSION}.",
        )
    manifest_sha256 = verify_signed_document(
        manifest,
        digest_field="manifestSha256",
        trust_store_path=trust_store_path,
        invalid_code="MODEL_ARTIFACT_SIGNATURE_INVALID",
    )
    artifact_id = manifest.get("artifactId")
    try:
        identity = parse_artifact_id(artifact_id)
    except ValueError as error:
        raise RegistryManagerError("MODEL_ARTIFACT_ID_INVALID", str(error)) from error
    if root.name != identity.artifact_id:
        raise RegistryManagerError(
            "MODEL_ARTIFACT_ID_MISMATCH", "Artifact directory name does not match its signed identity."
        )
    if manifest.get("immutable") is not True:
        raise RegistryManagerError(
            "MODEL_ARTIFACT_MUTABLE", "Artifact manifest must declare immutable=true."
        )

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RegistryManagerError(
            "MODEL_ARTIFACT_FILES_INVALID", "Artifact manifest files must be a non-empty array."
        )
    declared_paths: set[str] = set()
    package_bytes = 0
    for item in files:
        if not isinstance(item, dict):
            raise RegistryManagerError(
                "MODEL_ARTIFACT_FILES_INVALID", "Every artifact file entry must be an object."
            )
        relative = _safe_relative_path(item.get("path"), "MODEL_ARTIFACT_PATH_INVALID")
        relative_text = relative.as_posix()
        if relative_text == "artifact-manifest.json" or relative_text in declared_paths:
            raise RegistryManagerError(
                "MODEL_ARTIFACT_FILES_INVALID", "Artifact file paths must be unique and exclude the manifest."
            )
        declared_paths.add(relative_text)
        path = _resolve_beneath(root, relative_text, "MODEL_ARTIFACT_PATH_INVALID")
        if not path.is_file():
            raise RegistryManagerError(
                "MODEL_ARTIFACT_FILE_MISSING", f"Declared artifact file is missing: {relative_text}."
            )
        expected_bytes = item.get("bytes")
        expected_sha256 = item.get("sha256")
        if not isinstance(expected_bytes, int) or expected_bytes < 0:
            raise RegistryManagerError(
                "MODEL_ARTIFACT_MANIFEST_INVALID", f"Invalid byte count for {relative_text}."
            )
        if not isinstance(expected_sha256, str) or SHA256_PATTERN.fullmatch(expected_sha256) is None:
            raise RegistryManagerError(
                "MODEL_ARTIFACT_MANIFEST_INVALID", f"Invalid SHA-256 for {relative_text}."
            )
        if path.stat().st_size != expected_bytes or sha256_file(path) != expected_sha256:
            raise RegistryManagerError(
                "MODEL_ARTIFACT_CHECKSUM_MISMATCH",
                f"Artifact file integrity check failed: {relative_text}.",
            )
        package_bytes += expected_bytes

    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.resolve() != manifest_path.resolve()
    }
    if actual_paths != declared_paths:
        raise RegistryManagerError(
            "MODEL_ARTIFACT_FILE_SET_MISMATCH",
            "Artifact directory contains undeclared files or omits declared files.",
        )
    package_sha256 = manifest.get("packageSha256")
    if (
        not isinstance(package_sha256, str)
        or SHA256_PATTERN.fullmatch(package_sha256) is None
        or sha256_bytes(canonical_json_bytes(files)) != package_sha256
    ):
        raise RegistryManagerError(
            "MODEL_ARTIFACT_PACKAGE_CHECKSUM_INVALID",
            "Artifact package checksum does not match its signed file inventory.",
        )

    compatibility_path = _resolve_beneath(
        root, manifest.get("compatibilityPath"), "MODEL_ARTIFACT_MANIFEST_INVALID"
    )
    compatibility = read_json(compatibility_path, "MODEL_ARTIFACT_COMPATIBILITY_INVALID")
    if not isinstance(compatibility, dict):
        raise RegistryManagerError(
            "MODEL_ARTIFACT_COMPATIBILITY_INVALID", "Compatibility document must be an object."
        )
    validate_compatibility(compatibility, runtime_contract)

    release = manifest.get("releaseState")
    model = manifest.get("model")
    if not isinstance(release, dict) or not isinstance(model, dict):
        raise RegistryManagerError(
            "MODEL_ARTIFACT_MANIFEST_INVALID", "Release state and model identity are required."
        )
    release_status = release.get("releaseStatus")
    activation_eligible = release.get("activationEligible") is True
    if release_status not in {"experimental", "approved", "retired"}:
        raise RegistryManagerError(
            "MODEL_ARTIFACT_RELEASE_STATE_INVALID", "Artifact release status is invalid."
        )
    if require_activation and (release_status != "approved" or not activation_eligible):
        raise RegistryManagerError(
            "MODEL_ARTIFACT_NOT_APPROVED",
            f"Artifact {artifact_id!r} is not approved and activation-eligible.",
        )
    generation_policy_path = _resolve_beneath(
        root, manifest.get("generationPolicyPath"), "MODEL_ARTIFACT_MANIFEST_INVALID"
    )
    generation_policy = read_json(
        generation_policy_path, "MODEL_ARTIFACT_GENERATION_POLICY_INVALID"
    )
    if (
        not isinstance(generation_policy, dict)
        or generation_policy.get("artifactId") != artifact_id
        or generation_policy.get("releaseStatus") != release_status
        or (generation_policy.get("activationEligible") is True) != activation_eligible
    ):
        raise RegistryManagerError(
            "MODEL_ARTIFACT_GENERATION_POLICY_INVALID",
            "Generation policy does not match the signed artifact release state.",
        )
    if require_activation and generation_policy.get("servingEnabled") is not True:
        raise RegistryManagerError(
            "MODEL_ARTIFACT_SERVING_DISABLED",
            "Artifact generation policy does not permit serving.",
        )
    try:
        parameter_count = int(model["parameterCount"])
        context_length = int(model["contextLength"])
    except (KeyError, TypeError, ValueError) as error:
        raise RegistryManagerError(
            "MODEL_ARTIFACT_MANIFEST_INVALID", "Model parameter and context counts are required."
        ) from error
    quantization = model.get("quantization")
    if parameter_count <= 0 or context_length <= 0 or quantization != "fp32":
        raise RegistryManagerError(
            "MODEL_ARTIFACT_MANIFEST_INVALID", "Model dimensions or quantization are invalid."
        )
    if manifest.get("runtime") != SUPPORTED_RUNTIME:
        raise RegistryManagerError(
            "MODEL_ARTIFACT_INCOMPATIBLE", "Artifact runtime is unsupported."
        )
    return VerifiedArtifact(
        artifact_id=artifact_id,
        root=root,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        manifest_file_sha256=sha256_file(manifest_path),
        release_status=release_status,
        activation_eligible=activation_eligible,
        runtime=manifest["runtime"],
        parameter_count=parameter_count,
        context_length=context_length,
        quantization=quantization,
        package_bytes=package_bytes + manifest_path.stat().st_size,
    )


def verify_registry(
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    *,
    trust_store_path: Path = DEFAULT_TRUST_STORE_PATH,
) -> dict[str, Any]:
    registry_path = registry_path.resolve()
    registry = read_json(registry_path, "MODEL_REGISTRY_INVALID")
    if not isinstance(registry, dict) or registry.get("schemaVersion") != REGISTRY_SCHEMA_VERSION:
        raise RegistryManagerError(
            "MODEL_REGISTRY_SCHEMA_UNSUPPORTED",
            f"Signed registry schemaVersion must be {REGISTRY_SCHEMA_VERSION}.",
        )
    verify_signed_document(
        registry,
        digest_field="registrySha256",
        trust_store_path=trust_store_path,
        invalid_code="MODEL_REGISTRY_SIGNATURE_INVALID",
    )
    revision = registry.get("revision")
    if not isinstance(revision, int) or revision < 1:
        raise RegistryManagerError("MODEL_REGISTRY_INVALID", "Registry revision must be positive.")
    artifacts = registry.get("artifacts")
    if not isinstance(artifacts, list):
        raise RegistryManagerError("MODEL_REGISTRY_INVALID", "Registry artifacts must be an array.")
    seen: set[str] = set()
    for entry in artifacts:
        if not isinstance(entry, dict):
            raise RegistryManagerError("MODEL_REGISTRY_INVALID", "Registry entries must be objects.")
        artifact_id = entry.get("artifactId")
        try:
            parse_artifact_id(artifact_id)
        except ValueError as error:
            raise RegistryManagerError("MODEL_ARTIFACT_ID_INVALID", str(error)) from error
        if artifact_id in seen:
            raise RegistryManagerError("MODEL_REGISTRY_INVALID", "Artifact IDs must be unique.")
        seen.add(artifact_id)
        if entry.get("releaseStatus") not in {"experimental", "approved", "retired"}:
            raise RegistryManagerError("MODEL_REGISTRY_INVALID", "Registry release status is invalid.")
        if not isinstance(entry.get("activationEligible"), bool):
            raise RegistryManagerError(
                "MODEL_REGISTRY_INVALID", "Registry activation eligibility must be boolean."
            )
        if entry.get("runtime") != SUPPORTED_RUNTIME:
            raise RegistryManagerError("MODEL_REGISTRY_INVALID", "Registry runtime is unsupported.")
        root = _safe_relative_path(entry.get("root"), "MODEL_ARTIFACT_ROOT_INVALID")
        if root.as_posix() != f"artifacts/{artifact_id}":
            raise RegistryManagerError(
                "MODEL_ARTIFACT_ROOT_INVALID", "Registry artifact root must match its immutable ID."
            )
        if entry.get("manifestPath") != "artifact-manifest.json":
            raise RegistryManagerError(
                "MODEL_REGISTRY_INVALID", "Registry manifest path must identify artifact-manifest.json."
            )
        for hash_key in ("manifestSha256", "manifestFileSha256"):
            if not isinstance(entry.get(hash_key), str) or SHA256_PATTERN.fullmatch(entry[hash_key]) is None:
                raise RegistryManagerError(
                    "MODEL_REGISTRY_INVALID", f"Registry entry {hash_key} must be SHA-256."
                )
        for count_key in ("parameterCount", "contextLength", "packageBytes"):
            if not isinstance(entry.get(count_key), int) or entry[count_key] <= 0:
                raise RegistryManagerError(
                    "MODEL_REGISTRY_INVALID", f"Registry entry {count_key} must be positive."
                )
        if entry.get("quantization") != "fp32":
            raise RegistryManagerError(
                "MODEL_REGISTRY_INVALID", "Registry entry quantization must be fp32."
            )

    active_id = registry.get("activeArtifactId")
    if active_id is None:
        if registry.get("releaseStatus") != "none" or registry.get("state") != "no-approved-artifact":
            raise RegistryManagerError(
                "MODEL_REGISTRY_STATE_INVALID", "Inactive registry status fields are inconsistent."
            )
    else:
        entry = next((item for item in artifacts if item.get("artifactId") == active_id), None)
        if entry is None:
            raise RegistryManagerError(
                "ACTIVE_MODEL_ARTIFACT_NOT_FOUND", "Active artifact is absent from the registry catalog."
            )
        if entry.get("releaseStatus") != "approved" or entry.get("activationEligible") is not True:
            raise RegistryManagerError(
                "MODEL_ARTIFACT_NOT_APPROVED", "Active artifact is not approved and activation-eligible."
            )
        if registry.get("releaseStatus") != "approved" or registry.get("state") != "active":
            raise RegistryManagerError(
                "MODEL_REGISTRY_STATE_INVALID", "Active registry status fields are inconsistent."
            )
    return registry


def artifact_root_from_entry(registry_path: Path, entry: Mapping[str, Any]) -> Path:
    return _resolve_beneath(
        registry_path.resolve().parent,
        entry.get("root"),
        "MODEL_ARTIFACT_ROOT_INVALID",
    )


def registry_entry(artifact: VerifiedArtifact) -> dict[str, Any]:
    return {
        "artifactId": artifact.artifact_id,
        "releaseStatus": artifact.release_status,
        "activationEligible": artifact.activation_eligible,
        "runtime": artifact.runtime,
        "root": f"artifacts/{artifact.artifact_id}",
        "manifestPath": "artifact-manifest.json",
        "manifestSha256": artifact.manifest_sha256,
        "manifestFileSha256": artifact.manifest_file_sha256,
        "parameterCount": artifact.parameter_count,
        "contextLength": artifact.context_length,
        "quantization": artifact.quantization,
        "packageBytes": artifact.package_bytes,
    }


def register_artifact(
    artifact: VerifiedArtifact,
    *,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    trust_store_path: Path = DEFAULT_TRUST_STORE_PATH,
    private_key_path: Path,
    key_id: str,
    replace: Callable[[str | bytes | os.PathLike[str], str | bytes | os.PathLike[str]], None] = os.replace,
) -> dict[str, Any]:
    """Catalog an immutable artifact without changing the active artifact."""

    registry_path = registry_path.resolve()
    expected_root = registry_path.parent / "artifacts" / artifact.artifact_id
    if artifact.root != expected_root.resolve():
        raise RegistryManagerError(
            "MODEL_ARTIFACT_ROOT_INVALID",
            "Registered artifact must live in registry/artifacts under its immutable ID.",
        )
    with _registry_lock(registry_path):
        current = read_json(registry_path, "MODEL_REGISTRY_INVALID")
        legacy = isinstance(current, dict) and current.get("schemaVersion") == 1
        if legacy:
            if current.get("activeArtifactId") is not None or current.get("artifacts") not in ([], None):
                raise RegistryManagerError(
                    "MODEL_REGISTRY_MIGRATION_DENIED",
                    "Only an empty inactive schema-1 registry can migrate automatically.",
                )
            legacy_bytes = registry_path.read_bytes()
            legacy_digest = sha256_bytes(legacy_bytes)
            legacy_path = registry_path.parent / "history" / f"legacy-schema1-{legacy_digest[:16]}.json"
            legacy_path.parent.mkdir(parents=True, exist_ok=True)
            if legacy_path.exists() and legacy_path.read_bytes() != legacy_bytes:
                raise RegistryManagerError(
                    "MODEL_REGISTRY_HISTORY_CONFLICT", "Legacy registry snapshot conflicts."
                )
            if not legacy_path.exists():
                with legacy_path.open("xb") as handle:
                    handle.write(legacy_bytes)
            current = {
                "schemaVersion": REGISTRY_SCHEMA_VERSION,
                "registryId": "vfdlm-local-registry-v1",
                "revision": 0,
                "identityContract": current.get("identityContract", {}),
                "activeArtifactId": None,
                "previousActiveArtifactId": None,
                "releaseStatus": "none",
                "state": "no-approved-artifact",
                "reason": "No release-approved VoltForge neural artifact is active.",
                "artifacts": [],
                "lastTransaction": None,
            }
        else:
            current = verify_registry(registry_path, trust_store_path=trust_store_path)

        existing = next(
            (item for item in current["artifacts"] if item.get("artifactId") == artifact.artifact_id),
            None,
        )
        expected_entry = registry_entry(artifact)
        if existing is not None:
            if existing != expected_entry:
                raise RegistryManagerError(
                    "MODEL_REGISTRY_ARTIFACT_CONFLICT",
                    "The artifact ID is already registered with different immutable content.",
                )
            return current

        updated = deepcopy(dict(current))
        updated.pop("signature", None)
        updated.pop("registrySha256", None)
        updated["revision"] = int(current["revision"]) + 1
        updated["artifacts"] = [*current["artifacts"], expected_entry]
        updated["lastTransaction"] = {
            "transactionId": f"registry-r{updated['revision']}-register",
            "type": "register",
            "artifactId": artifact.artifact_id,
            "fromArtifactId": current.get("activeArtifactId"),
            "toArtifactId": current.get("activeArtifactId"),
            "createdAtUtc": utc_now(),
        }
        if current.get("activeArtifactId") is None:
            updated.update(
                {
                    "releaseStatus": "none",
                    "state": "no-approved-artifact",
                    "reason": (
                        "Experimental artifacts are cataloged, but no release-approved "
                        "VoltForge neural artifact is active."
                    ),
                }
            )
        signed = sign_document(
            updated,
            digest_field="registrySha256",
            private_key_path=private_key_path,
            key_id=key_id,
        )
        if not legacy:
            _record_history(registry_path, current)
        _atomic_write_bytes(registry_path, json_file_bytes(signed), replace=replace)
        return signed


def _atomic_write_bytes(
    target: Path,
    data: bytes,
    *,
    replace: Callable[[str | bytes | os.PathLike[str], str | bytes | os.PathLike[str]], None] = os.replace,
) -> None:
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        replace(temporary, target)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


@contextmanager
def _registry_lock(registry_path: Path) -> Iterator[None]:
    lock_path = registry_path.with_suffix(registry_path.suffix + ".lock")
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RegistryManagerError(
            "MODEL_REGISTRY_LOCKED", "Another registry transaction is in progress."
        ) from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(f"pid={os.getpid()} createdAtUtc={utc_now()}\n")
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def _record_history(registry_path: Path, registry: Mapping[str, Any]) -> Path:
    revision = int(registry["revision"])
    digest = str(registry["registrySha256"])
    history_path = (
        registry_path.parent
        / "history"
        / f"revision-{revision:08d}-{digest[:16]}.json"
    )
    history_path.parent.mkdir(parents=True, exist_ok=True)
    data = json_file_bytes(registry)
    if history_path.exists():
        if history_path.read_bytes() != data:
            raise RegistryManagerError(
                "MODEL_REGISTRY_HISTORY_CONFLICT", "Registry history snapshot conflicts with prior data."
            )
    else:
        try:
            with history_path.open("xb") as handle:
                handle.write(data)
        except FileExistsError:
            if history_path.read_bytes() != data:
                raise RegistryManagerError(
                    "MODEL_REGISTRY_HISTORY_CONFLICT", "Registry history snapshot is inconsistent."
                )
    return history_path


def _signed_registry_revision(
    registry: Mapping[str, Any],
    *,
    private_key_path: Path,
    key_id: str,
    active_artifact_id: str | None,
    previous_active_artifact_id: str | None,
    transaction_type: str,
    transaction_artifact_id: str | None,
    release_id: str | None = None,
) -> dict[str, Any]:
    updated = deepcopy(dict(registry))
    updated.pop("signature", None)
    updated.pop("registrySha256", None)
    updated["revision"] = int(registry["revision"]) + 1
    updated["activeArtifactId"] = active_artifact_id
    updated["previousActiveArtifactId"] = previous_active_artifact_id
    if active_artifact_id is None:
        updated.update(
            {
                "releaseStatus": "none",
                "state": "no-approved-artifact",
                "reason": "No release-approved VoltForge neural artifact is active.",
            }
        )
    else:
        updated.update(
            {
                "releaseStatus": "approved",
                "state": "active",
                "reason": "The signed release-approved artifact is active.",
            }
        )
    transaction = {
        "transactionId": f"registry-r{updated['revision']}-{transaction_type}",
        "type": transaction_type,
        "artifactId": transaction_artifact_id,
        "fromArtifactId": registry.get("activeArtifactId"),
        "toArtifactId": active_artifact_id,
        "createdAtUtc": utc_now(),
    }
    if release_id is not None:
        transaction["releaseId"] = release_id
    updated["lastTransaction"] = transaction
    return sign_document(
        updated,
        digest_field="registrySha256",
        private_key_path=private_key_path,
        key_id=key_id,
    )


def _validate_activation_target(
    registry_path: Path,
    registry: Mapping[str, Any],
    artifact_id: str,
    *,
    trust_store_path: Path,
    runtime_contract: RuntimeContract | None,
) -> None:
    entry = next(
        (item for item in registry["artifacts"] if item.get("artifactId") == artifact_id),
        None,
    )
    if entry is None:
        raise RegistryManagerError(
            "ACTIVE_MODEL_ARTIFACT_NOT_FOUND", f"Artifact {artifact_id!r} is not registered."
        )
    if entry.get("releaseStatus") != "approved" or entry.get("activationEligible") is not True:
        raise RegistryManagerError(
            "MODEL_ARTIFACT_NOT_APPROVED",
            f"Artifact {artifact_id!r} is experimental, retired, or activation-ineligible.",
        )
    artifact = verify_artifact_directory(
        artifact_root_from_entry(registry_path, entry),
        trust_store_path=trust_store_path,
        runtime_contract=runtime_contract,
        require_activation=True,
    )
    if (
        artifact.manifest_sha256 != entry.get("manifestSha256")
        or artifact.manifest_file_sha256 != entry.get("manifestFileSha256")
        or artifact.parameter_count != entry.get("parameterCount")
        or artifact.context_length != entry.get("contextLength")
        or artifact.package_bytes != entry.get("packageBytes")
    ):
        raise RegistryManagerError(
            "MODEL_REGISTRY_ARTIFACT_MISMATCH",
            "Registered artifact identity does not match the verified immutable package.",
        )


def activate_artifact(
    artifact_id: str,
    *,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    trust_store_path: Path = DEFAULT_TRUST_STORE_PATH,
    private_key_path: Path,
    key_id: str,
    expected_revision: int | None = None,
    runtime_contract: RuntimeContract | None = None,
    release_id: str | None = None,
    replace: Callable[[str | bytes | os.PathLike[str], str | bytes | os.PathLike[str]], None] = os.replace,
) -> dict[str, Any]:
    registry_path = registry_path.resolve()
    with _registry_lock(registry_path):
        registry = verify_registry(registry_path, trust_store_path=trust_store_path)
        if expected_revision is not None and registry["revision"] != expected_revision:
            raise RegistryManagerError(
                "MODEL_REGISTRY_REVISION_CONFLICT", "Registry revision changed before activation."
            )
        _validate_activation_target(
            registry_path,
            registry,
            artifact_id,
            trust_store_path=trust_store_path,
            runtime_contract=runtime_contract,
        )
        updated = _signed_registry_revision(
            registry,
            private_key_path=private_key_path,
            key_id=key_id,
            active_artifact_id=artifact_id,
            previous_active_artifact_id=registry.get("activeArtifactId"),
            transaction_type="activate",
            transaction_artifact_id=artifact_id,
            release_id=release_id,
        )
        _record_history(registry_path, registry)
        _atomic_write_bytes(registry_path, json_file_bytes(updated), replace=replace)
        return updated


def rollback_registry(
    *,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    trust_store_path: Path = DEFAULT_TRUST_STORE_PATH,
    private_key_path: Path,
    key_id: str,
    expected_revision: int | None = None,
    runtime_contract: RuntimeContract | None = None,
    release_id: str | None = None,
    replace: Callable[[str | bytes | os.PathLike[str], str | bytes | os.PathLike[str]], None] = os.replace,
) -> dict[str, Any]:
    registry_path = registry_path.resolve()
    with _registry_lock(registry_path):
        registry = verify_registry(registry_path, trust_store_path=trust_store_path)
        if expected_revision is not None and registry["revision"] != expected_revision:
            raise RegistryManagerError(
                "MODEL_REGISTRY_REVISION_CONFLICT", "Registry revision changed before rollback."
            )
        target = registry.get("previousActiveArtifactId")
        if target is not None:
            _validate_activation_target(
                registry_path,
                registry,
                target,
                trust_store_path=trust_store_path,
                runtime_contract=runtime_contract,
            )
        updated = _signed_registry_revision(
            registry,
            private_key_path=private_key_path,
            key_id=key_id,
            active_artifact_id=target,
            previous_active_artifact_id=registry.get("activeArtifactId"),
            transaction_type="rollback",
            transaction_artifact_id=target,
            release_id=release_id,
        )
        _record_history(registry_path, registry)
        _atomic_write_bytes(registry_path, json_file_bytes(updated), replace=replace)
        return updated


def retire_artifact(
    artifact_id: str,
    *,
    reason: str,
    operator_id: str,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    trust_store_path: Path = DEFAULT_TRUST_STORE_PATH,
    private_key_path: Path,
    key_id: str,
    expected_revision: int | None = None,
    replace: Callable[[str | bytes | os.PathLike[str], str | bytes | os.PathLike[str]], None] = os.replace,
) -> dict[str, Any]:
    """Retire an inactive catalog entry through a signed registry revision.

    Retirement never deletes package bytes and never changes the active or
    rollback pointer. Active and previous-active artifacts remain protected so
    an operator can still perform the documented rollback procedure.
    """

    try:
        parse_artifact_id(artifact_id)
    except (TypeError, ValueError) as error:
        raise RegistryManagerError(
            "MODEL_ARTIFACT_ID_INVALID", "Artifact identity is invalid."
        ) from error
    if not isinstance(reason, str) or not reason.strip() or len(reason.strip()) > 500:
        raise RegistryManagerError(
            "MODEL_ARTIFACT_RETIREMENT_INVALID",
            "Retirement reason must be between 1 and 500 characters.",
        )
    if any(character in reason for character in "\r\n"):
        raise RegistryManagerError(
            "MODEL_ARTIFACT_RETIREMENT_INVALID",
            "Retirement reason must be a single line.",
        )
    if not isinstance(operator_id, str) or not operator_id.strip() or len(operator_id.strip()) > 120:
        raise RegistryManagerError(
            "MODEL_ARTIFACT_RETIREMENT_INVALID",
            "Operator identity must be between 1 and 120 characters.",
        )
    if any(character in operator_id for character in "\r\n"):
        raise RegistryManagerError(
            "MODEL_ARTIFACT_RETIREMENT_INVALID",
            "Operator identity must be a single line.",
        )

    registry_path = registry_path.resolve()
    with _registry_lock(registry_path):
        registry = verify_registry(registry_path, trust_store_path=trust_store_path)
        if expected_revision is not None and registry["revision"] != expected_revision:
            raise RegistryManagerError(
                "MODEL_REGISTRY_REVISION_CONFLICT",
                "Registry revision changed before retirement.",
            )
        if artifact_id == registry.get("activeArtifactId"):
            raise RegistryManagerError(
                "MODEL_ARTIFACT_RETIRE_ACTIVE_DENIED",
                "The active artifact cannot be retired; switch or roll back first.",
            )
        if artifact_id == registry.get("previousActiveArtifactId"):
            raise RegistryManagerError(
                "MODEL_ARTIFACT_RETIRE_ROLLBACK_DENIED",
                "The previous active artifact cannot be retired until a later stable release replaces it.",
            )
        index = next(
            (
                position
                for position, item in enumerate(registry["artifacts"])
                if item.get("artifactId") == artifact_id
            ),
            None,
        )
        if index is None:
            raise RegistryManagerError(
                "ACTIVE_MODEL_ARTIFACT_NOT_FOUND", "Artifact is not registered."
            )
        entry = deepcopy(registry["artifacts"][index])
        if entry.get("releaseStatus") == "retired":
            raise RegistryManagerError(
                "MODEL_ARTIFACT_ALREADY_RETIRED", "Artifact is already retired."
            )
        entry.update(
            {
                "releaseStatus": "retired",
                "activationEligible": False,
                "retirement": {
                    "reason": reason.strip(),
                    "operatorId": operator_id.strip(),
                    "retiredAtUtc": utc_now(),
                },
            }
        )
        catalog = deepcopy(registry)
        catalog["artifacts"][index] = entry
        updated = _signed_registry_revision(
            catalog,
            private_key_path=private_key_path,
            key_id=key_id,
            active_artifact_id=registry.get("activeArtifactId"),
            previous_active_artifact_id=registry.get("previousActiveArtifactId"),
            transaction_type="retire",
            transaction_artifact_id=artifact_id,
        )
        _record_history(registry_path, registry)
        _atomic_write_bytes(registry_path, json_file_bytes(updated), replace=replace)
        return updated


def recover_latest_registry_history(
    *,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    trust_store_path: Path = DEFAULT_TRUST_STORE_PATH,
    replace: Callable[[str | bytes | os.PathLike[str], str | bytes | os.PathLike[str]], None] = os.replace,
) -> dict[str, Any]:
    """Restore the highest signed history snapshot when the live file is damaged."""

    registry_path = registry_path.resolve()
    history_dir = registry_path.parent / "history"
    candidates: list[tuple[int, Path, dict[str, Any]]] = []
    for path in history_dir.glob("revision-*.json") if history_dir.is_dir() else ():
        try:
            registry = verify_registry(path, trust_store_path=trust_store_path)
        except RegistryManagerError:
            continue
        candidates.append((int(registry["revision"]), path, registry))
    if not candidates:
        raise RegistryManagerError(
            "MODEL_REGISTRY_HISTORY_UNAVAILABLE", "No valid signed registry history is available."
        )
    _, _, registry = max(candidates, key=lambda item: item[0])
    with _registry_lock(registry_path):
        _atomic_write_bytes(registry_path, json_file_bytes(registry), replace=replace)
    return registry
