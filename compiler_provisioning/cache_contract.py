"""Content-addressed compiler-cache identity and verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


CACHE_MANIFEST_KIND = "vfai-fu-014-compiler-cache-export-v1"
CACHE_KEY_REQUIRED_FIELDS = frozenset(
    {
        "toolchainManifestSha256",
        "profileId",
        "coreId",
        "coreVersion",
        "fqbns",
        "host",
        "arduinoCliVersion",
        "arduinoCliExecutableSha256",
        "cliCommandContract",
        "packageIndexSha256",
        "dependencyArchives",
        "source",
    }
)


class CompilerCacheContractError(RuntimeError):
    """A compiler cache is partial, stale, malformed, or unsafe."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_key_from_inputs(inputs: Mapping[str, Any]) -> str:
    """Return the exact content key for one profile/host/command closure."""

    if set(inputs) != CACHE_KEY_REQUIRED_FIELDS:
        raise CompilerCacheContractError("compiler cache key fields are incomplete or unknown")
    if not isinstance(inputs.get("fqbns"), list) or not inputs["fqbns"]:
        raise CompilerCacheContractError("compiler cache key requires exact FQBNs")
    if not isinstance(inputs.get("dependencyArchives"), list):
        raise CompilerCacheContractError("compiler cache dependencies must be an array")
    for name in (
        "toolchainManifestSha256",
        "profileId",
        "coreId",
        "coreVersion",
        "host",
        "arduinoCliVersion",
        "arduinoCliExecutableSha256",
    ):
        if not isinstance(inputs.get(name), str) or not inputs[name]:
            raise CompilerCacheContractError(f"compiler cache key field is invalid: {name}")
    return hashlib.sha256(_canonical_json(dict(inputs))).hexdigest()


def _safe_relative_path(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise CompilerCacheContractError("compiler cache file path is invalid")
    relative = Path(value)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() != value
    ):
        raise CompilerCacheContractError("compiler cache file path escapes its root")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise CompilerCacheContractError("compiler cache file path escapes its root") from exc
    return path


def verify_cache_export(manifest: Mapping[str, Any], cache_root: Path) -> dict[str, Any]:
    """Verify an exact cache file set, hashes, byte count, and key binding."""

    if manifest.get("schemaVersion") != 1 or manifest.get("manifestKind") != CACHE_MANIFEST_KIND:
        raise CompilerCacheContractError("compiler cache manifest schema is invalid")
    key_inputs = manifest.get("keyInputs")
    if not isinstance(key_inputs, Mapping):
        raise CompilerCacheContractError("compiler cache key inputs are missing")
    expected_key = cache_key_from_inputs(key_inputs)
    if manifest.get("cacheKey") != expected_key:
        raise CompilerCacheContractError("compiler cache key is stale")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise CompilerCacheContractError("compiler cache file inventory is empty")
    root = cache_root.resolve()
    if not root.is_dir():
        raise CompilerCacheContractError("compiler cache root is missing")
    declared_paths: set[str] = set()
    total_bytes = 0
    for item in files:
        if not isinstance(item, Mapping):
            raise CompilerCacheContractError("compiler cache file descriptor is invalid")
        relative = item.get("path")
        if relative in declared_paths:
            raise CompilerCacheContractError("compiler cache file path is duplicated")
        path = _safe_relative_path(root, relative)
        declared_paths.add(str(relative))
        expected_bytes = item.get("bytes")
        expected_sha = item.get("sha256")
        if (
            not isinstance(expected_bytes, int)
            or isinstance(expected_bytes, bool)
            or expected_bytes < 0
            or not isinstance(expected_sha, str)
            or len(expected_sha) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha)
        ):
            raise CompilerCacheContractError("compiler cache file identity is invalid")
        if not path.is_file():
            raise CompilerCacheContractError(f"compiler cache file is missing: {relative}")
        if path.stat().st_size != expected_bytes or _sha256_file(path) != expected_sha:
            raise CompilerCacheContractError(f"compiler cache file changed: {relative}")
        total_bytes += expected_bytes
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual_paths != declared_paths:
        raise CompilerCacheContractError(
            "compiler cache contains undeclared files or omits declared files"
        )
    if manifest.get("totalBytes") != total_bytes:
        raise CompilerCacheContractError("compiler cache byte total is invalid")
    return {
        "cacheKey": expected_key,
        "fileCount": len(files),
        "totalBytes": total_bytes,
        "exactFileSet": True,
    }
