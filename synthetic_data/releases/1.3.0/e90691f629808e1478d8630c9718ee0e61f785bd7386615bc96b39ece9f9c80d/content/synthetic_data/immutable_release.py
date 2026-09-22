"""Publish and verify immutable, content-addressed synthetic-data releases.

Release directories mirror the original AI-workspace paths under ``content``.
They are write-once: an existing directory is verified but never repaired or
overwritten. ``CURRENT.json`` is only a mutable pointer to a verified release.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

from data_governance.governance import sha256_file


AI_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_ROOT = AI_ROOT / "synthetic_data"
RELEASES_ROOT = SYNTHETIC_ROOT / "releases"
CURRENT_POINTER_PATH = RELEASES_ROOT / "CURRENT.json"
REPORT_PATH = SYNTHETIC_ROOT / "reports/current-generation-report.json"
RECEIPT_PATH = SYNTHETIC_ROOT / "receipts/v1/compile-receipts.jsonl"
PIPELINE_LOCK_PATH = SYNTHETIC_ROOT / "pipeline-lock.v1.json"
TOOLCHAIN_MANIFEST_PATH = SYNTHETIC_ROOT / "toolchains.v1.json"
SOURCE_REGISTRY_PATH = AI_ROOT / "data_governance/source-registry.v1.json"
DATA_POLICY_PATH = AI_ROOT / "data_governance/policy.v1.json"
RELEASE_MANIFEST_NAME = "release-manifest.json"
RELEASE_MANIFEST_KIND = "vfai-fu-015-immutable-synthetic-release-v1"
CURRENT_POINTER_KIND = "vfai-fu-015-current-synthetic-release-pointer-v1"


class ImmutableReleaseError(RuntimeError):
    """A release input, immutable directory, pointer, or lineage is invalid."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Mapping[str, Any], field: str) -> str:
    unsigned = dict(value)
    unsigned.pop(field, None)
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ImmutableReleaseError(f"unable to read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ImmutableReleaseError(f"{label} must be a JSON object")
    return value


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_bytes(_json_bytes(value))
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _relative_to(root: Path, path: Path, label: str) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ImmutableReleaseError(f"{label} escapes its root") from exc


def _safe_relative(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ImmutableReleaseError(f"{label} path is invalid")
    relative = Path(value)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() != value
    ):
        raise ImmutableReleaseError(f"{label} path is not canonical and relative")
    path = (root / relative).resolve()
    _relative_to(root, path, label)
    return path


def _validate_self_digest(value: Mapping[str, Any], field: str, label: str) -> None:
    declared = value.get(field)
    if not isinstance(declared, str) or declared != _digest(value, field):
        raise ImmutableReleaseError(f"{label} checksum is invalid")


def _governance_manifest_path(ai_root: Path, shard_path: Path) -> Path:
    relative = shard_path.resolve().relative_to(ai_root.resolve())
    encoded = "__".join(relative.parts)
    return ai_root / "data_governance/manifests" / f"{encoded}.manifest.json"


def _source_paths(ai_root: Path) -> tuple[dict[str, Any], list[Path], list[dict[str, Any]]]:
    report_path = ai_root / "synthetic_data/reports/current-generation-report.json"
    report = _read_json(report_path, "current generation report")
    _validate_self_digest(report, "reportSha256", "current generation report")
    shard_values = report.get("shards")
    if not isinstance(shard_values, list) or not shard_values:
        raise ImmutableReleaseError("current generation report has no shards")

    paths = {
        report_path,
        ai_root / "synthetic_data/receipts/v1/compile-receipts.jsonl",
        ai_root / "synthetic_data/pipeline-lock.v1.json",
        ai_root / "synthetic_data/toolchains.v1.json",
        ai_root / "data_governance/source-registry.v1.json",
        ai_root / "data_governance/policy.v1.json",
    }
    release_shards: list[dict[str, Any]] = []
    for descriptor in shard_values:
        if not isinstance(descriptor, Mapping):
            raise ImmutableReleaseError("current shard descriptor is invalid")
        source_path = _safe_relative(ai_root, descriptor.get("path"), "shard")
        manifest_path = _governance_manifest_path(ai_root, source_path)
        manifest = _read_json(manifest_path, "current shard manifest")
        _validate_self_digest(manifest, "manifestSha256", "current shard manifest")
        if (
            not source_path.is_file()
            or sha256_file(source_path) != descriptor.get("sha256")
            or source_path.stat().st_size < 1
            or manifest.get("sha256") != descriptor.get("sha256")
            or manifest.get("recordCount") != descriptor.get("recordCount")
        ):
            raise ImmutableReleaseError("current shard bytes or governance identity changed")
        paths.update((source_path, manifest_path))
        release_shards.append(
            {
                "shardId": manifest.get("shardId"),
                "sourcePath": _relative_to(ai_root, source_path, "shard"),
                "sha256": descriptor.get("sha256"),
                "recordCount": descriptor.get("recordCount"),
                "manifestSourcePath": _relative_to(ai_root, manifest_path, "shard manifest"),
                "manifestSha256": sha256_file(manifest_path),
                "sourceIds": list(manifest.get("recordProvenance", {}).get("sourceIds", [])),
            }
        )
        for section in ("producer", "preprocessing", "taskSchema"):
            dependency = manifest.get(section, {}).get("path")
            if dependency:
                paths.add(_safe_relative(ai_root, dependency, f"{section} dependency"))

    pipeline_lock = _read_json(
        ai_root / "synthetic_data/pipeline-lock.v1.json", "synthetic pipeline lock"
    )
    _validate_self_digest(pipeline_lock, "lockSha256", "synthetic pipeline lock")
    for dependency in pipeline_lock.get("dependencies", []):
        if not isinstance(dependency, Mapping):
            raise ImmutableReleaseError("pipeline dependency descriptor is invalid")
        path = _safe_relative(ai_root, dependency.get("path"), "pipeline dependency")
        if not path.is_file() or sha256_file(path) != dependency.get("sha256"):
            raise ImmutableReleaseError("pipeline dependency checksum changed")
        paths.add(path)

    registry = _read_json(
        ai_root / "data_governance/source-registry.v1.json", "source registry"
    )
    required_source_ids = {
        source_id for shard in release_shards for source_id in shard["sourceIds"]
    }
    for source in registry.get("sources", []):
        if not isinstance(source, Mapping) or source.get("sourceId") not in required_source_ids:
            continue
        origin_path = source.get("origin", {}).get("path")
        if origin_path:
            path = _safe_relative(ai_root, origin_path, "source origin")
            if not path.is_file() or sha256_file(path) != source.get("checksum"):
                raise ImmutableReleaseError("source origin checksum changed")
            paths.add(path)

    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise ImmutableReleaseError(f"release source file is missing: {missing[0]}")
    return report, sorted(paths), release_shards


def build_current_release_manifest(
    *, ai_root: Path = AI_ROOT
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Build a deterministic manifest and source map without writing files."""

    root = ai_root.resolve()
    report, paths, shards = _source_paths(root)
    pipeline_lock = _read_json(
        root / "synthetic_data/pipeline-lock.v1.json", "synthetic pipeline lock"
    )
    sources: dict[str, Path] = {}
    files: list[dict[str, Any]] = []
    for source_path in paths:
        source_relative = _relative_to(root, source_path, "release source")
        release_relative = f"content/{source_relative}"
        sources[release_relative] = source_path
        files.append(
            {
                "path": release_relative,
                "sourcePath": source_relative,
                "bytes": source_path.stat().st_size,
                "sha256": sha256_file(source_path),
            }
        )
    files.sort(key=lambda item: item["path"])
    key_inputs = {
        "generatorVersion": report.get("generator", {}).get("version"),
        "pipelineVersion": pipeline_lock.get("pipelineVersion"),
        "pipelineLockSha256": report.get("pipelineLockSha256"),
        "generationReportSha256": report.get("reportSha256"),
        "compileReceiptsSha256": report.get("compileVerification", {}).get(
            "receiptsSha256"
        ),
        "files": [
            {key: item[key] for key in ("path", "bytes", "sha256")} for item in files
        ],
    }
    content_key = hashlib.sha256(_canonical_json(key_inputs)).hexdigest()
    version = str(key_inputs["pipelineVersion"])
    release_path = f"synthetic_data/releases/{version}/{content_key}"
    manifest: dict[str, Any] = {
        "schemaVersion": 1,
        "manifestKind": RELEASE_MANIFEST_KIND,
        "releaseId": f"vf-synthetic-{version}-{content_key[:20]}",
        "releaseVersion": version,
        "contentKey": content_key,
        "releasePath": release_path,
        "keyInputs": key_inputs,
        "source": {
            "generationReportPath": "synthetic_data/reports/current-generation-report.json",
            "generationReportSha256": report.get("reportSha256"),
            "pipelineLockSha256": report.get("pipelineLockSha256"),
            "compileReceiptsSha256": report.get("compileVerification", {}).get(
                "receiptsSha256"
            ),
            "acceptedRecordCount": report.get("summary", {}).get("acceptedRecordCount"),
        },
        "sourceIds": sorted(
            {source_id for shard in shards for source_id in shard["sourceIds"]}
        ),
        "shards": shards,
        "files": files,
        "exactFileSetRequired": True,
        "overwriteAllowed": False,
        "externalBackupVerified": False,
        "releaseManifestSha256": "",
    }
    manifest["releaseManifestSha256"] = _digest(manifest, "releaseManifestSha256")
    return manifest, sources


def verify_release(manifest_path: Path, *, ai_root: Path = AI_ROOT) -> dict[str, Any]:
    """Verify one immutable directory, including its exact file inventory."""

    path = manifest_path.resolve()
    release_root = path.parent
    manifest = _read_json(path, "immutable release manifest")
    if (
        manifest.get("schemaVersion") != 1
        or manifest.get("manifestKind") != RELEASE_MANIFEST_KIND
        or manifest.get("overwriteAllowed") is not False
    ):
        raise ImmutableReleaseError("immutable release manifest contract is invalid")
    _validate_self_digest(manifest, "releaseManifestSha256", "immutable release manifest")
    if _relative_to(ai_root, release_root, "immutable release") != manifest.get("releasePath"):
        raise ImmutableReleaseError("immutable release directory binding is invalid")
    key_inputs = manifest.get("keyInputs")
    if not isinstance(key_inputs, Mapping):
        raise ImmutableReleaseError("immutable release key inputs are missing")
    expected_key = hashlib.sha256(_canonical_json(key_inputs)).hexdigest()
    if manifest.get("contentKey") != expected_key or release_root.name != expected_key:
        raise ImmutableReleaseError("immutable release content key is stale")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ImmutableReleaseError("immutable release file inventory is empty")
    declared: set[str] = set()
    total_bytes = 0
    for descriptor in files:
        if not isinstance(descriptor, Mapping):
            raise ImmutableReleaseError("immutable release file descriptor is invalid")
        relative = descriptor.get("path")
        file_path = _safe_relative(release_root, relative, "immutable release file")
        if relative in declared:
            raise ImmutableReleaseError("immutable release file path is duplicated")
        declared.add(str(relative))
        size = descriptor.get("bytes")
        digest = descriptor.get("sha256")
        if (
            not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or not file_path.is_file()
            or file_path.stat().st_size != size
            or sha256_file(file_path) != digest
        ):
            raise ImmutableReleaseError(f"immutable release file changed: {relative}")
        total_bytes += size
    actual = {
        item.relative_to(release_root).as_posix()
        for item in release_root.rglob("*")
        if item.is_file() and item != path
    }
    if actual != declared:
        raise ImmutableReleaseError("immutable release contains an undeclared or missing file")
    expected_key_files = [
        {key: item.get(key) for key in ("path", "bytes", "sha256")} for item in files
    ]
    if key_inputs.get("files") != expected_key_files:
        raise ImmutableReleaseError("immutable release inventory is not content-key bound")
    retained_report = _read_json(
        release_root / "content/synthetic_data/reports/current-generation-report.json",
        "retained generation report",
    )
    _validate_self_digest(retained_report, "reportSha256", "retained generation report")
    retained_receipts = release_root / "content/synthetic_data/receipts/v1/compile-receipts.jsonl"
    source = manifest.get("source", {})
    if (
        retained_report.get("reportSha256") != source.get("generationReportSha256")
        or retained_report.get("pipelineLockSha256") != source.get("pipelineLockSha256")
        or retained_report.get("summary", {}).get("acceptedRecordCount")
        != source.get("acceptedRecordCount")
        or sha256_file(retained_receipts) != source.get("compileReceiptsSha256")
        or key_inputs.get("generationReportSha256")
        != source.get("generationReportSha256")
        or key_inputs.get("pipelineLockSha256") != source.get("pipelineLockSha256")
        or key_inputs.get("compileReceiptsSha256") != source.get("compileReceiptsSha256")
    ):
        raise ImmutableReleaseError("immutable release source binding is invalid")
    return {
        "releaseId": manifest.get("releaseId"),
        "releasePath": manifest.get("releasePath"),
        "contentKey": expected_key,
        "releaseManifestSha256": manifest.get("releaseManifestSha256"),
        "fileCount": len(files),
        "totalBytes": total_bytes,
        "shardCount": len(manifest.get("shards", [])),
        "acceptedRecordCount": manifest.get("source", {}).get("acceptedRecordCount"),
        "externalBackupVerified": manifest.get("externalBackupVerified") is True,
        "manifest": manifest,
    }


def _pointer_for(manifest: Mapping[str, Any]) -> dict[str, Any]:
    pointer: dict[str, Any] = {
        "schemaVersion": 1,
        "pointerKind": CURRENT_POINTER_KIND,
        "releaseId": manifest.get("releaseId"),
        "releasePath": manifest.get("releasePath"),
        "contentKey": manifest.get("contentKey"),
        "releaseManifestSha256": manifest.get("releaseManifestSha256"),
        "pointerSha256": "",
    }
    pointer["pointerSha256"] = _digest(pointer, "pointerSha256")
    return pointer


def publish_current_release(
    *,
    ai_root: Path = AI_ROOT,
    releases_root: Path | None = None,
    pointer_path: Path | None = None,
) -> dict[str, Any]:
    """Publish current bytes once, or verify the identical existing release."""

    root = ai_root.resolve()
    release_base = (releases_root or root / "synthetic_data/releases").resolve()
    pointer = (pointer_path or release_base / "CURRENT.json").resolve()
    manifest, sources = build_current_release_manifest(ai_root=root)
    target = (root / str(manifest["releasePath"])).resolve()
    if releases_root is not None:
        target = release_base / str(manifest["releaseVersion"]) / str(manifest["contentKey"])
        manifest["releasePath"] = _relative_to(root, target, "immutable release")
        manifest["releaseManifestSha256"] = _digest(manifest, "releaseManifestSha256")
    _relative_to(release_base, target, "immutable release")
    manifest_path = target / RELEASE_MANIFEST_NAME
    if target.exists():
        summary = verify_release(manifest_path, ai_root=root)
        if summary["releaseManifestSha256"] != manifest["releaseManifestSha256"]:
            raise ImmutableReleaseError("existing immutable release differs and cannot be overwritten")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".publishing")
        if temporary.exists():
            raise ImmutableReleaseError("stale immutable release publication directory exists")
        try:
            temporary.mkdir()
            for relative, source in sources.items():
                destination = temporary / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
            (temporary / RELEASE_MANIFEST_NAME).write_bytes(_json_bytes(manifest))
            temporary.replace(target)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
        summary = verify_release(manifest_path, ai_root=root)
    _atomic_write(pointer, _pointer_for(manifest))
    return summary


def verify_current_release(
    *, ai_root: Path = AI_ROOT, pointer_path: Path | None = None
) -> dict[str, Any]:
    """Verify the pointer and prove it binds the exact current mutable aliases."""

    root = ai_root.resolve()
    pointer_file = (pointer_path or root / "synthetic_data/releases/CURRENT.json").resolve()
    pointer = _read_json(pointer_file, "current immutable release pointer")
    if pointer.get("schemaVersion") != 1 or pointer.get("pointerKind") != CURRENT_POINTER_KIND:
        raise ImmutableReleaseError("current immutable release pointer contract is invalid")
    _validate_self_digest(pointer, "pointerSha256", "current immutable release pointer")
    release_root = _safe_relative(root, pointer.get("releasePath"), "current release")
    summary = verify_release(release_root / RELEASE_MANIFEST_NAME, ai_root=root)
    for field in ("releaseId", "contentKey", "releaseManifestSha256"):
        if pointer.get(field) != summary.get(field):
            raise ImmutableReleaseError("current immutable release pointer is stale")
    expected, _ = build_current_release_manifest(ai_root=root)
    if expected.get("contentKey") != pointer.get("contentKey"):
        raise ImmutableReleaseError("current aliases are not retained by the current pointer")
    return summary


def preserve_current_release_if_needed(*, ai_root: Path = AI_ROOT) -> dict[str, Any] | None:
    """Retain current bytes before a mutable alias or pipeline lock can change."""

    root = ai_root.resolve()
    report_path = root / "synthetic_data/reports/current-generation-report.json"
    if not report_path.is_file():
        return None
    try:
        return verify_current_release(ai_root=root)
    except ImmutableReleaseError:
        pointer_path = root / "synthetic_data/releases/CURRENT.json"
        if pointer_path.is_file():
            try:
                pointer = _read_json(pointer_path, "current immutable release pointer")
                _validate_self_digest(
                    pointer, "pointerSha256", "current immutable release pointer"
                )
                release_root = _safe_relative(
                    root, pointer.get("releasePath"), "current release"
                )
                summary = verify_release(
                    release_root / RELEASE_MANIFEST_NAME, ai_root=root
                )
                report = _read_json(report_path, "current generation report")
                manifest = summary["manifest"]
                payload_matches = (
                    manifest.get("source", {}).get("generationReportSha256")
                    == report.get("reportSha256")
                    and manifest.get("source", {}).get("compileReceiptsSha256")
                    == report.get("compileVerification", {}).get("receiptsSha256")
                    and all(
                        sha256_file(root / str(shard["sourcePath"]))
                        == shard.get("sha256")
                        and sha256_file(root / str(shard["manifestSourcePath"]))
                        == shard.get("manifestSha256")
                        for shard in manifest.get("shards", [])
                    )
                )
                if payload_matches:
                    return summary
            except (ImmutableReleaseError, OSError, KeyError, TypeError):
                pass
        return publish_current_release(ai_root=root)


def verify_all_releases(*, ai_root: Path = AI_ROOT) -> list[dict[str, Any]]:
    root = ai_root.resolve()
    manifests = sorted(
        (root / "synthetic_data/releases").glob(f"*/*/{RELEASE_MANIFEST_NAME}")
    )
    if not manifests:
        raise ImmutableReleaseError("no immutable synthetic release is retained")
    return [verify_release(path, ai_root=root) for path in manifests]


def resolve_tokenizer_shard(
    descriptor: Mapping[str, Any], *, ai_root: Path = AI_ROOT
) -> dict[str, Any]:
    """Resolve historical tokenizer lineage to an exact immutable shard copy."""

    root = ai_root.resolve()
    matches: list[dict[str, Any]] = []
    for summary in verify_all_releases(ai_root=root):
        manifest = summary["manifest"]
        for shard in manifest.get("shards", []):
            if not isinstance(shard, Mapping):
                continue
            if all(
                (
                    shard.get("shardId") == descriptor.get("shardId"),
                    shard.get("sourcePath") == descriptor.get("path"),
                    shard.get("sha256") == descriptor.get("sha256"),
                    shard.get("recordCount") == descriptor.get("recordCount"),
                    shard.get("manifestSourcePath") == descriptor.get("manifestPath"),
                    shard.get("manifestSha256") == descriptor.get("manifestSha256"),
                )
            ):
                release_root = root / str(manifest["releasePath"])
                content_root = release_root / "content"
                matches.append(
                    {
                        "releaseId": manifest["releaseId"],
                        "releaseManifestSha256": manifest["releaseManifestSha256"],
                        "contentRoot": content_root,
                        "shardPath": content_root / str(shard["sourcePath"]),
                        "manifestPath": content_root / str(shard["manifestSourcePath"]),
                        "sourceRegistryPath": content_root
                        / "data_governance/source-registry.v1.json",
                        "dataPolicyPath": content_root / "data_governance/policy.v1.json",
                    }
                )
    if not matches:
        raise ImmutableReleaseError(
            f"no immutable release resolves tokenizer shard {descriptor.get('shardId')}"
        )
    matches.sort(key=lambda item: str(item["releaseId"]))
    return matches[0]
