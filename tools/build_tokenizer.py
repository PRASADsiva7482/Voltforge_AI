"""Build or no-write verify the VFAI-010 tokenizer release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


AI_ROOT = Path(__file__).resolve().parents[1]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

from data_governance.governance import require_approved_shard  # noqa: E402
from synthetic_data.immutable_release import resolve_tokenizer_shard  # noqa: E402
import tokenizer_training.pipeline as tokenizer_pipeline  # noqa: E402


def _check_retained_tokenizer_release() -> dict[str, Any]:
    """Retrain-check an approved tokenizer against its retained immutable bytes."""

    manifest = json.loads(tokenizer_pipeline.MANIFEST_PATH.read_text(encoding="utf-8"))
    descriptors = manifest.get("lineage", {}).get("shards", [])
    retained = [resolve_tokenizer_shard(item) for item in descriptors]
    release_ids = {item["releaseId"] for item in retained}
    content_roots = {Path(item["contentRoot"]) for item in retained}
    if len(retained) != len(descriptors) or len(release_ids) != 1 or len(content_roots) != 1:
        raise RuntimeError("approved tokenizer lineage does not resolve to one immutable release")
    content_root = next(iter(content_roots))
    retained_manifest_by_name = {
        Path(str(descriptor["path"])).name: Path(item["manifestPath"])
        for descriptor, item in zip(descriptors, retained, strict=True)
    }
    lineage_manifest_by_name = {
        Path(str(descriptor["path"])).name: AI_ROOT / str(descriptor["manifestPath"])
        for descriptor in descriptors
    }
    lineage_manifest_sha_by_name = {
        Path(str(descriptor["path"])).name: str(descriptor["manifestSha256"])
        for descriptor in descriptors
    }
    original_shard_root = tokenizer_pipeline.SHARD_ROOT
    original_manifest_path = tokenizer_pipeline.default_manifest_path
    original_require = tokenizer_pipeline.require_approved_shard
    original_sha256_file = tokenizer_pipeline.sha256_file

    def retained_manifest_path(path: str | Path) -> Path:
        return lineage_manifest_by_name[Path(path).name]

    def retained_require(path: str | Path, usage: str, **kwargs: Any) -> dict[str, Any]:
        return require_approved_shard(
            path,
            usage,
            manifest_path=retained_manifest_by_name[Path(path).name],
            registry_path=content_root / "data_governance/source-registry.v1.json",
            policy_path=content_root / "data_governance/policy.v1.json",
            dependency_root=content_root,
            source_root=content_root,
        )

    def retained_sha256_file(path: str | Path) -> str:
        candidate = Path(path)
        for shard_name, manifest_path in lineage_manifest_by_name.items():
            if candidate.resolve() == manifest_path.resolve():
                return lineage_manifest_sha_by_name[shard_name]
        return original_sha256_file(candidate)

    try:
        tokenizer_pipeline.SHARD_ROOT = content_root / "synthetic_data/shards/v1"
        tokenizer_pipeline.default_manifest_path = retained_manifest_path
        tokenizer_pipeline.require_approved_shard = retained_require
        tokenizer_pipeline.sha256_file = retained_sha256_file
        return tokenizer_pipeline.check_tokenizer_release()
    finally:
        tokenizer_pipeline.SHARD_ROOT = original_shard_root
        tokenizer_pipeline.default_manifest_path = original_manifest_path
        tokenizer_pipeline.require_approved_shard = original_require
        tokenizer_pipeline.sha256_file = original_sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="Train and write the immutable release")
    mode.add_argument("--check", action="store_true", help="Retrain and compare every artifact without writes")
    args = parser.parse_args()
    if args.write and tokenizer_pipeline.MANIFEST_PATH.is_file():
        existing = json.loads(tokenizer_pipeline.MANIFEST_PATH.read_text(encoding="utf-8"))
        if existing.get("releaseStatus") == "approved":
            parser.error(
                "refusing to overwrite an approved tokenizer release; publish a new versioned directory"
            )
    report = (
        tokenizer_pipeline.write_tokenizer_release()
        if args.write
        else _check_retained_tokenizer_release()
    )
    print(json.dumps({
        "decision": report["decision"],
        "vocabSize": report["tokenizer"]["vocabSize"],
        "mergeCount": report["tokenizer"]["mergeCount"],
        "tokenReductionPercent": report["compression"]["tokenReductionPercent"],
        "vocabularySizeReductionPercent": report["vocabularyCost"]["sizeReductionPercent"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
