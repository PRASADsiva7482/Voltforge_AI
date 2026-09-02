from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from data_governance.governance import analyze_source_removal
from synthetic_data.immutable_release import (
    AI_ROOT,
    ImmutableReleaseError,
    publish_current_release,
    resolve_tokenizer_shard,
    verify_all_releases,
    verify_current_release,
    verify_release,
)
from tools.build_tokenizer import _check_retained_tokenizer_release


TOKENIZER_V1_1_MANIFEST = (
    AI_ROOT / "model/tokenizers/vfdlm-byte-bpe-v1.1.0/tokenizer_manifest.json"
)
TOKENIZER_V1_0_MANIFEST = (
    AI_ROOT / "model/tokenizers/vfdlm-byte-bpe-v1.0.0/tokenizer_manifest.json"
)


def _tokenizer_shards(path: Path) -> list[dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8"))["lineage"]["shards"]


def _first_release() -> dict[str, object]:
    return verify_all_releases()[0]


def _restore_release_as_workspace(tmp_path: Path) -> Path:
    summary = _first_release()
    release_source = AI_ROOT / str(summary["releasePath"])
    content_source = release_source / "content"
    shutil.copytree(content_source, tmp_path, dirs_exist_ok=True)
    return tmp_path


def test_current_tokenizer_lineage_resolves_only_to_immutable_bytes() -> None:
    resolved = [resolve_tokenizer_shard(item) for item in _tokenizer_shards(TOKENIZER_V1_1_MANIFEST)]

    assert len(resolved) == 4
    assert len({item["releaseId"] for item in resolved}) == 1
    assert all("synthetic_data\\releases" in str(item["shardPath"]) for item in resolved)


def test_approved_tokenizer_retrain_check_uses_retained_lineage_without_rewrite() -> None:
    before = TOKENIZER_V1_1_MANIFEST.read_bytes()

    report = _check_retained_tokenizer_release()

    assert report["decision"] == "pass"
    assert TOKENIZER_V1_1_MANIFEST.read_bytes() == before


def test_historical_unavailable_lineage_fails_closed() -> None:
    for descriptor in _tokenizer_shards(TOKENIZER_V1_0_MANIFEST):
        with pytest.raises(ImmutableReleaseError, match="no immutable release resolves"):
            resolve_tokenizer_shard(descriptor)


def test_release_restore_verifies_exact_inventory(tmp_path: Path) -> None:
    for summary in verify_all_releases():
        source = AI_ROOT / str(summary["releasePath"])
        restored = tmp_path / str(summary["releasePath"])
        shutil.copytree(source, restored)

        verified = verify_release(restored / "release-manifest.json", ai_root=tmp_path)

        assert verified["contentKey"] == summary["contentKey"]
        assert verified["fileCount"] == summary["fileCount"]
        assert verified["totalBytes"] == summary["totalBytes"]


def test_release_verifier_rejects_changed_and_extra_files(tmp_path: Path) -> None:
    summary = _first_release()
    source = AI_ROOT / str(summary["releasePath"])
    restored = tmp_path / str(summary["releasePath"])
    shutil.copytree(source, restored)
    shard = next((restored / "content/synthetic_data/shards/v1").glob("*.jsonl"))
    shard.write_bytes(shard.read_bytes() + b"\n")
    with pytest.raises(ImmutableReleaseError, match="file changed"):
        verify_release(restored / "release-manifest.json", ai_root=tmp_path)

    shutil.rmtree(restored)
    shutil.copytree(source, restored)
    (restored / "extra.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(ImmutableReleaseError, match="undeclared"):
        verify_release(restored / "release-manifest.json", ai_root=tmp_path)


def test_publish_is_idempotent_but_never_repairs_existing_release(tmp_path: Path) -> None:
    workspace = _restore_release_as_workspace(tmp_path)
    first = publish_current_release(ai_root=workspace)
    second = publish_current_release(ai_root=workspace)
    assert second["releaseManifestSha256"] == first["releaseManifestSha256"]
    assert verify_current_release(ai_root=workspace)["contentKey"] == first["contentKey"]

    release_root = workspace / str(first["releasePath"])
    shard = next((release_root / "content/synthetic_data/shards/v1").glob("*.jsonl"))
    shard.write_bytes(shard.read_bytes() + b"tamper")
    with pytest.raises(ImmutableReleaseError, match="file changed"):
        publish_current_release(ai_root=workspace)


def test_source_removal_reports_retained_immutable_release_paths() -> None:
    report = analyze_source_removal("vf-src-verified-synthetic-pipeline-v1")

    retained = [
        item for item in report["affectedShards"] if item["retainedImmutableRelease"]
    ]
    assert len(retained) == len(verify_all_releases()) * 4
    assert all("synthetic_data/releases/" in item["manifestPath"] for item in retained)
