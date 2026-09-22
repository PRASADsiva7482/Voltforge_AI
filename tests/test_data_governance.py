from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


AI_ROOT = Path(__file__).resolve().parents[1]

from data_governance.governance import (
    DEFAULT_MANIFEST_DIR,
    DataGovernanceError,
    analyze_source_removal,
    audit_current_corpora,
    default_manifest_path,
    load_policy,
    load_source_registry,
    require_approved_shard,
    require_approved_source,
    sha256_file,
    source_approval_decision,
    write_approved_shard_manifest,
)
from task_schema.adapters import legacy_example_to_task_record
from task_schema.io import write_task_shard


def test_registry_has_complete_evidence_and_current_repository_checksums() -> None:
    registry = load_source_registry()
    assert registry["version"] == "1.2.0"
    assert len(registry["sources"]) == 8
    for source in registry["sources"]:
        assert source["revision"]
        assert source["license"]["status"]
        assert source["privacy"]["classification"]
        assert source["preprocessing"]["version"]
        assert source["deletionProcedure"]
        origin_path = source["origin"].get("path")
        if origin_path:
            assert source["checksum"] == sha256_file(AI_ROOT / origin_path)


def test_policy_denies_private_unlicensed_web_and_unregistered_sources() -> None:
    policy = load_policy()
    private_source = {
        "sourceId": "vf-src-test-private",
        "kind": "private-user-project",
        "license": {"status": "approved"},
        "privacy": {"containsPrivateUserData": True, "trainingAllowed": True},
        "allowedUses": ["training"],
        "prohibitedUses": [],
        "approval": {"status": "approved", "approvedUses": ["training"]},
        "origin": {"path": None},
        "checksum": None,
    }
    web_source = {
        **private_source,
        "sourceId": "vf-src-test-web",
        "kind": "web-scrape",
        "license": {"status": "unverified"},
        "privacy": {"containsPrivateUserData": False, "trainingAllowed": True},
    }
    assert source_approval_decision(private_source, "training", policy=policy)["allowed"] is False
    assert source_approval_decision(web_source, "training", policy=policy)["allowed"] is False
    assert policy["privateConsentWorkflowImplemented"] is False
    with pytest.raises(DataGovernanceError) as error:
        require_approved_source("vf-src-does-not-exist", "training")
    assert error.value.code == "DATA_SOURCE_NOT_REGISTERED"


def test_current_corpus_is_fully_cataloged_and_quarantined() -> None:
    report = audit_current_corpora(write=False)
    assert report["summary"] == {
        "retainedDatasetCount": 12,
        "recordCount": 17814,
        "approvedTrainingShardCount": 0,
        "approvedRuntimeRetrievalCount": 0,
        "quarantinedDatasetCount": 12,
        "catalogCoverageComplete": True,
        "decision": "pass",
    }
    assert all(item["trainingEligible"] is False for item in report["datasets"])
    assert all(item["runtimeRetrievalEligible"] is False for item in report["datasets"])
    with pytest.raises(DataGovernanceError) as error:
        require_approved_shard(AI_ROOT / "model" / "artifacts" / "master_domain_dataset.jsonl")
    assert error.value.code == "DATA_SHARD_USE_DENIED"


def test_approved_future_shard_has_record_level_inherited_lineage_and_detects_tampering(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "future.jsonl"
    records = [
        legacy_example_to_task_record(
            {"task": "chat_qa", "prompt": f"[SYS] Be safe. [USER] p{index}", "completion": f"c{index}"},
            source_ids=("vf-src-project-domain-generator-v1",),
            generator_id="vf-domain-corpus-generator",
            generator_version="2.0.0",
        )
        for index in (1, 2)
    ]
    write_task_shard(shard, records)
    manifest_path = write_approved_shard_manifest(
        shard,
        source_ids=("vf-src-project-domain-generator-v1",),
        producer_id="vf-domain-corpus-generator",
        producer_version="1.0.0",
        producer_path="model/generate_domain_corpus.py",
    )
    manifest = require_approved_shard(shard, manifest_path=manifest_path)
    assert manifest["status"] == "approved"
    assert manifest["recordCount"] == 2
    assert manifest["recordFormat"] == "vf-task-record-jsonl-v1"
    assert manifest["taskSchema"]["version"] == "1.0.0"
    assert manifest["recordProvenance"] == {
        "mode": "shard-inherited",
        "selector": "all-records",
        "sourceIds": ["vf-src-project-domain-generator-v1"],
        "statement": "Every non-empty record in this shard inherits this immutable source and producer snapshot.",
    }

    shard.write_text(shard.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(DataGovernanceError) as error:
        require_approved_shard(shard, manifest_path=manifest_path)
    assert error.value.code == "DATA_SHARD_CHECKSUM_MISMATCH"


def test_source_removal_finds_shards_and_artifact_retraining_impact(tmp_path: Path) -> None:
    source_id = "vf-src-project-domain-generator-v1"
    manifests = sorted(DEFAULT_MANIFEST_DIR.glob("*.manifest.json"))
    affected_shard_id = json.loads(manifests[1].read_text(encoding="utf-8"))["shardId"]
    model_registry = tmp_path / "active_model.json"
    model_registry.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "activeArtifactId": "vfdlm-g1-edge-v1.0.0",
                "artifacts": [
                    {
                        "artifactId": "vfdlm-g1-edge-v1.0.0",
                        "releaseStatus": "approved",
                        "dataLineage": {
                            "sourceIds": [source_id],
                            "shardIds": [affected_shard_id],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    impact = analyze_source_removal(source_id, model_registry_path=model_registry)
    assert len(impact["affectedShards"]) == 8
    assert impact["affectedArtifacts"] == [
        {
            "artifactId": "vfdlm-g1-edge-v1.0.0",
            "releaseStatus": "approved",
            "action": "deactivate-delete-and-retrain",
        }
    ]
    assert impact["requiresRetraining"] is True
    assert impact["potentiallyAffectedLegacyArtifacts"]


def test_generation_training_and_retrieval_paths_enforce_governance() -> None:
    required = {
        "model/generate_dataset.py": "require_approved_shard",
        "model/generate_domain_corpus.py": "write_approved_shard_manifest",
        "model/train.py": "write_approved_shard_manifest",
        "model/train_chunks.py": "require_approved_shard",
        "engine/reasoning.py": "require_approved_shard",
    }
    for relative_path, marker in required.items():
        source = (AI_ROOT / relative_path).read_text(encoding="utf-8")
        assert "data_governance" in source
        assert marker in source


def test_checked_in_audit_and_manifest_integrity_are_current() -> None:
    report_path = AI_ROOT / "data_governance" / "reports" / "current-corpus-audit.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    payload = {key: value for key, value in report.items() if key != "reportSha256"}
    actual = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    assert report["reportSha256"] == actual
    assert len(list(DEFAULT_MANIFEST_DIR.glob("*.manifest.json"))) == 16
    assert default_manifest_path(AI_ROOT / "dataset.txt").is_file()
