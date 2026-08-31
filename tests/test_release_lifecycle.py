from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from model.registry_manager import (
    RuntimeContract,
    canonical_json_bytes,
    initialize_signing_key,
    json_file_bytes,
    register_artifact,
    sha256_bytes,
    sha256_file,
    sign_document,
    verify_artifact_directory,
)
from model.release_lifecycle import (
    ReleaseLifecycleError,
    activate_stable_release,
    create_candidate_release,
    current_release_compatibility,
    promote_candidate_to_stable,
    rollback_reasons,
    rollback_stable_release,
    validate_release_compatibility,
    verify_release_record,
)


class TestReleaseLifecycle(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.registry_dir = self.root / "registry"
        self.registry_dir.mkdir()
        self.registry_path = self.registry_dir / "active_model.json"
        self.trust_store = self.registry_dir / "trust" / "trusted-keys.json"
        self.private_key = self.root / "private" / "signing-key.pem"
        self.key_id = "vf-test-release-key-v1"
        initialize_signing_key(self.private_key, self.trust_store, key_id=self.key_id)
        self.registry_path.write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "identityContract": {"productName": "VoltForge AI", "familyName": "VoltForge Domain Language Model", "familySlug": "vfdlm"},
                    "activeArtifactId": None,
                    "releaseStatus": "none",
                    "state": "no-approved-artifact",
                    "reason": "No test artifact is approved.",
                    "artifacts": [],
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _artifact(self, artifact_id: str, *, release_status: str = "approved", activation_eligible: bool = True):
        root = self.registry_dir / "artifacts" / artifact_id
        (root / "model").mkdir(parents=True)
        (root / "model" / "weights.pt").write_bytes(b"signed-release-test-weights")
        (root / "model" / "config.json").write_text(json.dumps({"maxSequenceLength": 16, "vocabSize": 32}), encoding="utf-8")
        compatibility = RuntimeContract.current().as_compatibility()
        (root / "compatibility.json").write_bytes(json_file_bytes(compatibility))
        (root / "generation-policy.json").write_bytes(
            json_file_bytes(
                {
                    "schemaVersion": 1,
                    "artifactId": artifact_id,
                    "releaseStatus": release_status,
                    "activationEligible": activation_eligible,
                    "servingEnabled": release_status == "approved" and activation_eligible,
                    "promptLoggingAllowed": False,
                }
            )
        )
        files = [
            {
                "path": path.relative_to(root).as_posix(),
                "role": "test",
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(item for item in root.rglob("*") if item.is_file())
        ]
        manifest = {
            "schemaVersion": 2,
            "artifactKind": "vfdlm-gen1-local-model-package",
            "artifactId": artifact_id,
            "immutable": True,
            "runtime": "pytorch-gen1-v1",
            "compatibilityPath": "compatibility.json",
            "generationPolicyPath": "generation-policy.json",
            "packageSha256": sha256_bytes(canonical_json_bytes(files)),
            "model": {"parameterCount": 128, "contextLength": 16, "quantization": "fp32"},
            "releaseState": {"releaseStatus": release_status, "activationEligible": activation_eligible},
            "files": files,
        }
        signed = sign_document(manifest, digest_field="manifestSha256", private_key_path=self.private_key, key_id=self.key_id)
        (root / "artifact-manifest.json").write_bytes(json_file_bytes(signed))
        return verify_artifact_directory(root, trust_store_path=self.trust_store)

    def _register(self, artifact):
        return register_artifact(
            artifact,
            registry_path=self.registry_path,
            trust_store_path=self.trust_store,
            private_key_path=self.private_key,
            key_id=self.key_id,
        )

    def _scorecard(self, artifact) -> dict[str, object]:
        return {
            "reportSha256": "a" * 64,
            "scorecardSha256": "b" * 64,
            "artifactManifestSha256": artifact.manifest_sha256,
            "releaseDecision": "pass",
            "criticalCaseFailures": [],
            "failedMetricIds": [],
        }

    def _canary(self, scorecard: dict[str, object]) -> dict[str, object]:
        return {
            "status": "pass",
            "scorecardSha256": scorecard["scorecardSha256"],
            "observations": 100,
            "errorRate": 0.0,
            "contractFailureRate": 0.0,
            "safetyFailureCount": 0,
            "p95LatencyMs": 1000,
        }

    def test_passing_candidate_promotes_and_activates_with_signed_release_id(self) -> None:
        artifact = self._artifact("vfdlm-g1-edge-v1.0.0-test")
        registered = self._register(artifact)
        scorecard = self._scorecard(artifact)
        candidate_path = self.root / "candidate.json"
        candidate = create_candidate_release(
            artifact.artifact_id,
            scorecard=scorecard,
            registry_path=self.registry_path,
            trust_store_path=self.trust_store,
            private_key_path=self.private_key,
            key_id=self.key_id,
            output_path=candidate_path,
        )
        self.assertEqual(candidate["state"], "candidate")
        stable_path = self.root / "stable.json"
        stable = promote_candidate_to_stable(
            candidate_path,
            canary=self._canary(scorecard),
            registry_path=self.registry_path,
            trust_store_path=self.trust_store,
            private_key_path=self.private_key,
            key_id=self.key_id,
            output_path=stable_path,
        )
        self.assertEqual(stable["state"], "stable")
        verified = verify_release_record(stable_path, trust_store_path=self.trust_store, registry_path=self.registry_path)
        self.assertEqual(verified["releaseId"], stable["releaseId"])
        active = activate_stable_release(
            stable_path,
            registry_path=self.registry_path,
            trust_store_path=self.trust_store,
            private_key_path=self.private_key,
            key_id=self.key_id,
            expected_revision=registered["revision"],
        )
        self.assertEqual(active["activeArtifactId"], artifact.artifact_id)
        self.assertEqual(active["lastTransaction"]["releaseId"], stable["releaseId"])

    def test_experimental_artifact_cannot_become_stable(self) -> None:
        artifact = self._artifact(
            "vfdlm-g1-edge-v1.0.1-experimental",
            release_status="experimental",
            activation_eligible=False,
        )
        self._register(artifact)
        scorecard = self._scorecard(artifact)
        candidate_path = self.root / "experimental-candidate.json"
        create_candidate_release(
            artifact.artifact_id,
            scorecard=scorecard,
            registry_path=self.registry_path,
            trust_store_path=self.trust_store,
            private_key_path=self.private_key,
            key_id=self.key_id,
            output_path=candidate_path,
        )
        with self.assertRaises(ReleaseLifecycleError) as raised:
            promote_candidate_to_stable(
                candidate_path,
                canary=self._canary(scorecard),
                registry_path=self.registry_path,
                trust_store_path=self.trust_store,
                private_key_path=self.private_key,
                key_id=self.key_id,
                output_path=self.root / "should-not-exist.json",
            )
        self.assertEqual(raised.exception.code, "MODEL_ARTIFACT_NOT_APPROVED")

    def test_two_stable_releases_roll_back_to_previous_signed_release(self) -> None:
        first = self._artifact("vfdlm-g1-edge-v1.1.0-first")
        first_registered = self._register(first)
        first_scorecard = self._scorecard(first)
        first_candidate_path = self.root / "first.candidate.json"
        create_candidate_release(first.artifact_id, scorecard=first_scorecard, registry_path=self.registry_path, trust_store_path=self.trust_store, private_key_path=self.private_key, key_id=self.key_id, output_path=first_candidate_path)
        first_stable_path = self.root / f"{first.artifact_id}.stable.json"
        first_stable = promote_candidate_to_stable(first_candidate_path, canary=self._canary(first_scorecard), registry_path=self.registry_path, trust_store_path=self.trust_store, private_key_path=self.private_key, key_id=self.key_id, output_path=first_stable_path)
        active_first = activate_stable_release(first_stable_path, registry_path=self.registry_path, trust_store_path=self.trust_store, private_key_path=self.private_key, key_id=self.key_id, expected_revision=first_registered["revision"])

        second = self._artifact("vfdlm-g1-edge-v1.1.1-second")
        second_registered = self._register(second)
        second_scorecard = self._scorecard(second)
        second_candidate_path = self.root / "second.candidate.json"
        create_candidate_release(second.artifact_id, scorecard=second_scorecard, registry_path=self.registry_path, trust_store_path=self.trust_store, private_key_path=self.private_key, key_id=self.key_id, output_path=second_candidate_path)
        second_stable_path = self.root / f"{second.artifact_id}.stable.json"
        second_stable = promote_candidate_to_stable(second_candidate_path, canary=self._canary(second_scorecard), registry_path=self.registry_path, trust_store_path=self.trust_store, private_key_path=self.private_key, key_id=self.key_id, output_path=second_stable_path)
        active_second = activate_stable_release(second_stable_path, registry_path=self.registry_path, trust_store_path=self.trust_store, private_key_path=self.private_key, key_id=self.key_id, expected_revision=second_registered["revision"])
        self.assertEqual(active_second["activeArtifactId"], second.artifact_id)

        rolled_back = rollback_stable_release(second_stable_path, registry_path=self.registry_path, trust_store_path=self.trust_store, private_key_path=self.private_key, key_id=self.key_id, expected_revision=active_second["revision"])
        self.assertEqual(rolled_back["activeArtifactId"], first.artifact_id)
        self.assertEqual(rolled_back["lastTransaction"]["releaseId"], first_stable["releaseId"])
        self.assertEqual(active_first["activeArtifactId"], first.artifact_id)

    def test_compatibility_drift_requires_explicit_migration(self) -> None:
        compatibility = current_release_compatibility()
        compatibility["api"]["contractVersion"] = "2.0.0"
        with self.assertRaises(ReleaseLifecycleError) as raised:
            validate_release_compatibility(compatibility)
        self.assertEqual(raised.exception.code, "MODEL_RELEASE_API_MIGRATION_REQUIRED")

    def test_rollback_reasons_are_bounded_and_deterministic(self) -> None:
        reasons = rollback_reasons(
            {
                "safetyFailureCount": 1,
                "contractFailureRate": 0.1,
                "errorRate": 0.03,
                "p95LatencyMs": 6000,
                "artifactIntegrity": False,
                "compatibility": False,
            }
        )
        self.assertEqual(
            reasons,
            ["SAFETY_FAILURE", "CONTRACT_FAILURE_RATE", "ERROR_RATE", "P95_LATENCY", "ARTIFACT_INTEGRITY", "COMPATIBILITY_FAILURE"],
        )

    def test_malformed_rollback_metrics_fail_closed(self) -> None:
        with self.assertRaises(ReleaseLifecycleError) as raised:
            rollback_reasons({"errorRate": "not-a-number"})
        self.assertEqual(raised.exception.code, "MODEL_RELEASE_ROLLBACK_METRICS_INVALID")


if __name__ == "__main__":
    unittest.main()
