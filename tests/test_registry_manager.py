import json
from pathlib import Path
import tempfile
import unittest

from model.registry_manager import (
    RegistryManagerError,
    RuntimeContract,
    activate_artifact,
    canonical_json_bytes,
    initialize_signing_key,
    json_file_bytes,
    recover_latest_registry_history,
    register_artifact,
    retire_artifact,
    rollback_registry,
    sha256_bytes,
    sha256_file,
    sign_document,
    verify_artifact_directory,
    verify_registry,
    verify_signed_document,
)


class TestSignedRegistryLifecycle(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.registry_dir = self.root / "registry"
        self.registry_dir.mkdir()
        self.registry_path = self.registry_dir / "active_model.json"
        self.trust_store = self.registry_dir / "trust" / "trusted-keys.json"
        self.private_key = self.root / "private" / "signing-key.pem"
        self.key_id = "vf-test-registry-key-v1"
        initialize_signing_key(
            self.private_key,
            self.trust_store,
            key_id=self.key_id,
        )
        self.registry_path.write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "identityContract": {
                        "productName": "VoltForge AI",
                        "familyName": "VoltForge Domain Language Model",
                        "familySlug": "vfdlm",
                    },
                    "activeArtifactId": None,
                    "releaseStatus": "none",
                    "state": "no-approved-artifact",
                    "reason": "No test artifact is approved.",
                    "artifacts": [],
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def _artifact(
        self,
        artifact_id: str,
        *,
        release_status: str = "approved",
        activation_eligible: bool = True,
        compatible: bool = True,
    ):
        root = self.registry_dir / "artifacts" / artifact_id
        (root / "model").mkdir(parents=True)
        (root / "model" / "weights.pt").write_bytes(b"signed-test-weights")
        (root / "model" / "config.json").write_text(
            json.dumps({"maxSequenceLength": 16, "vocabSize": 32}), encoding="utf-8"
        )
        compatibility = RuntimeContract.current().as_compatibility()
        if not compatible:
            compatibility["runtimeInterfaceVersion"] += 1
        (root / "compatibility.json").write_bytes(json_file_bytes(compatibility))
        policy = {
            "schemaVersion": 1,
            "artifactId": artifact_id,
            "releaseStatus": release_status,
            "activationEligible": activation_eligible,
            "servingEnabled": release_status == "approved" and activation_eligible,
            "promptLoggingAllowed": False,
        }
        (root / "generation-policy.json").write_bytes(json_file_bytes(policy))
        files = []
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "role": "test",
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        manifest = {
            "schemaVersion": 2,
            "artifactKind": "vfdlm-gen1-local-model-package",
            "artifactId": artifact_id,
            "immutable": True,
            "runtime": "pytorch-gen1-v1",
            "compatibilityPath": "compatibility.json",
            "generationPolicyPath": "generation-policy.json",
            "packageSha256": sha256_bytes(canonical_json_bytes(files)),
            "model": {
                "parameterCount": 128,
                "contextLength": 16,
                "quantization": "fp32",
            },
            "releaseState": {
                "releaseStatus": release_status,
                "activationEligible": activation_eligible,
            },
            "files": files,
        }
        signed = sign_document(
            manifest,
            digest_field="manifestSha256",
            private_key_path=self.private_key,
            key_id=self.key_id,
        )
        (root / "artifact-manifest.json").write_bytes(json_file_bytes(signed))
        return verify_artifact_directory(
            root,
            trust_store_path=self.trust_store,
        )

    def _register(self, artifact):
        return register_artifact(
            artifact,
            registry_path=self.registry_path,
            trust_store_path=self.trust_store,
            private_key_path=self.private_key,
            key_id=self.key_id,
        )

    def _activate(self, artifact_id: str, **kwargs):
        return activate_artifact(
            artifact_id,
            registry_path=self.registry_path,
            trust_store_path=self.trust_store,
            private_key_path=self.private_key,
            key_id=self.key_id,
            **kwargs,
        )

    def test_signed_artifact_and_registry_verify(self):
        artifact = self._artifact("vfdlm-g1-edge-v1.0.0-test")
        registry = self._register(artifact)

        verified = verify_registry(self.registry_path, trust_store_path=self.trust_store)

        self.assertEqual(verified["registrySha256"], registry["registrySha256"])
        self.assertEqual(verified["revision"], 1)
        self.assertIsNone(verified["activeArtifactId"])
        self.assertEqual(verified["artifacts"][0]["manifestSha256"], artifact.manifest_sha256)
        self.assertEqual(
            len(list((self.registry_dir / "history").glob("legacy-schema1-*.json"))),
            1,
        )

    def test_missing_private_key_cannot_silently_rotate_trusted_key(self):
        self.private_key.unlink()

        with self.assertRaises(RegistryManagerError) as raised:
            initialize_signing_key(
                self.private_key,
                self.trust_store,
                key_id=self.key_id,
            )

        self.assertEqual(raised.exception.code, "REGISTRY_PRIVATE_KEY_MISSING")
        self.assertFalse(self.private_key.exists())

    def test_tampered_manifest_signature_fails_closed(self):
        artifact = self._artifact("vfdlm-g1-edge-v1.0.1-test")
        manifest_path = artifact.root / "artifact-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["model"]["contextLength"] = 17
        manifest_path.write_bytes(json_file_bytes(manifest))

        with self.assertRaises(RegistryManagerError) as raised:
            verify_artifact_directory(artifact.root, trust_store_path=self.trust_store)

        self.assertEqual(raised.exception.code, "MODEL_ARTIFACT_SIGNATURE_INVALID")

    def test_corrupt_or_partial_artifact_fails_closed(self):
        corrupt = self._artifact("vfdlm-g1-edge-v1.0.2-test")
        (corrupt.root / "model" / "weights.pt").write_bytes(b"corrupt")
        with self.assertRaises(RegistryManagerError) as raised:
            verify_artifact_directory(corrupt.root, trust_store_path=self.trust_store)
        self.assertEqual(raised.exception.code, "MODEL_ARTIFACT_CHECKSUM_MISMATCH")

        partial = self._artifact("vfdlm-g1-edge-v1.0.3-test")
        (partial.root / "model" / "weights.pt").unlink()
        with self.assertRaises(RegistryManagerError) as raised:
            verify_artifact_directory(partial.root, trust_store_path=self.trust_store)
        self.assertEqual(raised.exception.code, "MODEL_ARTIFACT_FILE_MISSING")

    def test_incompatible_artifact_cannot_register_or_activate(self):
        root = self.registry_dir / "artifacts" / "vfdlm-g1-edge-v1.0.4-test"
        with self.assertRaises(RegistryManagerError) as raised:
            self._artifact("vfdlm-g1-edge-v1.0.4-test", compatible=False)

        self.assertTrue(root.exists())
        self.assertEqual(raised.exception.code, "MODEL_ARTIFACT_INCOMPATIBLE")

    def test_experimental_artifact_cannot_activate(self):
        artifact = self._artifact(
            "vfdlm-g1-edge-v0.1.0-bootstrap",
            release_status="experimental",
            activation_eligible=False,
        )
        self._register(artifact)
        before = self.registry_path.read_bytes()

        with self.assertRaises(RegistryManagerError) as raised:
            self._activate(artifact.artifact_id)

        self.assertEqual(raised.exception.code, "MODEL_ARTIFACT_NOT_APPROVED")
        self.assertEqual(self.registry_path.read_bytes(), before)
        self.assertFalse(self.registry_path.with_suffix(".json.lock").exists())

    def test_activation_is_atomic_when_replace_fails(self):
        artifact = self._artifact("vfdlm-g1-edge-v1.1.0-test")
        registry = self._register(artifact)
        before = self.registry_path.read_bytes()

        def fail_replace(_source, _target):
            raise OSError("simulated atomic replace failure")

        with self.assertRaisesRegex(OSError, "simulated atomic replace failure"):
            self._activate(
                artifact.artifact_id,
                expected_revision=registry["revision"],
                replace=fail_replace,
            )

        self.assertEqual(self.registry_path.read_bytes(), before)
        verified = verify_registry(self.registry_path, trust_store_path=self.trust_store)
        self.assertIsNone(verified["activeArtifactId"])

    def test_activation_and_rollback_are_signed_revision_switches(self):
        artifact = self._artifact("vfdlm-g1-edge-v1.2.0-test")
        registered = self._register(artifact)

        active = self._activate(
            artifact.artifact_id,
            expected_revision=registered["revision"],
        )
        self.assertEqual(active["revision"], 2)
        self.assertEqual(active["activeArtifactId"], artifact.artifact_id)
        self.assertIsNone(active["previousActiveArtifactId"])

        rolled_back = rollback_registry(
            registry_path=self.registry_path,
            trust_store_path=self.trust_store,
            private_key_path=self.private_key,
            key_id=self.key_id,
            expected_revision=active["revision"],
        )
        self.assertEqual(rolled_back["revision"], 3)
        self.assertIsNone(rolled_back["activeArtifactId"])
        self.assertEqual(rolled_back["previousActiveArtifactId"], artifact.artifact_id)
        self.assertEqual(rolled_back["lastTransaction"]["type"], "rollback")
        verify_registry(self.registry_path, trust_store_path=self.trust_store)

    def test_optimized_artifact_rollback_restores_previous_artifact(self):
        reference = self._artifact("vfdlm-g1-edge-v1.9.0-reference")
        optimized = self._artifact("vfdlm-g1-edge-v1.9.1-optimized")
        registered_reference = self._register(reference)
        active_reference = self._activate(
            reference.artifact_id,
            expected_revision=registered_reference["revision"],
        )
        registered_optimized = self._register(optimized)
        self.assertEqual(registered_optimized["activeArtifactId"], reference.artifact_id)
        active_optimized = self._activate(
            optimized.artifact_id,
            expected_revision=registered_optimized["revision"],
        )

        rolled_back = rollback_registry(
            registry_path=self.registry_path,
            trust_store_path=self.trust_store,
            private_key_path=self.private_key,
            key_id=self.key_id,
            expected_revision=active_optimized["revision"],
        )

        self.assertEqual(active_reference["activeArtifactId"], reference.artifact_id)
        self.assertEqual(active_optimized["activeArtifactId"], optimized.artifact_id)
        self.assertEqual(
            active_optimized["previousActiveArtifactId"], reference.artifact_id
        )
        self.assertEqual(rolled_back["activeArtifactId"], reference.artifact_id)
        self.assertEqual(
            rolled_back["previousActiveArtifactId"], optimized.artifact_id
        )
        self.assertEqual(rolled_back["lastTransaction"]["type"], "rollback")
        verify_registry(self.registry_path, trust_store_path=self.trust_store)

    def test_revision_conflict_does_not_mutate_registry(self):
        artifact = self._artifact("vfdlm-g1-edge-v1.3.0-test")
        self._register(artifact)
        before = self.registry_path.read_bytes()

        with self.assertRaises(RegistryManagerError) as raised:
            self._activate(artifact.artifact_id, expected_revision=999)

        self.assertEqual(raised.exception.code, "MODEL_REGISTRY_REVISION_CONFLICT")
        self.assertEqual(self.registry_path.read_bytes(), before)

    def test_signed_history_recovers_a_corrupt_live_registry(self):
        artifact = self._artifact("vfdlm-g1-edge-v1.4.0-test")
        registered = self._register(artifact)
        self._activate(artifact.artifact_id)
        self.registry_path.write_text("{broken", encoding="utf-8")

        recovered = recover_latest_registry_history(
            registry_path=self.registry_path,
            trust_store_path=self.trust_store,
        )

        self.assertEqual(recovered["revision"], registered["revision"])
        self.assertIsNone(recovered["activeArtifactId"])
        verify_registry(self.registry_path, trust_store_path=self.trust_store)

    def test_inactive_artifact_retirement_is_signed_and_non_destructive(self):
        artifact = self._artifact("vfdlm-g1-edge-v1.6.0-test")
        registered = self._register(artifact)

        retired = retire_artifact(
            artifact.artifact_id,
            reason="Superseded by a reviewed release candidate.",
            operator_id="operator-test-v1",
            registry_path=self.registry_path,
            trust_store_path=self.trust_store,
            private_key_path=self.private_key,
            key_id=self.key_id,
            expected_revision=registered["revision"],
        )

        self.assertEqual(retired["revision"], registered["revision"] + 1)
        self.assertIsNone(retired["activeArtifactId"])
        self.assertEqual(retired["lastTransaction"]["type"], "retire")
        entry = retired["artifacts"][0]
        self.assertEqual(entry["releaseStatus"], "retired")
        self.assertFalse(entry["activationEligible"])
        self.assertEqual(entry["retirement"]["operatorId"], "operator-test-v1")
        self.assertTrue(artifact.root.exists())
        verify_registry(self.registry_path, trust_store_path=self.trust_store)

    def test_active_and_rollback_artifacts_cannot_be_retired(self):
        artifact = self._artifact("vfdlm-g1-edge-v1.7.0-test")
        registered = self._register(artifact)
        active = self._activate(artifact.artifact_id, expected_revision=registered["revision"])

        with self.assertRaises(RegistryManagerError) as raised:
            retire_artifact(
                artifact.artifact_id,
                reason="Test retirement.",
                operator_id="operator-test-v1",
                registry_path=self.registry_path,
                trust_store_path=self.trust_store,
                private_key_path=self.private_key,
                key_id=self.key_id,
                expected_revision=active["revision"],
            )
        self.assertEqual(raised.exception.code, "MODEL_ARTIFACT_RETIRE_ACTIVE_DENIED")

        rolled_back = rollback_registry(
            registry_path=self.registry_path,
            trust_store_path=self.trust_store,
            private_key_path=self.private_key,
            key_id=self.key_id,
            expected_revision=active["revision"],
        )
        with self.assertRaises(RegistryManagerError) as raised:
            retire_artifact(
                artifact.artifact_id,
                reason="Test retirement.",
                operator_id="operator-test-v1",
                registry_path=self.registry_path,
                trust_store_path=self.trust_store,
                private_key_path=self.private_key,
                key_id=self.key_id,
                expected_revision=rolled_back["revision"],
            )
        self.assertEqual(raised.exception.code, "MODEL_ARTIFACT_RETIRE_ROLLBACK_DENIED")

    def test_tampered_registry_signature_fails_closed(self):
        artifact = self._artifact("vfdlm-g1-edge-v1.5.0-test")
        self._register(artifact)
        registry = json.loads(self.registry_path.read_text(encoding="utf-8"))
        registry["reason"] = "tampered"
        self.registry_path.write_bytes(json_file_bytes(registry))

        with self.assertRaises(RegistryManagerError) as raised:
            verify_registry(self.registry_path, trust_store_path=self.trust_store)

        self.assertEqual(raised.exception.code, "MODEL_REGISTRY_SIGNATURE_INVALID")


class TestCheckedInVfai016Package(unittest.TestCase):
    def test_checked_in_registry_catalogs_only_inactive_experimental_artifacts(self):
        project_root = Path(__file__).resolve().parents[1]
        registry_path = project_root / "model" / "registry" / "active_model.json"
        trust_store = project_root / "model" / "registry" / "trust" / "trusted-keys.json"

        registry = verify_registry(registry_path, trust_store_path=trust_store)

        self.assertIsNone(registry["activeArtifactId"])
        self.assertEqual(registry["state"], "no-approved-artifact")
        self.assertEqual(len(registry["artifacts"]), 3)
        entries = {item["artifactId"]: item for item in registry["artifacts"]}
        self.assertEqual(
            set(entries),
            {
                "vfdlm-g1-edge-v0.1.0-bootstrap",
                "vfdlm-g1-edge-v0.1.1-runtime",
                "vfdlm-g1-edge-v0.1.2-optimized",
            },
        )
        self.assertTrue(
            all(item["releaseStatus"] == "experimental" for item in entries.values())
        )
        self.assertTrue(
            all(item["activationEligible"] is False for item in entries.values())
        )

    def test_package_report_binds_signed_registry_and_artifact(self):
        project_root = Path(__file__).resolve().parents[1]
        report_path = (
            project_root
            / "model"
            / "registry"
            / "reports"
            / "vfdlm-g1-edge-v0.1.0-bootstrap-package.json"
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        unsigned = dict(report)
        expected = unsigned.pop("reportSha256")

        self.assertEqual(sha256_bytes(canonical_json_bytes(unsigned)), expected)
        self.assertTrue(report["verification"]["signatureVerified"])
        self.assertTrue(report["verification"]["exactFileSetVerified"])
        self.assertFalse(report["verification"]["activationEligible"])
        self.assertFalse(report["verification"]["active"])
        self.assertIsNone(report["registry"]["activeArtifactId"])
        manifest_path = project_root / report["manifestMirrorPath"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_sha256 = verify_signed_document(
            manifest,
            digest_field="manifestSha256",
            trust_store_path=(
                project_root / "model" / "registry" / "trust" / "trusted-keys.json"
            ),
            invalid_code="MODEL_ARTIFACT_SIGNATURE_INVALID",
        )
        self.assertEqual(manifest_sha256, report["manifestSha256"])
        self.assertEqual(
            sha256_file(manifest_path),
            report["manifestFileSha256"],
        )


if __name__ == "__main__":
    unittest.main()
