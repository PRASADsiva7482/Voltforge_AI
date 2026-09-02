import json
from pathlib import Path
import socket
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from model.registry_manager import (
    canonical_json_bytes,
    initialize_signing_key,
    json_file_bytes,
    sha256_bytes,
    sign_document,
)
from model.runtime_service import ModelRuntimeService, checkpoint_runtime_security_block


class _FakeRuntime:
    instances = []

    def __init__(self, *, trust_store_path, requested_device):
        self.trust_store_path = trust_store_path
        self.requested_device = requested_device
        self.loaded = []
        self.unloaded = False
        self.__class__.instances.append(self)

    def load(self, path):
        self.loaded.append(Path(path))

    def unload(self):
        self.unloaded = True

    def health(self):
        return {
            "productName": "VoltForge AI",
            "familyName": "VoltForge Domain Language Model",
            "familySlug": "vfdlm",
            "artifactId": "vfdlm-g1-edge-v1.0.0-test",
            "generation": 1,
            "deploymentProfile": "edge",
            "semanticVersion": "1.0.0-test",
            "ready": True,
            "runtimeOperational": True,
            "state": "ready",
            "code": "MODEL_RUNTIME_READY",
            "message": "ready",
            "device": "cpu",
            "generationNetworkAccess": False,
        }


class TestModelRuntimeService(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.registry_path = self.root / "registry" / "active_model.json"
        self.trust_store = self.root / "registry" / "trust" / "trusted-keys.json"
        self.private_key = self.root / "private.pem"
        self.key_id = "vf-runtime-service-test-v1"
        initialize_signing_key(
            self.private_key,
            self.trust_store,
            key_id=self.key_id,
        )
        _FakeRuntime.instances.clear()

    def tearDown(self):
        self.temporary.cleanup()

    def _registry(self, *, active: bool) -> None:
        artifact_id = "vfdlm-g1-edge-v1.0.0-test"
        artifacts = [
            {
                "artifactId": artifact_id,
                "releaseStatus": "approved",
                "activationEligible": True,
                "runtime": "pytorch-gen1-v1",
                "root": f"artifacts/{artifact_id}",
                "manifestPath": "artifact-manifest.json",
                "manifestSha256": "1" * 64,
                "manifestFileSha256": "2" * 64,
                "parameterCount": 100,
                "contextLength": 16,
                "quantization": "fp32",
                "packageBytes": 1000,
            }
        ]
        document = {
            "schemaVersion": 2,
            "registryId": "vfdlm-local-registry-v1",
            "revision": 1,
            "identityContract": {
                "productName": "VoltForge AI",
                "familyName": "VoltForge Domain Language Model",
                "familySlug": "vfdlm",
            },
            "activeArtifactId": artifact_id if active else None,
            "previousActiveArtifactId": None,
            "releaseStatus": "approved" if active else "none",
            "state": "active" if active else "no-approved-artifact",
            "reason": "test registry",
            "artifacts": artifacts,
            "lastTransaction": None,
        }
        signed = sign_document(
            document,
            digest_field="registrySha256",
            private_key_path=self.private_key,
            key_id=self.key_id,
        )
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_bytes(json_file_bytes(signed))

    def test_inactive_registry_stays_unavailable_without_importing_torch_runtime(self):
        self._registry(active=False)
        service = ModelRuntimeService(
            registry_path=self.registry_path,
            trust_store_path=self.trust_store,
        )

        with mock.patch(
            "model.runtime_service.importlib.import_module",
            side_effect=AssertionError("native runtime must not import"),
        ) as native_import, mock.patch.object(
            socket.socket,
            "connect",
            side_effect=AssertionError("startup attempted network access"),
        ):
            health = service.start()

        native_import.assert_not_called()
        self.assertFalse(health["ready"])
        self.assertEqual(health["runtimeState"], "unavailable")
        self.assertEqual(health["code"], "NO_APPROVED_MODEL_ARTIFACT")
        self.assertFalse(health["generationNetworkAccess"])
        self.assertFalse(health["trainingAvailableAtRuntime"])
        self.assertFalse(health["downloadAvailableAtRuntime"])

    def test_active_registry_lazily_loads_exactly_one_runtime_instance(self):
        self._registry(active=True)
        service = ModelRuntimeService(
            registry_path=self.registry_path,
            trust_store_path=self.trust_store,
            requested_device="cpu",
        )
        fake_module = SimpleNamespace(LocalGen1Runtime=_FakeRuntime)

        with mock.patch(
            "model.runtime_service.checkpoint_runtime_security_block", return_value=None
        ), mock.patch(
            "model.runtime_service.importlib.import_module", return_value=fake_module
        ) as native_import:
            first = service.start()
            second = service.start()

        native_import.assert_called_once_with("model.gen1.runtime")
        self.assertEqual(len(_FakeRuntime.instances), 1)
        self.assertEqual(len(_FakeRuntime.instances[0].loaded), 1)
        self.assertEqual(
            _FakeRuntime.instances[0].loaded[0].name,
            "vfdlm-g1-edge-v1.0.0-test",
        )
        self.assertTrue(first["ready"])
        self.assertTrue(second["ready"])
        self.assertEqual(first["runtimeState"], "ready")
        self.assertEqual(first["activeArtifactId"], "vfdlm-g1-edge-v1.0.0-test")
        self.assertEqual(service.runtime_for_generation(), _FakeRuntime.instances[0])

    def test_stop_unloads_runtime_and_reports_unavailable(self):
        self._registry(active=True)
        service = ModelRuntimeService(
            registry_path=self.registry_path,
            trust_store_path=self.trust_store,
        )
        with mock.patch(
            "model.runtime_service.checkpoint_runtime_security_block", return_value=None
        ), mock.patch(
            "model.runtime_service.importlib.import_module",
            return_value=SimpleNamespace(LocalGen1Runtime=_FakeRuntime),
        ):
            service.start()
        runtime = _FakeRuntime.instances[0]

        service.stop()
        health = service.health()

        self.assertTrue(runtime.unloaded)
        self.assertFalse(health["ready"])
        self.assertEqual(health["runtimeState"], "unavailable")
        self.assertEqual(health["code"], "MODEL_RUNTIME_STOPPED")
        with self.assertRaisesRegex(RuntimeError, "MODEL_RUNTIME_STOPPED"):
            service.runtime_for_generation()

    def test_concurrent_start_still_constructs_one_process_instance(self):
        self._registry(active=True)
        service = ModelRuntimeService(
            registry_path=self.registry_path,
            trust_store_path=self.trust_store,
        )
        entered = threading.Event()
        release = threading.Event()
        original_load = _FakeRuntime.load
        results = []

        def delayed_load(instance, path):
            entered.set()
            if not release.wait(timeout=5):
                raise AssertionError("concurrent-start test did not release load")
            original_load(instance, path)

        def start():
            results.append(service.start())

        with mock.patch(
            "model.runtime_service.checkpoint_runtime_security_block",
            return_value=None,
        ), mock.patch(
            "model.runtime_service.importlib.import_module",
            return_value=SimpleNamespace(LocalGen1Runtime=_FakeRuntime),
        ) as native_import, mock.patch.object(_FakeRuntime, "load", delayed_load):
            first = threading.Thread(target=start)
            second = threading.Thread(target=start)
            first.start()
            self.assertTrue(entered.wait(timeout=5))
            second.start()
            release.set()
            first.join(timeout=10)
            second.join(timeout=10)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(len(results), 2)
        self.assertTrue(all(item["ready"] for item in results))
        self.assertEqual(len(_FakeRuntime.instances), 1)
        native_import.assert_called_once_with("model.gen1.runtime")

    def test_vulnerable_torch_blocks_active_checkpoint_before_native_import(self):
        self._registry(active=True)
        service = ModelRuntimeService(
            registry_path=self.registry_path,
            trust_store_path=self.trust_store,
        )

        with mock.patch(
            "model.runtime_service.package_version", return_value="2.8.0+cpu"
        ), mock.patch(
            "model.runtime_service.importlib.import_module",
            side_effect=AssertionError("blocked runtime must not import"),
        ) as native_import:
            health = service.start()

        native_import.assert_not_called()
        self.assertFalse(health["ready"])
        self.assertEqual(health["runtimeState"], "failed")
        self.assertEqual(health["code"], "MODEL_RUNTIME_CHECKPOINT_SECURITY_BLOCKED")
        with self.assertRaisesRegex(
            RuntimeError, "MODEL_RUNTIME_CHECKPOINT_SECURITY_BLOCKED"
        ):
            service.runtime_for_generation()

    def test_patched_torch_clears_checkpoint_security_advisory_gate(self):
        with mock.patch(
            "model.runtime_service.package_version", return_value="2.10.0+cpu"
        ):
            self.assertIsNone(checkpoint_runtime_security_block())

    def test_invalid_registry_reports_failed_without_native_import(self):
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text("{broken", encoding="utf-8")
        service = ModelRuntimeService(
            registry_path=self.registry_path,
            trust_store_path=self.trust_store,
        )

        with mock.patch("model.runtime_service.importlib.import_module") as native_import:
            health = service.start()

        native_import.assert_not_called()
        self.assertFalse(health["ready"])
        self.assertEqual(health["runtimeState"], "failed")
        self.assertEqual(health["code"], "MODEL_REGISTRY_INVALID")

    def test_supervisor_source_contains_no_eager_native_or_training_import(self):
        source = (
            Path(__file__).resolve().parents[1] / "model" / "runtime_service.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("from model.gen1", source)
        self.assertNotIn("import model.gen1", source)
        self.assertNotIn("gen1_training", source)
        self.assertNotIn("requests", source)
        self.assertNotIn("urllib", source)

    def test_checked_in_runtime_evidence_is_complete_and_privacy_safe(self):
        project_root = Path(__file__).resolve().parents[1]
        smoke_path = (
            project_root / "evaluation" / "reports" / "gen1-runtime-smoke-v1.json"
        )
        package_path = (
            project_root
            / "model"
            / "registry"
            / "reports"
            / "vfdlm-g1-edge-v0.1.1-runtime-package.json"
        )
        smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
        package = json.loads(package_path.read_text(encoding="utf-8"))
        unsigned_smoke = dict(smoke)
        smoke_hash = unsigned_smoke.pop("reportSha256")
        unsigned_package = dict(package)
        package_hash = unsigned_package.pop("reportSha256")

        self.assertEqual(sha256_bytes(canonical_json_bytes(unsigned_smoke)), smoke_hash)
        self.assertEqual(
            sha256_bytes(canonical_json_bytes(unsigned_package)), package_hash
        )
        self.assertTrue(all(smoke["checks"].values()))
        self.assertFalse(smoke["serviceActive"])
        self.assertFalse(smoke["activationEligible"])
        self.assertFalse(smoke["prompt"]["stored"])
        self.assertFalse(smoke["greedy"]["outputStored"])
        self.assertFalse(smoke["sampling"]["outputStored"])
        self.assertNotIn("text", smoke["greedy"])
        self.assertNotIn("text", smoke["sampling"])
        self.assertEqual(package["artifactId"], smoke["artifactId"])
        self.assertEqual(package["manifestSha256"], smoke["artifactManifestSha256"])
        self.assertIsNone(package["registry"]["activeArtifactId"])


if __name__ == "__main__":
    unittest.main()
