import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from model.artifact_registry import (
    ArtifactRegistryError,
    get_artifact_health,
    resolve_active_artifact,
)
from model.infer import VoltForgeInferenceEngine
from model.model import TransformerConfig, VoltForgeTransformer
from model.tokenizer import VoltForgeTokenizer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class TestArtifactRegistry(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "model"
        self.registry_dir = self.root / "registry"
        self.registry_dir.mkdir(parents=True)
        self.registry_path = self.registry_dir / "active_model.json"

    def tearDown(self):
        self.temporary.cleanup()

    def _write_empty_registry(self) -> None:
        self.registry_path.write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "activeArtifactId": None,
                    "reason": "No test artifact is approved.",
                    "artifacts": [],
                }
            ),
            encoding="utf-8",
        )

    def _write_valid_artifact(self) -> Path:
        artifact_id = "vfdlm-g1-edge-v0.0.1-test"
        artifact_root = self.root / "approved" / artifact_id
        artifact_root.mkdir(parents=True)

        tokenizer = VoltForgeTokenizer()
        tokenizer.save(
            str(artifact_root),
            release_status="approved",
            lineage={
                "sourceIds": ["vf-src-test-project-v1"],
                "shardIds": ["vf-shard-test-training-v1"],
            },
        )
        config = TransformerConfig(
            vocab_size=len(tokenizer.vocab),
            context_length=16,
            d_model=8,
            n_heads=2,
            n_layers=1,
            d_ff=16,
        )
        (artifact_root / "config.json").write_text(
            json.dumps(config.to_dict()), encoding="utf-8"
        )
        (artifact_root / "metadata.json").write_text(
            json.dumps(
                {
                    "artifactId": artifact_id,
                    "familySlug": "vfdlm",
                    "releaseStatus": "approved",
                }
            ),
            encoding="utf-8",
        )
        model = VoltForgeTransformer(config)
        model.save_weights(str(artifact_root / "model_weights.npz"))

        names = {
            "config": "config.json",
            "metadata": "metadata.json",
            "weights": "model_weights.npz",
            "vocab": "vocab.json",
            "merges": "merges.json",
            "tokenizerConfig": "tokenizer_config.json",
            "tokenizerManifest": "tokenizer_manifest.json",
        }
        files = {
            key: {"path": name, "sha256": _sha256(artifact_root / name)}
            for key, name in names.items()
        }
        self.registry_path.write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "activeArtifactId": artifact_id,
                    "artifacts": [
                        {
                            "artifactId": artifact_id,
                            "releaseStatus": "approved",
                            "runtime": "numpy-transformer-v1",
                            "root": f"../approved/{artifact_id}",
                            "parameterCount": model.count_parameters(),
                            "quantization": "fp32",
                            "checksum": files["weights"]["sha256"],
                            "dataLineage": {
                                "sourceIds": ["vf-src-test-project-v1"],
                                "shardIds": ["vf-shard-test-training-v1"],
                            },
                            "files": files,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return artifact_root

    def _refresh_weight_hash(self, artifact_root: Path) -> None:
        registry = json.loads(self.registry_path.read_text(encoding="utf-8"))
        registry["artifacts"][0]["files"]["weights"]["sha256"] = _sha256(
            artifact_root / "model_weights.npz"
        )
        registry["artifacts"][0]["checksum"] = registry["artifacts"][0]["files"][
            "weights"
        ]["sha256"]
        self.registry_path.write_text(json.dumps(registry), encoding="utf-8")

    def test_unmanifested_legacy_file_is_not_activated(self):
        legacy = self.root / "artifacts"
        legacy.mkdir()
        np.savez_compressed(legacy / "model_weights.npz", random=np.ones((2, 2)))
        self._write_empty_registry()

        health = get_artifact_health(self.registry_path)

        self.assertFalse(health["ready"])
        self.assertEqual(health["code"], "NO_APPROVED_MODEL_ARTIFACT")

    def test_valid_approved_artifact_resolves_and_loads_without_partial_state(self):
        self._write_valid_artifact()

        artifact = resolve_active_artifact(self.registry_path)
        engine = VoltForgeInferenceEngine(str(self.registry_path))

        self.assertEqual(artifact.artifact_id, "vfdlm-g1-edge-v0.0.1-test")
        self.assertTrue(engine.is_loaded)
        self.assertEqual(engine.model.count_parameters(), artifact.parameter_count)
        health = engine.health()
        self.assertEqual(health["code"], "MODEL_ARTIFACT_READY")
        self.assertEqual(health["familySlug"], "vfdlm")
        self.assertEqual(health["generation"], 1)
        self.assertEqual(health["deploymentProfile"], "edge")
        self.assertEqual(health["semanticVersion"], "0.0.1-test")
        self.assertEqual(health["parameterCount"], artifact.parameter_count)
        self.assertEqual(health["contextLength"], 16)
        self.assertEqual(health["quantization"], "fp32")
        self.assertEqual(health["checksum"], artifact.checksum)
        self.assertEqual(health["releaseStatus"], "approved")
        self.assertEqual(health["dataSourceCount"], 1)
        self.assertEqual(health["trainingShardCount"], 1)

    def test_approved_artifact_without_data_lineage_fails_closed(self):
        self._write_valid_artifact()
        registry = json.loads(self.registry_path.read_text(encoding="utf-8"))
        del registry["artifacts"][0]["dataLineage"]
        self.registry_path.write_text(json.dumps(registry), encoding="utf-8")

        with self.assertRaises(ArtifactRegistryError) as raised:
            resolve_active_artifact(self.registry_path)

        self.assertEqual(raised.exception.code, "MODEL_DATA_LINEAGE_MISSING")

    def test_approved_artifact_without_tokenizer_manifest_dependency_fails_closed(self):
        self._write_valid_artifact()
        registry = json.loads(self.registry_path.read_text(encoding="utf-8"))
        del registry["artifacts"][0]["files"]["tokenizerManifest"]
        self.registry_path.write_text(json.dumps(registry), encoding="utf-8")

        with self.assertRaises(ArtifactRegistryError) as raised:
            resolve_active_artifact(self.registry_path)

        self.assertEqual(raised.exception.code, "MODEL_ARTIFACT_FILE_UNDECLARED")

    def test_checksum_mismatch_fails_with_precise_reason(self):
        artifact_root = self._write_valid_artifact()
        (artifact_root / "config.json").write_text("{}", encoding="utf-8")

        with self.assertRaises(ArtifactRegistryError) as raised:
            resolve_active_artifact(self.registry_path)

        self.assertEqual(raised.exception.code, "MODEL_ARTIFACT_CHECKSUM_MISMATCH")

    def test_tensor_shape_mismatch_fails_before_runtime_loading(self):
        artifact_root = self._write_valid_artifact()
        weights_path = artifact_root / "model_weights.npz"
        with np.load(weights_path, allow_pickle=False) as archive:
            tensors = {name: np.array(archive[name], copy=True) for name in archive.files}
        tensors["wte"] = tensors["wte"][:-1]
        np.savez_compressed(weights_path, **tensors)
        self._refresh_weight_hash(artifact_root)

        with self.assertRaises(ArtifactRegistryError) as raised:
            resolve_active_artifact(self.registry_path)

        self.assertEqual(raised.exception.code, "WEIGHT_SHAPE_MISMATCH")
        self.assertIn("wte", raised.exception.message)

    def test_missing_registry_does_not_create_random_runtime_model(self):
        engine = VoltForgeInferenceEngine(str(self.registry_path))

        self.assertFalse(engine.is_loaded)
        self.assertIsNone(engine.model)
        self.assertEqual(engine.compute_confidence("test"), 0.0)
        self.assertEqual(engine.status_code, "MODEL_REGISTRY_NOT_FOUND")
        with self.assertRaisesRegex(RuntimeError, "MODEL_REGISTRY_NOT_FOUND"):
            engine.generate_neural_text("test")

    def test_numpy_model_rejects_partial_checkpoint(self):
        config = TransformerConfig(
            vocab_size=32,
            context_length=8,
            d_model=8,
            n_heads=2,
            n_layers=1,
            d_ff=16,
        )
        model = VoltForgeTransformer(config, initialize_weights=False)
        checkpoint = self.root / "partial.npz"
        np.savez_compressed(checkpoint, wte=np.zeros((32, 8), dtype=np.float32))

        with self.assertRaisesRegex(ValueError, "Checkpoint tensor mismatch"):
            model.load_weights(str(checkpoint))
        self.assertEqual(model.weights, {})


class TestCheckedInArtifactInventory(unittest.TestCase):
    def test_inventory_covers_every_retained_artifact_and_config(self):
        model_root = Path(__file__).resolve().parents[1] / "model"
        inventory = json.loads(
            (model_root / "artifact_inventory.json").read_text(encoding="utf-8")
        )
        actual = {
            path.relative_to(model_root.parent).as_posix()
            for directory in (model_root / "artifacts", model_root / "configs")
            for path in directory.rglob("*")
            if path.is_file()
        }
        actual.add("model/legacy/retired_scale_experiment/architecture_config.yaml")
        declared = {item["path"] for item in inventory["artifacts"]}

        self.assertEqual(declared, actual)
        self.assertEqual(inventory["summary"]["productionEligibleCount"], 0)
        self.assertEqual(
            inventory["productionPolicy"]["reasonCode"], "NO_APPROVED_MODEL_ARTIFACT"
        )
        by_path = {item["path"]: item for item in inventory["artifacts"]}
        self.assertEqual(
            by_path["model/artifacts/model_weights.npz"]["classification"],
            "random-untrained",
        )
        self.assertTrue(
            all(
                item["classification"] == "orphaned-incompatible-checkpoint"
                for path, item in by_path.items()
                if Path(path).name.startswith("checkpoint_")
            )
        )


if __name__ == "__main__":
    unittest.main()
