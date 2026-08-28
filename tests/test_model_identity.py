import hashlib
import json
from pathlib import Path
import re
import unittest

from model.artifact_registry import get_artifact_health
from model.identity import (
    ARTIFACT_ID_PATTERN_TEXT,
    FAMILY_NAME,
    FAMILY_SLUG,
    PRODUCT_NAME,
    parse_artifact_id,
)


AI_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = AI_ROOT.parent


class TestModelIdentity(unittest.TestCase):
    def test_canonical_identity_parses_deployment_profile_and_version(self):
        identity = parse_artifact_id("vfdlm-g2-core-v1.4.0-rc.2")

        self.assertEqual(identity.generation, 2)
        self.assertEqual(identity.deployment_profile, "core")
        self.assertEqual(identity.semantic_version, "1.4.0")
        self.assertEqual(identity.display_version, "1.4.0-rc.2")
        self.assertEqual(identity.health_fields()["familySlug"], "vfdlm")

    def test_size_marketing_and_invalid_profiles_are_not_valid_artifact_ids(self):
        invalid = (
            "voltforge-large-model",
            "vfdlm-1b-v1.0.0",
            "vfdlm-g0-edge-v1.0.0",
            "vfdlm-g1-cloud-v1.0.0",
            "vfdlm-g1-core-v1",
        )
        for artifact_id in invalid:
            with self.subTest(artifact_id=artifact_id):
                with self.assertRaises(ValueError):
                    parse_artifact_id(artifact_id)

    def test_unavailable_health_reports_complete_truthful_identity_shape(self):
        health = get_artifact_health()

        self.assertFalse(health["ready"])
        self.assertEqual(health["productName"], PRODUCT_NAME)
        self.assertEqual(health["familyName"], FAMILY_NAME)
        self.assertEqual(health["familySlug"], FAMILY_SLUG)
        self.assertIsNone(health["artifactId"])
        self.assertIsNone(health["parameterCount"])
        self.assertIsNone(health["contextLength"])
        self.assertIsNone(health["quantization"])
        self.assertIsNone(health["checksum"])
        self.assertEqual(health["releaseStatus"], "none")

    def test_identity_contract_is_declared_by_the_registry(self):
        registry = json.loads(
            (AI_ROOT / "model" / "registry" / "active_model.json").read_text(
                encoding="utf-8"
            )
        )
        contract = registry["identityContract"]

        self.assertEqual(contract["productName"], PRODUCT_NAME)
        self.assertEqual(contract["familyName"], FAMILY_NAME)
        self.assertEqual(contract["familySlug"], FAMILY_SLUG)
        self.assertIn("vfdlm-g{generation}", contract["artifactIdPattern"])
        self.assertTrue(ARTIFACT_ID_PATTERN_TEXT.startswith("^vfdlm-g"))


class TestRetiredLabelMigration(unittest.TestCase):
    def setUp(self):
        self.migration = json.loads(
            (AI_ROOT / "model" / "legacy" / "retired_model_labels.json").read_text(
                encoding="utf-8"
            )
        )

    def test_every_retired_path_is_mapped_and_retained(self):
        expected_old_paths = {
            "model/model_1b.py",
            "model/pytorch_1b.py",
            "model/calculate_1b_params.py",
            "model/train_1b.py",
            "model/generate_1b_dataset.py",
            "model/export_1b.py",
            "model/configs/1b_model.yaml",
            "tests/test_model_1b.py",
            "model/artifacts/model_1b_meta.json",
            "model/artifacts/voltforge_1b_config.json",
            "model/artifacts/dataset_1b.jsonl",
        }
        mappings = self.migration["legacyArtifacts"]

        self.assertEqual({item["oldPath"] for item in mappings}, expected_old_paths)
        for item in mappings:
            current_path = AI_ROOT / item["currentPath"]
            self.assertTrue(current_path.is_file(), item["currentPath"])
            if "sha256" in item:
                actual = hashlib.sha256(current_path.read_bytes()).hexdigest()
                self.assertEqual(actual, item["sha256"])

    def test_retired_labels_have_no_active_capability_claim(self):
        replacement = self.migration["replacementIdentity"]
        self.assertIsNone(replacement["activeArtifactId"])
        self.assertEqual(replacement["releaseStatus"], "none")
        self.assertEqual(replacement["familySlug"], "vfdlm")
        self.assertGreaterEqual(len(self.migration["retiredLabels"]), 2)

    def test_active_source_config_ui_and_architecture_doc_have_no_retired_model_label(self):
        active_paths = [
            *(path for path in (AI_ROOT / "model").glob("*.py")),
            *(path for path in (AI_ROOT / "model" / "configs").glob("*")),
            *(path for path in (AI_ROOT / "api").glob("*.py")),
            AI_ROOT / "config.py",
            AI_ROOT / "main.py",
            WORKSPACE_ROOT / "VOLTFORGE_E2E_ARCHITECTURE_AND_FEATURES.md",
            WORKSPACE_ROOT / "custom_ai_from_scratch_prompt.md",
            WORKSPACE_ROOT / "Voltforge_UI" / "src" / "features" / "ai" / "AiChatPanel.tsx",
        ]
        retired_pattern = re.compile(
            r"voltforge[-_ ]?1b|\b1b parameter|1\.1b parameter|1\.036 billion|billion parameter",
            re.IGNORECASE,
        )
        violations = {
            str(path.relative_to(WORKSPACE_ROOT)): retired_pattern.findall(
                path.read_text(encoding="utf-8")
            )
            for path in active_paths
            if path.is_file() and retired_pattern.search(path.read_text(encoding="utf-8"))
        }

        self.assertEqual(violations, {})


if __name__ == "__main__":
    unittest.main()
