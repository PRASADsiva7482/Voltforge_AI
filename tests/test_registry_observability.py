import ast
from pathlib import Path
import unittest

from api.routes import chat, health_check
from api.schemas import ChatRequest
from main import log_model_registry_identity


ARTIFACT_ID = "vfdlm-g1-edge-v0.1.0-bootstrap"
RUNTIME_ARTIFACT_ID = "vfdlm-g1-edge-v0.1.1-runtime"
OPTIMIZED_ARTIFACT_ID = "vfdlm-g1-edge-v0.1.2-optimized"


class TestRegistryObservability(unittest.TestCase):
    def test_health_identifies_exact_inactive_artifact_without_claiming_readiness(self):
        health = health_check()["localModel"]

        self.assertFalse(health["ready"])
        self.assertEqual(health["code"], "NO_APPROVED_MODEL_ARTIFACT")
        self.assertIsNone(health["artifactId"])
        self.assertEqual(health["catalogArtifactCount"], 3)
        candidates = {item["artifactId"]: item for item in health["catalogArtifacts"]}
        self.assertEqual(
            set(candidates), {ARTIFACT_ID, RUNTIME_ARTIFACT_ID, OPTIMIZED_ARTIFACT_ID}
        )
        for candidate in candidates.values():
            self.assertEqual(candidate["releaseStatus"], "experimental")
            self.assertFalse(candidate["activationEligible"])
            self.assertRegex(candidate["manifestSha256"], r"^[0-9a-f]{64}$")

    def test_startup_log_identifies_registry_and_artifact(self):
        with self.assertLogs("voltforge-ai.startup", level="INFO") as captured:
            log_model_registry_identity()

        message = "\n".join(captured.output)
        self.assertIn(ARTIFACT_ID, message)
        self.assertIn(RUNTIME_ARTIFACT_ID, message)
        self.assertIn(OPTIMIZED_ARTIFACT_ID, message)
        self.assertIn("activeArtifactId=None", message)
        self.assertIn("registryRevision=3", message)

    def test_chat_prompt_is_not_logged(self):
        sentinel = "VFAI016_PRIVATE_PROMPT_7f2d5f4b"

        with self.assertLogs(level="INFO") as captured:
            chat(ChatRequest(message=sentinel))

        self.assertNotIn(sentinel, "\n".join(captured.output))

    def test_logger_calls_do_not_reference_request_text(self):
        project_root = Path(__file__).resolve().parents[1]
        paths = (
            project_root / "api" / "routes.py",
            project_root / "api" / "database.py",
            project_root / "engine" / "reasoning.py",
            project_root / "web_search_engine.py",
            project_root / "internet_retrieval" / "provider.py",
            project_root / "internet_retrieval" / "service.py",
        )
        blocked = (
            "payload.message",
            "payload.prompt",
            "payload.query",
            "combined_prompt",
            "user_message",
            "ai_response",
            "part_number",
        )
        violations = []
        for path in paths:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                if (
                    not isinstance(node.func.value, ast.Name)
                    or node.func.value.id != "logger"
                    or node.func.attr not in {"debug", "info", "warning", "error", "exception"}
                ):
                    continue
                segment = ast.get_source_segment(source, node) or ""
                if any(value in segment for value in blocked):
                    violations.append(f"{path.name}:{node.lineno}")

        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
