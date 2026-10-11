"""
Backend Vulnerability Assessment (VA) Verification Suite for VoltForge AI
Tests:
1. Environment variable loading and overrides
2. Token authentication with constant-time comparison
3. Swagger / OpenAPI documentation suppression in production mode
"""

import os
import unittest
from fastapi import HTTPException
from fastapi.testclient import TestClient

class TestBackendSecurityVA(unittest.TestCase):

    def test_01_env_var_overrides_settings(self):
        os.environ["VOLTFORGE_AI_HOST"] = "0.0.0.0"
        os.environ["VOLTFORGE_AI_PORT"] = "2002"
        os.environ["VOLTFORGE_AI_API_TOKEN"] = "test-secret-token-v1"
        os.environ["VOLTFORGE_AI_ENV"] = "production"

        from config import load_settings
        settings = load_settings()

        self.assertEqual(settings.host, "0.0.0.0")
        self.assertEqual(settings.port, 2002)
        self.assertEqual(settings.api_token, "test-secret-token-v1")
        self.assertEqual(settings.environment, "production")

    def test_02_token_auth_enforcement(self):
        os.environ["VOLTFORGE_AI_API_TOKEN"] = "test-secret-token-v1"
        from config import load_settings
        import config
        config.settings = load_settings()

        from api.security import require_service_token

        # 1. Missing token raises 401
        with self.assertRaises(HTTPException) as ctx:
            require_service_token(authorization=None, x_voltforge_ai_token=None)
        self.assertEqual(ctx.exception.status_code, 401)

        # 2. Invalid token raises 401
        with self.assertRaises(HTTPException) as ctx:
            require_service_token(authorization="Bearer wrong-token", x_voltforge_ai_token=None)
        self.assertEqual(ctx.exception.status_code, 401)

        # 3. Invalid custom header raises 401
        with self.assertRaises(HTTPException) as ctx:
            require_service_token(authorization=None, x_voltforge_ai_token="attacker-token")
        self.assertEqual(ctx.exception.status_code, 401)

        # 4. Valid custom header succeeds without exception
        try:
            require_service_token(authorization=None, x_voltforge_ai_token="test-secret-token-v1")
        except HTTPException:
            self.fail("Valid X-Voltforge-AI-Token unexpectedly raised HTTPException")

        # 5. Valid Bearer token succeeds without exception
        try:
            require_service_token(authorization="Bearer test-secret-token-v1", x_voltforge_ai_token=None)
        except HTTPException:
            self.fail("Valid Bearer token unexpectedly raised HTTPException")

    def test_03_production_docs_suppressed(self):
        os.environ["VOLTFORGE_AI_ENV"] = "production"
        from config import load_settings
        import config
        config.settings = load_settings()

        # Re-import main to trigger FastAPI app creation under production settings
        import importlib
        import main
        importlib.reload(main)

        # In production, docs_url and openapi_url must be disabled (None)
        self.assertIsNone(main.app.docs_url)
        self.assertIsNone(main.app.redoc_url)
        self.assertIsNone(main.app.openapi_url)


if __name__ == "__main__":
    unittest.main()
