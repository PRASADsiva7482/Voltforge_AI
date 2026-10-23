import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from config import Settings
from api.security import require_service_token
from fastapi import HTTPException


class HttpLimitTests(unittest.TestCase):
    def test_service_does_not_accept_chats_after_model_startup_failure(self):
        from main import app
        from types import SimpleNamespace
        engine = SimpleNamespace(llm=None, initialize=lambda:None, close=lambda:None)
        with patch('main.get_llm_engine',return_value=engine), patch('main.config.get_settings',return_value=Settings(api_token='test-token')):
            with self.assertRaisesRegex(RuntimeError,'model failed to load'):
                with TestClient(app): self.fail('Startup should have failed')
    def test_import_main_never_reexecutes_or_loads_model(self):
        with patch('subprocess.call', side_effect=AssertionError('Unexpected process')):
            import main
            self.assertIsNotNone(main.app)
    def test_body_limit_precedes_json_parsing(self):
        from main import app
        client = TestClient(app)
        response = client.post('/voltForge-ai/api/v1/model/chat',content=b'x'*2_000_001)
        self.assertEqual(response.status_code,413)
    def test_auth_without_token_fails_closed(self):
        with patch('api.security.get_settings',return_value=Settings(api_token='')):
            with self.assertRaises(HTTPException) as caught:
                require_service_token(None,None)
            self.assertEqual(caught.exception.status_code,503)
    def test_invalid_runtime_settings_are_not_silently_ignored(self):
        for data in ({'port':0},{'n_threads':0},{'model_provider':'ollama'},{'n_ctx':2048,'max_tokens':2048}):
            with self.assertRaises(ValueError): Settings(**data)


if __name__ == '__main__':
    unittest.main()
