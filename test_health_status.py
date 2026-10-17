"""Readiness must not claim a loaded model when only the service is available."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from api.routes import health


class HealthStatusTests(unittest.TestCase):
    def test_fallback_does_not_claim_model_readiness(self):
        with patch('api.routes.get_llm_engine', return_value=SimpleNamespace(llm=None)):
            status = health()
        self.assertTrue(status['serviceReady'])
        self.assertFalse(status['ready'])
        self.assertFalse(status['neuralLoaded'])
        self.assertEqual(status['responseMode'], 'deterministic')
        self.assertEqual(status['code'], 'MODEL_NOT_LOADED')

    def test_loaded_model_reports_neural_readiness(self):
        with patch('api.routes.get_llm_engine', return_value=SimpleNamespace(llm=object())):
            status = health()
        self.assertTrue(status['ready'])
        self.assertTrue(status['neuralLoaded'])
        self.assertEqual(status['responseMode'], 'neural')


if __name__ == '__main__':
    unittest.main()
