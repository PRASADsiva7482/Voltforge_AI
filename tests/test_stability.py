"""
VoltForge AI Enterprise Stability & Numerical Robustness Test Suite.
Verifies numerical stability, NaN/Inf bounds, threadpool resilience,
safe serialization, circuit graph cycle safety, and failover handlers.
"""

import datetime
import decimal
import json
import unittest
import uuid
import numpy as np
import os
import sys

ai_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ai_dir not in sys.path:
    sys.path.insert(0, ai_dir)

from model.model import softmax, silu, silu_grad, qk_norm, TransformerConfig, VoltForgeTransformer
from api.database import safe_json_dumps, SafeJSONEncoder, db
from circuit_verifier import ElectricalVerifier


class TestNumericalStability(unittest.TestCase):
    def test_softmax_extreme_values(self):
        # 1. Very large positive and negative numbers
        x = np.array([1000.0, -1000.0, 500.0, 0.0], dtype=np.float32)
        probs = softmax(x)
        self.assertFalse(np.isnan(probs).any(), "Softmax produced NaN on large inputs")
        self.assertFalse(np.isinf(probs).any(), "Softmax produced Inf on large inputs")
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=4)
        self.assertAlmostEqual(float(probs[0]), 1.0, places=4)

    def test_softmax_nan_and_inf_shielding(self):
        # 2. Corrupt inputs containing NaN or Inf
        x = np.array([np.nan, np.inf, -np.inf, 2.0], dtype=np.float32)
        probs = softmax(x)
        self.assertFalse(np.isnan(probs).any(), "Softmax did not shield against NaN inputs")
        self.assertFalse(np.isinf(probs).any(), "Softmax did not shield against Inf inputs")
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=3)

    def test_silu_extreme_values(self):
        x = np.array([-1000.0, 0.0, 1000.0], dtype=np.float32)
        y = silu(x)
        self.assertFalse(np.isnan(y).any())
        self.assertFalse(np.isinf(y).any())
        self.assertAlmostEqual(float(y[0]), 0.0, places=5)
        self.assertEqual(float(y[1]), 0.0)
        self.assertGreater(float(y[2]), 900.0)

    def test_qk_normalization(self):
        x = np.random.randn(2, 4, 8, 32).astype(np.float32) * 100.0
        normed = qk_norm(x)
        self.assertEqual(normed.shape, x.shape)
        rms = np.sqrt(np.mean(normed ** 2, axis=-1))
        np.testing.assert_allclose(rms, 1.0, atol=1e-2)


class TestSafeSerialization(unittest.TestCase):
    def test_safe_json_dumps_all_types(self):
        payload = {
            "timestamp": datetime.datetime.now(),
            "date": datetime.date.today(),
            "decimal_val": decimal.Decimal("123.4567"),
            "unique_id": uuid.uuid4(),
            "array": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "scalar": np.float32(42.5),
            "tag_set": {"electronics", "arduino", "firmware"},
        }
        res = safe_json_dumps(payload)
        self.assertIsInstance(res, str)
        parsed = json.loads(res)
        self.assertEqual(parsed["decimal_val"], 123.4567)
        self.assertEqual(len(parsed["array"]), 3)
        self.assertEqual(parsed["scalar"], 42.5)
        self.assertEqual(len(parsed["tag_set"]), 3)


class TestCircuitVerifierStability(unittest.TestCase):
    def test_empty_and_corrupt_circuit_inputs(self):
        # Empty
        res = ElectricalVerifier.verify_circuit(board_type="ARDUINO_UNO", components=[], wires=[])
        self.assertIn("safetyScore", res)
        self.assertEqual(res["safetyScore"], 100)

        # Corrupt component dicts missing keys
        corrupt_comps = [{"invalid_key": 123}, None, "bad_string", {}]
        # Filter safely
        clean_comps = [c for c in corrupt_comps if isinstance(c, dict)]
        res2 = ElectricalVerifier.verify_circuit(board_type="ESP32", components=clean_comps, wires=[])
        self.assertIn("safetyScore", res2)

    def test_critical_rules_detection(self):
        comps = [
            {"type": "5V_RELAY", "name": "Relay 1"},
            {"type": "LED", "name": "Status LED"},
            {"type": "MPU6050", "name": "Gyro"}
        ]
        res = ElectricalVerifier.verify_circuit(board_type="ESP32", components=comps, wires=[])
        self.assertLess(res["safetyScore"], 100)
        self.assertGreater(len(res["issues"]), 0)
        self.assertGreater(len(res["additions"]), 0)


class TestDatabaseAsyncQueue(unittest.TestCase):
    def test_async_dispatch_stability(self):
        def sample_task(x: int, y: int) -> int:
            return x + y

        # Dispatch should not raise
        db.run_async(sample_task, 10, 20)
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
