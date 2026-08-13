import unittest
import numpy as np
import os
import sys

ai_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ai_dir not in sys.path:
    sys.path.insert(0, ai_dir)

from model.model_1b import (
    VoltForge1BConfig,
    VoltForge1BTransformer,
    KVCache1B,
    rms_norm,
    silu,
    precompute_rope_freqs,
    apply_rope,
)
from model.calculate_1b_params import ParameterAuditor, Model1BConfig


class TestVoltForge1BArchitecture(unittest.TestCase):
    def test_config_initialization(self):
        cfg = VoltForge1BConfig()
        self.assertEqual(cfg.vocab_size, 32768)
        self.assertEqual(cfg.d_model, 2048)
        self.assertEqual(cfg.n_layers, 20)
        self.assertEqual(cfg.n_heads, 16)
        self.assertEqual(cfg.n_kv_heads, 4)
        self.assertEqual(cfg.d_ff, 5632)
        self.assertEqual(cfg.head_dim, 128)

    def test_parameter_auditor_count(self):
        auditor = ParameterAuditor(Model1BConfig())
        res = auditor.calculate_layer_parameters()
        total_params = res["total_params"]
        # Must be between 1.0B and 1.1B parameters
        self.assertGreaterEqual(total_params, 1_000_000_000, f"Expected >1B params, got {total_params:,}")
        self.assertEqual(total_params, 1_036_077_056)

    def test_rms_norm(self):
        x = np.random.randn(2, 4, 64).astype(np.float32)
        w = np.ones((64,), dtype=np.float32)
        normed = rms_norm(x, w)
        self.assertEqual(normed.shape, x.shape)
        # RMS of normed should be approx 1
        rms_val = np.sqrt(np.mean(normed ** 2, axis=-1))
        np.testing.assert_allclose(rms_val, 1.0, atol=1e-2)

    def test_silu_activation(self):
        x = np.array([-2.0, 0.0, 2.0], dtype=np.float32)
        out = silu(x)
        self.assertEqual(out.shape, x.shape)
        self.assertAlmostEqual(float(out[1]), 0.0, places=4)
        self.assertGreater(float(out[2]), 1.7)

    def test_rope_embedding(self):
        cos, sin = precompute_rope_freqs(head_dim=32, max_seq_len=64)
        self.assertEqual(cos.shape, (64, 16))
        self.assertEqual(sin.shape, (64, 16))

        q = np.random.randn(1, 16, 4, 32).astype(np.float32)
        q_rotated = apply_rope(q, cos, sin)
        self.assertEqual(q_rotated.shape, q.shape)

    def test_kv_cache(self):
        kv = KVCache1B(n_layers=2, n_kv_heads=2, head_dim=16)
        k1 = np.random.randn(1, 4, 2, 16).astype(np.float32)
        v1 = np.random.randn(1, 4, 2, 16).astype(np.float32)

        k_out, v_out = kv.update(0, k1, v1)
        self.assertEqual(k_out.shape, (1, 4, 2, 16))

        k2 = np.random.randn(1, 1, 2, 16).astype(np.float32)
        v2 = np.random.randn(1, 1, 2, 16).astype(np.float32)
        k_out2, v_out2 = kv.update(0, k2, v2)
        self.assertEqual(k_out2.shape, (1, 5, 2, 16))

    def test_mini_1b_transformer_forward(self):
        # Test 1B forward logic on scaled down layer config
        mini_cfg = VoltForge1BConfig(
            vocab_size=256,
            context_length=64,
            d_model=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            head_dim=16,
            d_ff=128
        )
        model = VoltForge1BTransformer(mini_cfg)
        dummy_in = np.random.randint(0, 256, size=(1, 8), dtype=np.int32)
        dummy_tgt = np.random.randint(0, 256, size=(1, 8), dtype=np.int32)

        logits, loss = model.forward(dummy_in, targets=dummy_tgt)
        self.assertEqual(logits.shape, (1, 8, 256))
        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0.0)


if __name__ == "__main__":
    unittest.main()
