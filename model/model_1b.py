"""
VoltForge-1B: Enterprise Foundation LLM Architecture (1.036 Billion Parameters).
Supports Grouped-Query Attention (GQA), SwiGLU FFN, Rotary Position Embeddings (RoPE),
RMSNorm, KV-Caching, and pure high-performance matrix execution.
"""

import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class VoltForge1BConfig:
    model_name: str = "VoltForge-1B"
    vocab_size: int = 32768
    context_length: int = 4096
    d_model: int = 2048
    n_layers: int = 20
    n_heads: int = 16
    n_kv_heads: int = 4
    head_dim: int = 128
    d_ff: int = 5632
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    dropout: float = 0.0
    tie_embeddings: bool = False

    @classmethod
    def from_yaml_or_json(cls, path: str) -> "VoltForge1BConfig":
        if path.endswith(".yaml") or path.endswith(".yml"):
            try:
                import yaml
                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f).get("model", {})
            except ImportError:
                data = {}
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -----------------------------------------------------------------------------
# Activation & Normalization Modules
# -----------------------------------------------------------------------------

def silu(x: np.ndarray) -> np.ndarray:
    """Swish / SiLU activation function: x * sigmoid(x)."""
    return x / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Root Mean Square Layer Normalization."""
    variance = np.mean(x ** 2, axis=-1, keepdims=True)
    return x / np.sqrt(variance + eps) * weight


def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> Tuple[np.ndarray, np.ndarray]:
    """Precomputes Cosine and Sine frequencies for Rotary Positional Embeddings."""
    dim_indices = np.arange(0, head_dim, 2, dtype=np.float32)
    freqs = 1.0 / (theta ** (dim_indices / head_dim))
    t = np.arange(max_seq_len, dtype=np.float32)
    angles = np.outer(t, freqs)  # [seq_len, head_dim / 2]
    cos = np.cos(angles)
    sin = np.sin(angles)
    return cos, sin


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Applies 2D Rotary Position Embeddings to Q or K tensors."""
    # x shape: [batch, seq_len, n_heads, head_dim]
    seq_len = x.shape[1]
    cos_t = cos[:seq_len, np.newaxis, :]  # [seq_len, 1, head_dim/2]
    sin_t = sin[:seq_len, np.newaxis, :]

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    out = np.empty_like(x)
    out[..., 0::2] = x_even * cos_t - x_odd * sin_t
    out[..., 1::2] = x_even * sin_t + x_odd * cos_t
    return out


# -----------------------------------------------------------------------------
# KV Cache for Autoregressive Generation
# -----------------------------------------------------------------------------

class KVCache1B:
    """Key-Value Cache for 1B model fast sequential inference."""

    def __init__(self, n_layers: int, n_kv_heads: int, head_dim: int, max_len: int = 4096):
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.max_len = max_len
        self.k_cache: List[Optional[np.ndarray]] = [None] * n_layers
        self.v_cache: List[Optional[np.ndarray]] = [None] * n_layers
        self.cur_len = 0

    def update(self, layer_idx: int, k_new: np.ndarray, v_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.k_cache[layer_idx] is None:
            self.k_cache[layer_idx] = k_new
            self.v_cache[layer_idx] = v_new
        else:
            self.k_cache[layer_idx] = np.concatenate([self.k_cache[layer_idx], k_new], axis=1)
            self.v_cache[layer_idx] = np.concatenate([self.v_cache[layer_idx], v_new], axis=1)
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def reset(self):
        self.k_cache = [None] * self.n_layers
        self.v_cache = [None] * self.n_layers
        self.cur_len = 0


# -----------------------------------------------------------------------------
# 1B Transformer Engine
# -----------------------------------------------------------------------------

class VoltForge1BTransformer:
    """Full 1.036 Billion Parameter Transformer Engine."""

    def __init__(self, config: Optional[VoltForge1BConfig] = None):
        self.config = config or VoltForge1BConfig()
        self.weights: Dict[str, np.ndarray] = {}
        self.cos_freqs, self.sin_freqs = precompute_rope_freqs(
            self.config.head_dim, self.config.context_length, self.config.rope_theta
        )
        self.num_params = 0
        self.init_weights(random_init=True)

    def init_weights(self, random_init: bool = True) -> None:
        """Initializes all 1.036B parameter tensors with scaled standard normal."""
        c = self.config
        scale = 1.0 / math.sqrt(c.d_model)

        # 1. Token Embeddings: [vocab_size, d_model]
        self.weights["wte"] = np.random.randn(c.vocab_size, c.d_model).astype(np.float32) * scale

        # 2. Transformer Layers (20 blocks)
        for i in range(c.n_layers):
            # Attention Norm
            self.weights[f"l{i}_rms_attn"] = np.ones((c.d_model,), dtype=np.float32)

            # GQA Projections: Q, K, V, O
            self.weights[f"l{i}_Wq"] = np.random.randn(c.d_model, c.n_heads * c.head_dim).astype(np.float32) * scale
            self.weights[f"l{i}_Wk"] = np.random.randn(c.d_model, c.n_kv_heads * c.head_dim).astype(np.float32) * scale
            self.weights[f"l{i}_Wv"] = np.random.randn(c.d_model, c.n_kv_heads * c.head_dim).astype(np.float32) * scale
            self.weights[f"l{i}_Wo"] = np.random.randn(c.n_heads * c.head_dim, c.d_model).astype(np.float32) * scale

            # FFN Norm
            self.weights[f"l{i}_rms_ffn"] = np.ones((c.d_model,), dtype=np.float32)

            # SwiGLU FFN Projections
            self.weights[f"l{i}_Wgate"] = np.random.randn(c.d_model, c.d_ff).astype(np.float32) * scale
            self.weights[f"l{i}_Wup"] = np.random.randn(c.d_model, c.d_ff).astype(np.float32) * scale
            self.weights[f"l{i}_Wdown"] = np.random.randn(c.d_ff, c.d_model).astype(np.float32) * (1.0 / math.sqrt(c.d_ff))

        # 3. Final RMSNorm & LM Head
        self.weights["rms_final"] = np.ones((c.d_model,), dtype=np.float32)
        if not c.tie_embeddings:
            self.weights["lm_head"] = np.random.randn(c.d_model, c.vocab_size).astype(np.float32) * scale

        self.num_params = sum(w.size for w in self.weights.values())

    def forward_layer(
        self,
        x: np.ndarray,
        layer_idx: int,
        kv_cache: Optional[KVCache1B] = None,
    ) -> np.ndarray:
        c = self.config
        B, T, D = x.shape

        # 1. Self-Attention with GQA
        h = rms_norm(x, self.weights[f"l{layer_idx}_rms_attn"], c.norm_eps)
        q = np.matmul(h, self.weights[f"l{layer_idx}_Wq"]).reshape(B, T, c.n_heads, c.head_dim)
        k = np.matmul(h, self.weights[f"l{layer_idx}_Wk"]).reshape(B, T, c.n_kv_heads, c.head_dim)
        v = np.matmul(h, self.weights[f"l{layer_idx}_Wv"]).reshape(B, T, c.n_kv_heads, c.head_dim)

        # Apply RoPE
        q = apply_rope(q, self.cos_freqs, self.sin_freqs)
        k = apply_rope(k, self.cos_freqs, self.sin_freqs)

        # Update KV Cache if present
        if kv_cache:
            k, v = kv_cache.update(layer_idx, k, v)
            T_k = k.shape[1]
        else:
            T_k = T

        # Repeat KV heads for GQA to match Q heads (Group ratio: n_heads // n_kv_heads = 4)
        rep = c.n_heads // c.n_kv_heads
        if rep > 1:
            k = np.repeat(k, rep, axis=2)  # [B, T_k, n_heads, head_dim]
            v = np.repeat(v, rep, axis=2)

        # Scaled Dot-Product Attention: [B, n_heads, T, T_k]
        q_t = np.transpose(q, (0, 2, 1, 3))
        k_t = np.transpose(k, (0, 2, 1, 3))
        v_t = np.transpose(v, (0, 2, 1, 3))

        scores = np.matmul(q_t, np.transpose(k_t, (0, 1, 3, 2))) / math.sqrt(c.head_dim)

        # Causal mask for autoregression
        if T > 1 and T == T_k:
            causal_mask = np.triu(np.full((T, T), -1e9, dtype=np.float32), k=1)
            scores = scores + causal_mask

        # Softmax
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        attn_out = np.matmul(attn_weights, v_t)  # [B, n_heads, T, head_dim]
        attn_out = np.transpose(attn_out, (0, 2, 1, 3)).reshape(B, T, D)
        attn_proj = np.matmul(attn_out, self.weights[f"l{layer_idx}_Wo"])
        x = x + attn_proj  # Residual connection

        # 2. SwiGLU Feed-Forward
        h_ffn = rms_norm(x, self.weights[f"l{layer_idx}_rms_ffn"], c.norm_eps)
        gate = silu(np.matmul(h_ffn, self.weights[f"l{layer_idx}_Wgate"]))
        up = np.matmul(h_ffn, self.weights[f"l{layer_idx}_Wup"])
        ffn_out = np.matmul(gate * up, self.weights[f"l{layer_idx}_Wdown"])
        x = x + ffn_out  # Residual connection

        return x

    def forward(
        self,
        input_ids: np.ndarray,
        targets: Optional[np.ndarray] = None,
        kv_cache: Optional[KVCache1B] = None,
    ) -> Tuple[np.ndarray, Optional[float]]:
        """
        Forward pass over full 20-layer 1B Transformer.
        Returns (logits, loss).
        """
        B, T = input_ids.shape
        x = self.weights["wte"][input_ids]  # [B, T, d_model]

        for i in range(self.config.n_layers):
            x = self.forward_layer(x, i, kv_cache=kv_cache)

        x = rms_norm(x, self.weights["rms_final"], self.config.norm_eps)

        # Compute Logits
        if self.config.tie_embeddings:
            logits = np.matmul(x, self.weights["wte"].T)
        else:
            logits = np.matmul(x, self.weights["lm_head"])

        loss = None
        if targets is not None:
            # Cross-Entropy Loss computation
            flat_logits = logits.reshape(-1, self.config.vocab_size)
            flat_targets = targets.reshape(-1)
            probs = np.exp(flat_logits - np.max(flat_logits, axis=-1, keepdims=True))
            probs /= np.sum(probs, axis=-1, keepdims=True)
            correct_log_probs = -np.log(np.clip(probs[np.arange(len(flat_targets)), flat_targets], 1e-12, 1.0))
            loss = float(np.mean(correct_log_probs))

        return logits, loss

    def generate(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
    ) -> List[int]:
        """Autoregressive generation with sampling."""
        generated = list(prompt_tokens)
        kv = KVCache1B(self.config.n_layers, self.config.n_kv_heads, self.config.head_dim)

        # Initial prompt pass
        input_ids = np.array([prompt_tokens], dtype=np.int32)
        logits, _ = self.forward(input_ids, kv_cache=kv)

        for _ in range(max_new_tokens):
            next_token_logits = logits[0, -1, :] / max(temperature, 1e-5)

            # Top-K filter
            if top_k > 0:
                indices_to_remove = next_token_logits < np.sort(next_token_logits)[-top_k]
                next_token_logits[indices_to_remove] = -1e9

            # Top-P (Nucleus) filter
            sorted_indices = np.argsort(next_token_logits)[::-1]
            sorted_logits = next_token_logits[sorted_indices]
            cumulative_probs = np.cumsum(np.exp(sorted_logits - np.max(sorted_logits)) / np.sum(np.exp(sorted_logits - np.max(sorted_logits))))
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1]
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = -1e9

            # Sample next token
            probs = np.exp(next_token_logits - np.max(next_token_logits))
            probs /= np.sum(probs)
            next_token = int(np.random.choice(len(probs), p=probs))

            generated.append(next_token)
            if next_token == 3:  # [EOS]
                break

            # Forward pass single new token
            next_input = np.array([[next_token]], dtype=np.int32)
            logits, _ = self.forward(next_input, kv_cache=kv)

        return generated
