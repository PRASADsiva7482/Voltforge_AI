"""
VoltForge Foundation LLM — Proprietary Decoder-Only Transformer (~9.3M Parameters).
Built entirely from scratch using pure NumPy. No external model weights or frameworks.

Architecture (per layer):
    RMSNorm → Multi-Head Causal Self-Attention (with RoPE) → Residual
    RMSNorm → SwiGLU Feed-Forward Network → Residual
    × 6 Layers

Components:
    - RMSNorm (Root Mean Square Normalization, no bias term)
    - Rotary Positional Embeddings (RoPE) on Q and K
    - SwiGLU gated activation in FFN (gate * up projections with SiLU)
    - Full analytical backpropagation through every parameter
    - Top-K / Top-P nucleus sampling for generation
"""

import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# 1. MODEL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

class TransformerConfig:
    """Configuration for VoltForge Foundation Transformer."""

    def __init__(
        self,
        vocab_size: int = 8192,
        context_length: int = 512,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 768,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
    ):
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.dropout = dropout
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
            "rope_theta": self.rope_theta,
            "dropout": self.dropout,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransformerConfig":
        valid = {"vocab_size", "context_length", "d_model", "n_heads",
                 "n_layers", "d_ff", "rope_theta", "dropout"}
        return cls(**{k: v for k, v in data.items() if k in valid})

    @property
    def num_params(self) -> int:
        """Estimate total parameter count."""
        c = self
        emb = c.vocab_size * c.d_model
        per_layer = (
            c.d_model +                          # rms1
            3 * c.d_model * c.d_model +           # Wq, Wk, Wv
            c.d_model * c.d_model +               # Wo
            c.d_model +                          # rms2
            2 * c.d_model * c.d_ff +              # Wgate, Wup
            c.d_ff * c.d_model                    # Wdown
        )
        head = c.d_model + c.d_model * c.vocab_size  # rms_f + lm_head
        return emb + c.n_layers * per_layer + head


# ═══════════════════════════════════════════════════════════════════════
# 2. PRIMITIVE OPERATIONS
# ═══════════════════════════════════════════════════════════════════════

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Ultra-stable softmax with NaN protection, logit clipping, and non-zero division guard."""
    x_safe = np.nan_to_num(x, nan=-1e9, posinf=65.0, neginf=-1e9)
    x_max = np.max(x_safe, axis=axis, keepdims=True)
    e = np.exp(np.clip(x_safe - x_max, -65.0, 0.0))
    s = np.sum(e, axis=axis, keepdims=True)
    return np.where(s > 1e-12, e / s, 1.0 / max(1, x.shape[axis]))


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU / Swish activation: x · σ(x) with exponential numerical clipping."""
    s = 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))
    return x * s


def silu_grad(x: np.ndarray) -> np.ndarray:
    """Derivative of SiLU: σ(x) + x · σ(x) · (1 − σ(x))."""
    s = 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))
    return s + x * s * (1.0 - s)


def qk_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """QK Normalization across head dimension for attention logit stabilization."""
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return x / rms


# Legacy compatibility (used by reasoning_llm.py)
def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * np.power(x, 3))))


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


# ═══════════════════════════════════════════════════════════════════════
# 3. RMSNORM (Root Mean Square Layer Normalization)
# ═══════════════════════════════════════════════════════════════════════

def rms_norm_fwd(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """Forward pass. Returns (output, rms_inv) for backward."""
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    rms_inv = 1.0 / rms
    return (x * rms_inv) * gamma, rms_inv


def rms_norm_bwd(dy: np.ndarray, x: np.ndarray, gamma: np.ndarray,
                 rms_inv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Backward pass. Returns (dx, dgamma)."""
    x_hat = x * rms_inv
    dgamma = np.sum(dy * x_hat, axis=tuple(range(dy.ndim - 1)))
    dx_hat = dy * gamma
    dx = rms_inv * (dx_hat - x_hat * np.mean(dx_hat * x_hat, axis=-1, keepdims=True))
    return dx, dgamma


# ═══════════════════════════════════════════════════════════════════════
# 4. ROTARY POSITIONAL EMBEDDINGS (RoPE)
# ═══════════════════════════════════════════════════════════════════════

def build_rope_cache(head_dim: int, max_len: int, theta: float = 10000.0) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute cos/sin frequency tables. Returns (cos, sin), each (max_len, head_dim//2)."""
    freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    positions = np.arange(max_len, dtype=np.float32)
    angles = np.outer(positions, freqs)  # (max_len, head_dim//2)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def rope_fwd(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Apply RoPE rotation. x shape: (..., T, head_dim)."""
    T = x.shape[-2]
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    c, s = cos[:T], sin[:T]
    return np.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1)


def rope_bwd(dy: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Inverse RoPE (transpose of rotation matrix)."""
    T = dy.shape[-2]
    d2 = dy.shape[-1] // 2
    dy1, dy2 = dy[..., :d2], dy[..., d2:]
    c, s = cos[:T], sin[:T]
    return np.concatenate([dy1 * c + dy2 * s, -dy1 * s + dy2 * c], axis=-1)


# ═══════════════════════════════════════════════════════════════════════
# 5. VOLTFORGE FOUNDATION TRANSFORMER
# ═══════════════════════════════════════════════════════════════════════

class VoltForgeTransformer:
    """
    Proprietary decoder-only causal Transformer built from scratch.
    Default config: 6 layers, 256 d_model, 8 heads, SwiGLU FFN → ~9.3M parameters.
    """

    def __init__(self, config: TransformerConfig):
        self.config = config
        self.weights: Dict[str, np.ndarray] = {}
        self._init_weights()
        self._rope_cos, self._rope_sin = build_rope_cache(
            config.head_dim, config.context_length, config.rope_theta
        )

    # ── Weight Initialization ──

    def _init_weights(self) -> None:
        """Xavier-style initialization with scaled residual projections."""
        np.random.seed(42)
        c = self.config
        std = 0.02
        res_std = std / math.sqrt(2 * c.n_layers)  # Scale for residual path stability

        # Token embeddings (RoPE replaces learned positional embeddings)
        self.weights["wte"] = np.random.randn(c.vocab_size, c.d_model).astype(np.float32) * std

        for i in range(c.n_layers):
            # Attention block
            self.weights[f"l{i}_rms1"] = np.ones(c.d_model, dtype=np.float32)
            self.weights[f"l{i}_Wq"] = np.random.randn(c.d_model, c.d_model).astype(np.float32) * std
            self.weights[f"l{i}_Wk"] = np.random.randn(c.d_model, c.d_model).astype(np.float32) * std
            self.weights[f"l{i}_Wv"] = np.random.randn(c.d_model, c.d_model).astype(np.float32) * std
            self.weights[f"l{i}_Wo"] = np.random.randn(c.d_model, c.d_model).astype(np.float32) * res_std
            # FFN block
            self.weights[f"l{i}_rms2"] = np.ones(c.d_model, dtype=np.float32)
            self.weights[f"l{i}_Wgate"] = np.random.randn(c.d_model, c.d_ff).astype(np.float32) * std
            self.weights[f"l{i}_Wup"] = np.random.randn(c.d_model, c.d_ff).astype(np.float32) * std
            self.weights[f"l{i}_Wdown"] = np.random.randn(c.d_ff, c.d_model).astype(np.float32) * res_std

        # Final RMSNorm + LM Head (not weight-tied for better output quality)
        self.weights["rms_f"] = np.ones(c.d_model, dtype=np.float32)
        self.weights["lm_head"] = np.random.randn(c.d_model, c.vocab_size).astype(np.float32) * std

    # ── Forward Pass (Inference) ──

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        """
        Forward pass for inference.
        input_ids: (T,) or (B, T) int array.
        Returns: logits (T, V) or (B, T, V).
        """
        squeezed = input_ids.ndim == 1
        if squeezed:
            input_ids = input_ids[np.newaxis, :]

        c = self.config
        B, T = input_ids.shape
        T = min(T, c.context_length)
        input_ids = input_ids[:, :T]
        hd = c.head_dim
        scale = 1.0 / math.sqrt(hd)
        mask = np.triu(np.ones((T, T), dtype=np.float32) * -1e9, k=1)

        x = self.weights["wte"][input_ids]  # (B, T, D)

        for i in range(c.n_layers):
            # ── Attention ──
            n1, _ = rms_norm_fwd(x, self.weights[f"l{i}_rms1"])
            q = np.dot(n1, self.weights[f"l{i}_Wq"]).reshape(B, T, c.n_heads, hd).transpose(0, 2, 1, 3)
            k = np.dot(n1, self.weights[f"l{i}_Wk"]).reshape(B, T, c.n_heads, hd).transpose(0, 2, 1, 3)
            v = np.dot(n1, self.weights[f"l{i}_Wv"]).reshape(B, T, c.n_heads, hd).transpose(0, 2, 1, 3)
            q = rope_fwd(q, self._rope_cos, self._rope_sin)
            k = rope_fwd(k, self._rope_cos, self._rope_sin)
            scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale + mask[np.newaxis, np.newaxis]
            attn = softmax(scores, axis=-1)
            out = np.matmul(attn, v).transpose(0, 2, 1, 3).reshape(B, T, c.d_model)
            x = x + np.dot(out, self.weights[f"l{i}_Wo"])

            # ── SwiGLU FFN ──
            n2, _ = rms_norm_fwd(x, self.weights[f"l{i}_rms2"])
            gate = silu(np.dot(n2, self.weights[f"l{i}_Wgate"]))
            up = np.dot(n2, self.weights[f"l{i}_Wup"])
            x = x + np.dot(gate * up, self.weights[f"l{i}_Wdown"])

        nf, _ = rms_norm_fwd(x, self.weights["rms_f"])
        logits = np.dot(nf, self.weights["lm_head"])
        return logits[0] if squeezed else logits

    # ── Training: Forward + Full Backward ──

    def compute_loss_and_grads(
        self, input_ids: np.ndarray, targets: np.ndarray
    ) -> Tuple[float, Dict[str, np.ndarray]]:
        """
        Complete forward and backward pass with analytical gradient computation.
        input_ids: (T,) int32, targets: (T,) int32 (shifted by 1 position).
        Returns: (cross_entropy_loss, gradient_dict).

        Gradient flows through every weight: embeddings, all 6 layers
        (RMSNorm γ, Wq, Wk, Wv, Wo, Wgate, Wup, Wdown), final RMSNorm, LM head.
        """
        c = self.config
        hd = c.head_dim

        if input_ids.ndim == 1:
            input_ids = input_ids[np.newaxis, :]
            targets = targets[np.newaxis, :]
        B, T = input_ids.shape
        scale = 1.0 / math.sqrt(hd)
        mask = np.triu(np.ones((T, T), dtype=np.float32) * -1e9, k=1)
        grads: Dict[str, np.ndarray] = {}

        # ═══════════════════════ FORWARD (with caching) ═══════════════════════

        x = self.weights["wte"][input_ids]  # (B, T, D)
        layer_cache = []

        for i in range(c.n_layers):
            C: Dict[str, Any] = {}
            C["x"] = x.copy()

            # ── Attention Block ──
            n1, ri1 = rms_norm_fwd(x, self.weights[f"l{i}_rms1"])
            C["n1"], C["ri1"] = n1, ri1

            qf = np.dot(n1, self.weights[f"l{i}_Wq"])  # (B, T, D)
            kf = np.dot(n1, self.weights[f"l{i}_Wk"])
            vf = np.dot(n1, self.weights[f"l{i}_Wv"])

            q = qf.reshape(B, T, c.n_heads, hd).transpose(0, 2, 1, 3)  # (B, H, T, hd)
            k = kf.reshape(B, T, c.n_heads, hd).transpose(0, 2, 1, 3)
            v = vf.reshape(B, T, c.n_heads, hd).transpose(0, 2, 1, 3)
            C["v"] = v

            q_r = rope_fwd(q, self._rope_cos, self._rope_sin)
            k_r = rope_fwd(k, self._rope_cos, self._rope_sin)
            C["q_r"], C["k_r"] = q_r, k_r

            scores = np.matmul(q_r, k_r.transpose(0, 1, 3, 2)) * scale + mask[np.newaxis, np.newaxis]
            att = softmax(scores, axis=-1)  # (B, H, T, T)
            C["att"] = att

            ctx = np.matmul(att, v).transpose(0, 2, 1, 3).reshape(B, T, c.d_model)  # (B, T, D)
            C["ctx"] = ctx

            x = C["x"] + np.dot(ctx, self.weights[f"l{i}_Wo"])  # Residual
            C["xa"] = x.copy()  # x after attention

            # ── SwiGLU FFN Block ──
            n2, ri2 = rms_norm_fwd(x, self.weights[f"l{i}_rms2"])
            C["n2"], C["ri2"] = n2, ri2

            gp = np.dot(n2, self.weights[f"l{i}_Wgate"])  # (B, T, d_ff)
            ga = silu(gp)
            up = np.dot(n2, self.weights[f"l{i}_Wup"])    # (B, T, d_ff)
            hid = ga * up                                  # (B, T, d_ff)
            C["gp"], C["ga"], C["up"], C["hid"] = gp, ga, up, hid

            x = C["xa"] + np.dot(hid, self.weights[f"l{i}_Wdown"])  # Residual
            layer_cache.append(C)

        # Final RMSNorm + LM Head
        x_pre = x.copy()
        nf, rif = rms_norm_fwd(x, self.weights["rms_f"])
        logits = np.dot(nf, self.weights["lm_head"])  # (B, T, V)

        # ═══════════════════════ CROSS-ENTROPY LOSS ═══════════════════════

        probs = softmax(logits, axis=-1)
        bi = np.arange(B)[:, None]
        ti = np.arange(T)[None, :]
        tgt_p = probs[bi, ti, targets]
        loss = -np.mean(np.log(np.maximum(tgt_p, 1e-9)))

        # ═══════════════════════ BACKWARD PASS ═══════════════════════

        # ── dLogits ──
        dl = probs.copy()  # (B, T, V)
        dl[bi, ti, targets] -= 1.0
        dl /= float(B * T)

        # ── LM Head ──
        grads["lm_head"] = np.tensordot(nf, dl, axes=([0, 1], [0, 1]))  # (D, V)
        dx = np.dot(dl, self.weights["lm_head"].T)  # (B, T, D)

        # ── Final RMSNorm ──
        dx, grads["rms_f"] = rms_norm_bwd(dx, x_pre, self.weights["rms_f"], rif)

        # ── Layers (reverse order) ──
        for i in reversed(range(c.n_layers)):
            C = layer_cache[i]

            # ── FFN Backward ──
            # Residual: dx flows through both the identity path and the FFN path
            grads[f"l{i}_Wdown"] = np.tensordot(C["hid"], dx, axes=([0, 1], [0, 1]))  # (d_ff, D)
            dh = np.dot(dx, self.weights[f"l{i}_Wdown"].T)  # (B, T, d_ff)

            d_ga = dh * C["up"]      # gradient through gate_act * up
            d_up = dh * C["ga"]
            d_gp = d_ga * silu_grad(C["gp"])  # backward through SiLU

            grads[f"l{i}_Wgate"] = np.tensordot(C["n2"], d_gp, axes=([0, 1], [0, 1]))  # (D, d_ff)
            grads[f"l{i}_Wup"] = np.tensordot(C["n2"], d_up, axes=([0, 1], [0, 1]))    # (D, d_ff)
            dn2 = np.dot(d_gp, self.weights[f"l{i}_Wgate"].T) + np.dot(d_up, self.weights[f"l{i}_Wup"].T)

            dx_rms2, grads[f"l{i}_rms2"] = rms_norm_bwd(dn2, C["xa"], self.weights[f"l{i}_rms2"], C["ri2"])
            dx = dx + dx_rms2  # Combine residual + norm gradient → dx is now d(x_after_attn)

            # ── Attention Backward ──
            grads[f"l{i}_Wo"] = np.tensordot(C["ctx"], dx, axes=([0, 1], [0, 1]))  # (D, D)
            dc = np.dot(dx, self.weights[f"l{i}_Wo"].T)  # (B, T, D)
            dc = dc.reshape(B, T, c.n_heads, hd).transpose(0, 2, 1, 3)  # (B, H, T, hd)

            # context = attn @ v
            d_att = np.matmul(dc, C["v"].transpose(0, 1, 3, 2))    # (B, H, T, T)
            d_v = np.matmul(C["att"].transpose(0, 1, 3, 2), dc)    # (B, H, T, hd)

            # Softmax backward: ds = attn ⊙ (d_att − sum(d_att ⊙ attn))
            d_scores = C["att"] * (d_att - np.sum(d_att * C["att"], axis=-1, keepdims=True))
            d_scores = d_scores * scale  # Through the 1/√d_k scaling

            # scores = q_r @ k_r^T
            d_qr = np.matmul(d_scores, C["k_r"])                         # (B, H, T, hd)
            d_kr = np.matmul(d_scores.transpose(0, 1, 3, 2), C["q_r"])   # (B, H, T, hd)

            # Inverse RoPE
            d_q = rope_bwd(d_qr, self._rope_cos, self._rope_sin)  # (B, H, T, hd)
            d_k = rope_bwd(d_kr, self._rope_cos, self._rope_sin)

            # Reshape back to (B, T, D)
            d_qf = d_q.transpose(0, 2, 1, 3).reshape(B, T, c.d_model)
            d_kf = d_k.transpose(0, 2, 1, 3).reshape(B, T, c.d_model)
            d_vf = d_v.transpose(0, 2, 1, 3).reshape(B, T, c.d_model)

            # Q, K, V projection gradients
            grads[f"l{i}_Wq"] = np.tensordot(C["n1"], d_qf, axes=([0, 1], [0, 1]))  # (D, D)
            grads[f"l{i}_Wk"] = np.tensordot(C["n1"], d_kf, axes=([0, 1], [0, 1]))
            grads[f"l{i}_Wv"] = np.tensordot(C["n1"], d_vf, axes=([0, 1], [0, 1]))

            dn1 = (np.dot(d_qf, self.weights[f"l{i}_Wq"].T) +
                   np.dot(d_kf, self.weights[f"l{i}_Wk"].T) +
                   np.dot(d_vf, self.weights[f"l{i}_Wv"].T))

            dx_rms1, grads[f"l{i}_rms1"] = rms_norm_bwd(dn1, C["x"], self.weights[f"l{i}_rms1"], C["ri1"])
            dx = dx + dx_rms1  # Combine residual + norm gradient → dx is now d(x_input)

        # ── Token Embedding Gradient ──
        grads["wte"] = np.zeros_like(self.weights["wte"])
        np.add.at(grads["wte"], input_ids, dx)

        return float(loss), grads

    # ── Autoregressive Text Generation ──

    def generate(
        self,
        input_ids: List[int],
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        eos_token_id: int = 3,
        repetition_penalty: float = 1.15,
    ) -> List[int]:
        """Autoregressively generate tokens using top-k/top-p nucleus sampling."""
        ids = list(input_ids)
        generated_set = set()

        for _ in range(max_new_tokens):
            ctx = ids[-self.config.context_length:]
            logits = self.forward(np.array(ctx, dtype=np.int32))  # (T, V)
            next_logits = logits[-1].copy()  # (V,)

            # Repetition penalty
            if repetition_penalty != 1.0:
                for tok in generated_set:
                    if next_logits[tok] > 0:
                        next_logits[tok] /= repetition_penalty
                    else:
                        next_logits[tok] *= repetition_penalty

            # Temperature scaling
            next_logits = next_logits / max(temperature, 1e-5)

            # Top-K filtering
            if top_k > 0:
                topk_idx = np.argsort(next_logits)[-top_k:]
                filtered = np.full_like(next_logits, -1e9)
                filtered[topk_idx] = next_logits[topk_idx]
                next_logits = filtered

            probs = softmax(next_logits)

            # Top-P (nucleus) filtering
            if top_p < 1.0:
                sorted_idx = np.argsort(-probs)
                cum = np.cumsum(probs[sorted_idx])
                cutoff = int(np.searchsorted(cum, top_p)) + 1
                keep = sorted_idx[:cutoff]
                nucleus = np.zeros_like(probs)
                nucleus[keep] = probs[keep]
                probs = nucleus / (nucleus.sum() + 1e-9)

            tok = int(np.random.choice(len(probs), p=probs))
            ids.append(tok)
            generated_set.add(tok)

            if tok == eos_token_id:
                break

        return ids

    # ── Persistence ──

    def save_weights(self, filepath: str) -> None:
        """Save all model weights to a compressed .npz file."""
        np.savez_compressed(filepath, **self.weights)

    def load_weights(self, filepath: str) -> None:
        """Load weights from .npz, matching by name and shape."""
        data = np.load(filepath)
        for k in data.files:
            if k in self.weights and data[k].shape == self.weights[k].shape:
                self.weights[k] = data[k].astype(np.float32)

    def count_parameters(self) -> int:
        """Count actual parameters in the weight dictionary."""
        return sum(w.size for w in self.weights.values())


# Legacy alias for backward compatibility with reasoning_llm.py
NumPyTransformer = VoltForgeTransformer
