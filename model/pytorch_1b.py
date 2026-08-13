"""
VoltForge-1B: Production PyTorch Transformer Architecture (1.036 Billion Parameters).
Supports FlashAttention-2 / SDPA, Grouped-Query Attention (GQA), SwiGLU, RoPE, RMSNorm,
torch.compile, and FSDP distributed training.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class PyTorch1BConfig:
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


if TORCH_AVAILABLE:
    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            variance = x.pow(2).mean(-1, keepdim=True)
            return x * torch.rsqrt(variance + self.eps) * self.weight

    def precompute_rope_freqs_torch(dim: int, max_seq_len: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, freqs)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        return cos, sin

    def apply_rope_torch(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: [B, T, n_heads, head_dim]
        T = x.shape[1]
        cos = cos[:T, :].unsqueeze(0).unsqueeze(2)  # [1, T, 1, head_dim/2]
        sin = sin[:T, :].unsqueeze(0).unsqueeze(2)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        return torch.stack([out_even, out_odd], dim=-1).flatten(-2)

    class GroupedQueryAttention(nn.Module):
        def __init__(self, cfg: PyTorch1BConfig):
            super().__init__()
            self.n_heads = cfg.n_heads
            self.n_kv_heads = cfg.n_kv_heads
            self.head_dim = cfg.head_dim
            self.d_model = cfg.d_model
            self.rep = cfg.n_heads // cfg.n_kv_heads

            self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
            self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
            self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
            self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)
            self.dropout = nn.Dropout(cfg.dropout)

        def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
            B, T, _ = x.shape
            q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
            k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
            v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

            q = apply_rope_torch(q, cos, sin)
            k = apply_rope_torch(k, cos, sin)

            # Repeat KV heads for Grouped-Query Attention
            if self.rep > 1:
                k = k.repeat_interleave(self.rep, dim=2)
                v = v.repeat_interleave(self.rep, dim=2)

            q = q.transpose(1, 2)  # [B, n_heads, T, head_dim]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # High-performance PyTorch FlashAttention-2 / SDPA
            out = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=True,
                dropout_p=self.dropout.p if self.training else 0.0
            )
            out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
            return self.o_proj(out)

    class SwiGLUFFN(nn.Module):
        def __init__(self, cfg: PyTorch1BConfig):
            super().__init__()
            self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
            self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
            self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    class TransformerBlock(nn.Module):
        def __init__(self, cfg: PyTorch1BConfig):
            super().__init__()
            self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
            self.attn = GroupedQueryAttention(cfg)
            self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
            self.ffn = SwiGLUFFN(cfg)

        def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
            x = x + self.attn(self.attn_norm(x), cos, sin)
            x = x + self.ffn(self.ffn_norm(x))
            return x

    class VoltForge1BPyTorch(nn.Module):
        """Complete 1.036 Billion Parameter PyTorch LLM."""
        def __init__(self, cfg: Optional[PyTorch1BConfig] = None):
            super().__init__()
            self.cfg = cfg or PyTorch1BConfig()
            self.wte = nn.Embedding(self.cfg.vocab_size, self.cfg.d_model)
            self.blocks = nn.ModuleList([TransformerBlock(self.cfg) for _ in range(self.cfg.n_layers)])
            self.final_norm = RMSNorm(self.cfg.d_model, self.cfg.norm_eps)

            if self.cfg.tie_embeddings:
                self.lm_head = None
            else:
                self.lm_head = nn.Linear(self.cfg.d_model, self.cfg.vocab_size, bias=False)

            # Register RoPE buffers
            cos, sin = precompute_rope_freqs_torch(self.cfg.head_dim, self.cfg.context_length, self.cfg.rope_theta)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

            self.apply(self._init_weights)

        def _init_weights(self, module: nn.Module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.cfg.n_layers))
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        def count_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

        def forward(
            self,
            input_ids: torch.Tensor,
            targets: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            B, T = input_ids.shape
            x = self.wte(input_ids)

            cos = self.rope_cos[:T].to(x.device)
            sin = self.rope_sin[:T].to(x.device)

            for block in self.blocks:
                x = block(x, cos, sin)

            x = self.final_norm(x)

            if self.lm_head is not None:
                logits = self.lm_head(x)
            else:
                logits = F.linear(x, self.wte.weight)

            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, self.cfg.vocab_size), targets.view(-1))

            return logits, loss
