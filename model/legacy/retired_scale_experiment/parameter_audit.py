"""
Retired scale-experiment parameter, memory, and training FLOPs calculator.
Calculates exact tensor dimensions, VRAM footprint, Chinchilla token scaling, and GPU cluster wall-clock times.
"""

import json
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Model1BConfig:
    vocab_size: int = 32768
    context_length: int = 4096
    d_model: int = 2048
    n_layers: int = 20
    n_heads: int = 16
    n_kv_heads: int = 4
    head_dim: int = 128
    d_ff: int = 5632
    tie_embeddings: bool = False


class ParameterAuditor:
    def __init__(self, config: Model1BConfig):
        self.cfg = config

    def calculate_layer_parameters(self) -> Dict[str, Any]:
        c = self.cfg
        # 1. Embedding
        embedding_params = c.vocab_size * c.d_model

        # 2. Per-Layer Attention (Grouped-Query Attention)
        q_params = c.d_model * (c.n_heads * c.head_dim)
        k_params = c.d_model * (c.n_kv_heads * c.head_dim)
        v_params = c.d_model * (c.n_kv_heads * c.head_dim)
        o_params = (c.n_heads * c.head_dim) * c.d_model
        attn_per_layer = q_params + k_params + v_params + o_params

        # 3. Per-Layer SwiGLU FFN
        gate_params = c.d_model * c.d_ff
        up_params = c.d_model * c.d_ff
        down_params = c.d_ff * c.d_model
        ffn_per_layer = gate_params + up_params + down_params

        # 4. Normalization
        norm_per_layer = 2 * c.d_model  # attn_norm + ffn_norm
        final_norm = c.d_model

        # 5. LM Head
        lm_head_params = 0 if c.tie_embeddings else (c.d_model * c.vocab_size)

        # Totals
        block_params = attn_per_layer + ffn_per_layer + norm_per_layer
        all_blocks_params = block_params * c.n_layers
        total_params = embedding_params + all_blocks_params + final_norm + lm_head_params

        return {
            "embedding_params": embedding_params,
            "attn_q_params": q_params,
            "attn_k_params": k_params,
            "attn_v_params": v_params,
            "attn_o_params": o_params,
            "attn_per_layer": attn_per_layer,
            "ffn_gate_params": gate_params,
            "ffn_up_params": up_params,
            "ffn_down_params": down_params,
            "ffn_per_layer": ffn_per_layer,
            "norm_per_layer": norm_per_layer,
            "block_params": block_params,
            "all_blocks_params": all_blocks_params,
            "final_norm": final_norm,
            "lm_head_params": lm_head_params,
            "total_params": total_params,
        }

    def calculate_memory_footprint(self, total_params: int) -> Dict[str, float]:
        # Memory in Gigabytes (GB)
        fp32_gb = (total_params * 4) / (1024 ** 3)
        bf16_gb = (total_params * 2) / (1024 ** 3)
        int8_gb = (total_params * 1) / (1024 ** 3)
        int4_gb = (total_params * 0.5) / (1024 ** 3)

        # Training VRAM (AdamW 32-bit master weights + momentum + variance = 12 bytes/param)
        grad_bf16_gb = bf16_gb
        optimizer_adamw_gb = (total_params * 12) / (1024 ** 3)
        optimizer_8bit_gb = (total_params * 4) / (1024 ** 3)

        return {
            "fp32_weights_gb": fp32_gb,
            "bf16_weights_gb": bf16_gb,
            "int8_quantized_gb": int8_gb,
            "int4_quantized_gb": int4_gb,
            "gradients_bf16_gb": grad_bf16_gb,
            "adamw_optimizer_gb": optimizer_adamw_gb,
            "adamw_8bit_optimizer_gb": optimizer_8bit_gb,
            "total_training_mem_single_gpu_gb": bf16_gb + grad_bf16_gb + optimizer_adamw_gb + 4.0,  # +4GB acts
            "fsdp_4gpu_per_gpu_gb": (bf16_gb + (grad_bf16_gb + optimizer_adamw_gb) / 4) + 2.5,
            "fsdp_8gpu_per_gpu_gb": (bf16_gb + (grad_bf16_gb + optimizer_adamw_gb) / 8) + 2.5,
        }

    def calculate_compute_and_time(self, total_params: int, tokens: int) -> Dict[str, Any]:
        # 6ND FLOPs calculation
        flops_total = 6 * total_params * tokens
        pflops_seconds = flops_total / 1e15

        # Hardware specs: (Effective TFLOPs/s per GPU, Typical Node Count, MFU)
        clusters = {
            "8x NVIDIA H100 (80GB SXM5)": {"tflops_peak": 989, "mfu": 0.48, "gpus": 8, "cost_hr": 28.0},
            "8x NVIDIA A100 (80GB SXM4)": {"tflops_peak": 312, "mfu": 0.42, "gpus": 8, "cost_hr": 16.0},
            "4x NVIDIA RTX 4090 (24GB)": {"tflops_peak": 165, "mfu": 0.38, "gpus": 4, "cost_hr": 8.0},
            "1x NVIDIA RTX 4090 (24GB)": {"tflops_peak": 165, "mfu": 0.35, "gpus": 1, "cost_hr": 2.0},
        }

        results = {}
        for name, spec in clusters.items():
            eff_tflops = spec["tflops_peak"] * spec["mfu"] * spec["gpus"]
            eff_flops_per_sec = eff_tflops * 1e12
            time_seconds = flops_total / eff_flops_per_sec
            time_hours = time_seconds / 3600
            time_days = time_hours / 24
            estimated_cost = time_hours * spec["cost_hr"]

            results[name] = {
                "effective_tflops_s": eff_tflops,
                "time_seconds": time_seconds,
                "time_hours": time_hours,
                "time_days": time_days,
                "estimated_cost_usd": estimated_cost,
            }

        return {
            "tokens": tokens,
            "flops_total": flops_total,
            "pflops_seconds": pflops_seconds,
            "clusters": results,
        }


def print_1b_report():
    cfg = Model1BConfig()
    auditor = ParameterAuditor(cfg)
    params = auditor.calculate_layer_parameters()
    mem = auditor.calculate_memory_footprint(params["total_params"])
    compute_20b = auditor.calculate_compute_and_time(params["total_params"], 20_000_000_000)
    compute_3b = auditor.calculate_compute_and_time(params["total_params"], 3_000_000_000)

    print("=" * 80)
    print("VOLTFORGE-1B PARAMETER, MEMORY & TRAINING REPORT")
    print("=" * 80)
    print(f"Total Model Parameters: {params['total_params']:,} ({params['total_params'] / 1e9:.3f} Billion)")
    print(f"Vocabulary Size:        {cfg.vocab_size:,} tokens")
    print(f"Hidden Dimension:       {cfg.d_model} (Heads: {cfg.n_heads}, KV Heads: {cfg.n_kv_heads})")
    print(f"Transformer Layers:     {cfg.n_layers}")
    print(f"SwiGLU FFN Dimension:   {cfg.d_ff}")
    print(f"Context Window:         {cfg.context_length} tokens")
    print("-" * 80)
    print("LAYER PARAMETER BREAKDOWN:")
    print(f"  * Token Embeddings (Wte):      {params['embedding_params']:,} ({params['embedding_params']/1e6:.2f}M)")
    print(f"  * Attention per layer (GQA):   {params['attn_per_layer']:,} ({params['attn_per_layer']/1e6:.2f}M)")
    print(f"  * SwiGLU FFN per layer:        {params['ffn_per_layer']:,} ({params['ffn_per_layer']/1e6:.2f}M)")
    print(f"  * Total per Transformer Block: {params['block_params']:,} ({params['block_params']/1e6:.2f}M)")
    print(f"  * All {cfg.n_layers} Transformer Blocks:   {params['all_blocks_params']:,} ({params['all_blocks_params']/1e6:.2f}M)")
    print(f"  * LM Head:                     {params['lm_head_params']:,} ({params['lm_head_params']/1e6:.2f}M)")
    print("-" * 80)
    print("MEMORY FOOTPRINT (VRAM):")
    print(f"  * FP32 Model Weights:          {mem['fp32_weights_gb']:.2f} GB")
    print(f"  * BF16 / FP16 Weights:         {mem['bf16_weights_gb']:.2f} GB")
    print(f"  * INT8 Quantized (GGUF):       {mem['int8_quantized_gb']:.2f} GB")
    print(f"  * INT4 Quantized (AWQ/GGUF):   {mem['int4_quantized_gb']:.2f} GB")
    print(f"  * FSDP on 4 GPUs (per GPU):    {mem['fsdp_4gpu_per_gpu_gb']:.2f} GB VRAM")
    print(f"  * FSDP on 8 GPUs (per GPU):    {mem['fsdp_8gpu_per_gpu_gb']:.2f} GB VRAM")
    print("-" * 80)
    print("TRAINING SCALING (20 BILLION TOKENS - CHINCHILLA OPTIMAL):")
    print(f"  * Compute Required:            {compute_20b['pflops_seconds']:.2f} PFLOP-seconds")
    for cluster, stats in compute_20b["clusters"].items():
        print(f"    - {cluster:<30}: {stats['time_hours']:>6.1f} hours ({stats['time_days']:>4.1f} days) | Est. Cost: ${stats['estimated_cost_usd']:>6.0f}")
    print("-" * 80)
    print("DOMAIN CONTINUAL TRAINING (3 BILLION TOKENS - ELECTRONICS FOCUS):")
    print(f"  * Compute Required:            {compute_3b['pflops_seconds']:.2f} PFLOP-seconds")
    for cluster, stats in compute_3b["clusters"].items():
        print(f"    - {cluster:<30}: {stats['time_hours']:>6.1f} hours ({stats['time_days']:>4.1f} days) | Est. Cost: ${stats['estimated_cost_usd']:>6.0f}")
    print("=" * 80)


if __name__ == "__main__":
    print_1b_report()
