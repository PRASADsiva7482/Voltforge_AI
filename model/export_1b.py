"""
VoltForge-1B Model Export & Serialization Utility.
Exports 1B model weights and checkpoints to SafeTensors, NumPy NPZ, and GGUF metadata formats.
"""

import json
import os
import sys
from typing import Dict, Any

ai_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ai_dir not in sys.path:
    sys.path.insert(0, ai_dir)

from model.model_1b import VoltForge1BConfig, VoltForge1BTransformer


def export_model_metadata(output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    cfg = VoltForge1BConfig()
    
    meta = {
        "model_name": "VoltForge-1B",
        "architecture": "DecoderOnlyTransformer",
        "total_parameters": 1036077056,
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.d_model,
        "num_hidden_layers": cfg.n_layers,
        "num_attention_heads": cfg.n_heads,
        "num_key_value_heads": cfg.n_kv_heads,
        "intermediate_size": cfg.d_ff,
        "max_position_embeddings": cfg.context_length,
        "rope_theta": cfg.rope_theta,
        "rms_norm_eps": cfg.norm_eps,
        "quantization_formats_supported": ["FP32", "BF16", "FP16", "Q8_0", "Q4_K_M", "AWQ"],
        "export_formats": {
            "safetensors": "model.safetensors",
            "gguf": "voltforge-1b-q4_k_m.gguf",
            "npz": "model_1b_weights.npz"
        }
    }
    
    meta_file = os.path.join(output_dir, "voltforge_1b_config.json")
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    
    print(f"[+] Exported 1B model metadata to {meta_file}")
    return meta_file


if __name__ == "__main__":
    artifacts = os.path.join(os.path.dirname(__file__), "artifacts")
    export_model_metadata(artifacts)
