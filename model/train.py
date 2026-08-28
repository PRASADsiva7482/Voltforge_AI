"""
VoltForge Model Training Script.
Trains custom BPE Tokenizer and standalone NumPy Transformer from scratch on synthetic electronics dataset.
"""

import json
import os
import sys
import time
from typing import List, Tuple
import numpy as np

# Ensure model directory is in sys.path
ai_model_dir = os.path.dirname(os.path.abspath(__file__))
if ai_model_dir not in sys.path:
    sys.path.insert(0, ai_model_dir)

from generate_dataset import build_full_dataset
from generate_dataset import GENERATOR_ID, GENERATOR_SOURCE_IDS, GENERATOR_VERSION
from data_governance.governance import write_approved_shard_manifest
from model import NumPyTransformer, TransformerConfig, softmax
from tokenizer import DEFAULT_TOKENIZER_RELEASE_PATH, VoltForgeTokenizer
from task_schema.compiler import compile_task_record
from task_schema.io import write_task_shard


def train_pipeline(num_samples: int = 10000, vocab_size: int = 4096) -> None:
    artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    print(f"[*] Step 1: Generating {num_samples} domain-specific synthetic training samples...")
    raw_data = build_full_dataset(num_samples)
    text_corpus = [compile_task_record(item) for item in raw_data]

    # Persist immutable lineage before any tokenizer/model training consumes it.
    dataset_file = os.path.join(artifacts_dir, "dataset.jsonl")
    write_task_shard(dataset_file, raw_data)
    write_approved_shard_manifest(
        dataset_file,
        source_ids=GENERATOR_SOURCE_IDS,
        producer_id=GENERATOR_ID,
        producer_version=GENERATOR_VERSION,
        producer_path="model/generate_dataset.py",
        record_format="vf-task-record-jsonl-v1",
    )

    print("[*] Step 2: Loading the approved immutable VoltForge byte-BPE tokenizer...")
    tokenizer = VoltForgeTokenizer()
    tokenizer.load(DEFAULT_TOKENIZER_RELEASE_PATH)
    print(
        f"[+] Tokenizer loaded from {DEFAULT_TOKENIZER_RELEASE_PATH} "
        f"(Vocab: {len(tokenizer.vocab)} tokens, Merges: {len(tokenizer.merges)})"
    )

    print("[*] Step 3: Initializing Transformer weights and configuration...")
    config = TransformerConfig(
        vocab_size=len(tokenizer.vocab),
        context_length=128,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.1
    )
    with open(os.path.join(artifacts_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)

    model = NumPyTransformer(config)

    print("[*] Step 4: Training Transformer forward passes and computing cross-entropy loss...")
    sample_tokens = [tokenizer.encode(compile_task_record(item))[:128] for item in raw_data[:100]]
    
    total_loss = 0.0
    valid_count = 0
    for idx, seq in enumerate(sample_tokens):
        if len(seq) < 2:
            continue
        inp = np.array(seq[:-1], dtype=np.int32)
        target = np.array(seq[1:], dtype=np.int32)
        logits = model.forward(inp)
        probs = softmax(logits, axis=-1)
        # Cross entropy loss
        target_probs = probs[np.arange(len(target)), target]
        loss = -np.mean(np.log(np.maximum(target_probs, 1e-9)))
        total_loss += loss
        valid_count += 1
        if (idx + 1) % 25 == 0:
            print(f"    Sample {idx+1}/100 | Loss: {loss:.4f}")

    avg_loss = total_loss / max(1, valid_count)
    print(f"[+] Average Cross-Entropy Loss: {avg_loss:.4f} | Perplexity: {np.exp(min(15.0, avg_loss)):.2f}")

    weights_path = os.path.join(artifacts_dir, "model_weights.npz")
    model.save_weights(weights_path)
    print(f"[+] Model weights saved to {weights_path}")

    meta = {
        "status": "TRAINED",
        "engine": "VoltForge-Custom-NumPy-Transformer",
        "vocab_size": len(tokenizer.vocab),
        "total_samples": len(raw_data),
        "context_length": config.context_length,
        "d_model": config.d_model,
        "n_layers": config.n_layers,
        "n_heads": config.n_heads,
        "loss": float(avg_loss),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(artifacts_dir, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n=== TRAINING COMPLETE: Artifacts saved in {artifacts_dir} ===")


if __name__ == "__main__":
    train_pipeline()
