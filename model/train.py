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
from model import NumPyTransformer, TransformerConfig, softmax
from tokenizer import VoltForgeTokenizer


def train_pipeline(num_samples: int = 10000, vocab_size: int = 4096) -> None:
    artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    print(f"[*] Step 1: Generating {num_samples} domain-specific synthetic training samples...")
    raw_data = build_full_dataset(num_samples)
    text_corpus = [f"{item['prompt']} {item['completion']}" for item in raw_data]

    print(f"[*] Step 2: Training custom BPE Tokenizer (target vocab={vocab_size})...")
    tokenizer = VoltForgeTokenizer(vocab_size=vocab_size)
    tokenizer.train(text_corpus[:2000], target_vocab_size=vocab_size)
    tokenizer.save(artifacts_dir)
    print(f"[+] Tokenizer saved to {artifacts_dir} (Vocab: {len(tokenizer.vocab)} tokens, Merges: {len(tokenizer.merges)})")

    # Save dataset jsonl
    dataset_file = os.path.join(artifacts_dir, "dataset.jsonl")
    with open(dataset_file, "w", encoding="utf-8") as f:
        for item in raw_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

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
    sample_tokens = [tokenizer.encode(item["prompt"] + " " + item["completion"])[:128] for item in raw_data[:100]]
    
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
