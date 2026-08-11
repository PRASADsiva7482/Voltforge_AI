"""
VoltForge Foundation LLM — Training Pipeline.
Two-stage training with full backpropagation through all ~9.3M parameters.

Stage 1 — Pretraining (Next-Token Prediction):
    AdamW optimizer, cosine LR decay with warmup, gradient norm clipping.
    Trains on entire corpus (prompt + completion) to learn language patterns.

Stage 2 — Supervised Fine-Tuning (SFT):
    Lower learning rate, fewer epochs. Focuses on completion quality.
"""

import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import TransformerConfig, VoltForgeTransformer
from tokenizer import VoltForgeTokenizer


# ═══════════════════════════════════════════════════════════════════════
# ADAMW OPTIMIZER (Pure NumPy)
# ═══════════════════════════════════════════════════════════════════════

class AdamW:
    """
    AdamW optimizer with decoupled weight decay.
    m_t = β₁·m_{t-1} + (1-β₁)·g
    v_t = β₂·v_{t-1} + (1-β₂)·g²
    θ = θ - lr·(m̂/(√v̂+ε) + λ·θ)
    """

    def __init__(
        self,
        params: Dict[str, np.ndarray],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.95,
        eps: float = 1e-8,
        weight_decay: float = 0.1,
    ):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads: Dict[str, np.ndarray], lr: Optional[float] = None) -> None:
        """Perform one optimization step with the given gradients."""
        self.t += 1
        current_lr = lr if lr is not None else self.lr
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t

        for name, param in self.params.items():
            if name not in grads:
                continue
            g = grads[name]

            # Momentum (first moment)
            self.m[name] = self.beta1 * self.m[name] + (1.0 - self.beta1) * g
            # Variance (second moment)
            self.v[name] = self.beta2 * self.v[name] + (1.0 - self.beta2) * (g * g)

            # Bias correction
            m_hat = self.m[name] / bc1
            v_hat = self.v[name] / bc2

            # Adam update + decoupled weight decay
            update = m_hat / (np.sqrt(v_hat) + self.eps)

            # Only apply weight decay to matrices (not norm scales or embeddings)
            if param.ndim >= 2 and self.weight_decay > 0:
                update = update + self.weight_decay * param

            self.params[name] -= current_lr * update


# ═══════════════════════════════════════════════════════════════════════
# LEARNING RATE SCHEDULER
# ═══════════════════════════════════════════════════════════════════════

def cosine_lr(step: int, warmup_steps: int, max_steps: int,
              peak_lr: float, min_lr: float) -> float:
    """Cosine annealing with linear warmup."""
    if step < warmup_steps:
        return peak_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    progress = min(progress, 1.0)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ═══════════════════════════════════════════════════════════════════════
# GRADIENT UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def clip_grad_norm(grads: Dict[str, np.ndarray], max_norm: float) -> float:
    """Clip gradient global norm. Returns the original norm."""
    total_norm_sq = sum(float(np.sum(g * g)) for g in grads.values())
    total_norm = math.sqrt(total_norm_sq)
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-9)
        for k in grads:
            grads[k] = grads[k] * scale
    return total_norm


def accumulate_grads(acc: Dict[str, np.ndarray], new: Dict[str, np.ndarray]) -> None:
    """Accumulate gradients in-place."""
    for k, g in new.items():
        if k in acc:
            acc[k] += g
        else:
            acc[k] = g.copy()


def zero_grads(acc: Dict[str, np.ndarray]) -> None:
    """Zero all accumulated gradients."""
    for k in acc:
        acc[k].fill(0.0)


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_training_data(
    artifacts_dir: str, tokenizer: VoltForgeTokenizer, max_seq_len: int = 512
) -> List[np.ndarray]:
    """Load and tokenize ALL training datasets. Returns list of token arrays."""
    sequences = []
    bos = tokenizer.special_token_to_id.get("[BOS]", 2)
    eos = tokenizer.special_token_to_id.get("[EOS]", 3)

    # Load from all available JSONL dataset files
    dataset_files = ["master_domain_dataset.jsonl", "dataset.jsonl"]
    for fname in dataset_files:
        fpath = os.path.join(artifacts_dir, fname)
        if not os.path.exists(fpath):
            continue
        count = 0
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                text = sample.get("prompt", "") + " " + sample.get("completion", "")
                tokens = [bos] + tokenizer.encode(text)[:max_seq_len - 2] + [eos]
                if len(tokens) >= 4:
                    sequences.append(np.array(tokens, dtype=np.int32))
                    count += 1
        print(f"    Loaded {count} sequences from {fname}")

    return sequences


# ═══════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════

def train(
    artifacts_dir: str,
    # Stage 1 params
    s1_epochs: int = 10,
    s1_peak_lr: float = 1e-3,
    s1_min_lr: float = 1e-5,
    s1_warmup_steps: int = 200,
    # Stage 2 params
    s2_epochs: int = 3,
    s2_lr: float = 5e-5,
    # Common
    grad_accum_steps: int = 2,
    gradient_clip: float = 1.0,
    weight_decay: float = 0.1,
    checkpoint_every: int = 200,
    max_seq_len: int = 512,
) -> None:
    """Full two-stage training pipeline."""

    # ── Load tokenizer ──
    tokenizer = VoltForgeTokenizer()
    tokenizer_path = artifacts_dir
    if os.path.exists(os.path.join(tokenizer_path, "vocab.json")):
        tokenizer.load(tokenizer_path)
        print(f"[+] Loaded tokenizer: {len(tokenizer.vocab)} tokens")
    else:
        print("[!] No trained tokenizer found. Training new tokenizer...")
        dataset_path = os.path.join(artifacts_dir, "master_domain_dataset.jsonl")
        corpus = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                s = json.loads(line.strip())
                corpus.append(s.get("prompt", "") + " " + s.get("completion", ""))
        tokenizer.train(corpus, target_vocab_size=8192)
        tokenizer.save(artifacts_dir)
        print(f"[+] Trained new tokenizer: {len(tokenizer.vocab)} tokens")

    # ── Create model ──
    config = TransformerConfig(
        vocab_size=len(tokenizer.vocab),
        context_length=max_seq_len,
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=768,
    )
    model = VoltForgeTransformer(config)
    print(f"[+] Model initialized: {model.count_parameters():,} parameters")
    print(f"    Config: {config.to_dict()}")

    # ── Load training data ──
    print(f"[+] Loading training data from {artifacts_dir}...")
    sequences = load_training_data(artifacts_dir, tokenizer, max_seq_len)
    print(f"[+] Loaded {len(sequences)} training sequences")
    avg_len = sum(len(s) for s in sequences) / len(sequences)
    print(f"    Average sequence length: {avg_len:.1f} tokens")

    # ── Save model config ──
    config_path = os.path.join(artifacts_dir, "model_meta.json")
    with open(config_path, "w") as f:
        json.dump({"model_config": config.to_dict(), "num_params": model.count_parameters(),
                    "num_sequences": len(sequences), "avg_seq_len": avg_len}, f, indent=2)

    # ═══════════ STAGE 1: PRETRAINING ═══════════
    print("\n" + "=" * 60)
    print("STAGE 1: PRETRAINING (Next-Token Prediction)")
    print("=" * 60)

    optimizer = AdamW(model.weights, lr=s1_peak_lr, weight_decay=weight_decay)
    total_s1_steps = (len(sequences) * s1_epochs) // grad_accum_steps
    global_step = 0
    best_loss = float("inf")

    for epoch in range(1, s1_epochs + 1):
        epoch_start = time.time()
        np.random.shuffle(sequences)
        epoch_loss = 0.0
        batch_count = 0
        accumulated_grads: Dict[str, np.ndarray] = {}

        for seq_idx, seq in enumerate(sequences):
            # Prepare input/target pairs (shifted by 1)
            input_ids = seq[:-1]
            targets = seq[1:]

            # Forward + backward
            loss, grads = model.compute_loss_and_grads(input_ids, targets)
            epoch_loss += loss

            # Accumulate gradients
            accumulate_grads(accumulated_grads, grads)

            if (seq_idx + 1) % grad_accum_steps == 0:
                # Average gradients
                for k in accumulated_grads:
                    accumulated_grads[k] /= grad_accum_steps

                # Clip gradients
                gnorm = clip_grad_norm(accumulated_grads, gradient_clip)

                # Compute learning rate
                lr = cosine_lr(global_step, s1_warmup_steps, total_s1_steps, s1_peak_lr, s1_min_lr)

                # Optimizer step
                optimizer.step(accumulated_grads, lr=lr)

                global_step += 1
                batch_count += 1
                accumulated_grads = {}

                # Logging
                if batch_count % 5 == 0:
                    avg_loss = epoch_loss / (seq_idx + 1)
                    elapsed = time.time() - epoch_start
                    tok_per_sec = sum(len(sequences[j]) for j in range(seq_idx + 1)) / elapsed
                    print(f"  Epoch {epoch}/{s1_epochs} | Step {global_step} | "
                          f"Loss {avg_loss:.4f} | LR {lr:.2e} | GNorm {gnorm:.2f} | "
                          f"{tok_per_sec:.0f} tok/s", flush=True)

                # Checkpoint
                if global_step % checkpoint_every == 0:
                    ckpt_path = os.path.join(artifacts_dir, f"checkpoint_s1_step{global_step}.npz")
                    model.save_weights(ckpt_path)
                    print(f"  [ckpt] Saved {ckpt_path}", flush=True)

        avg_epoch_loss = epoch_loss / len(sequences)
        elapsed = time.time() - epoch_start
        print(f"[Epoch {epoch}/{s1_epochs}] Avg Loss: {avg_epoch_loss:.4f} | "
              f"Time: {elapsed:.1f}s | Perplexity: {math.exp(min(avg_epoch_loss, 20)):.2f}", flush=True)

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            model.save_weights(os.path.join(artifacts_dir, "model_weights_best.npz"))
            print(f"  [*] New best model saved (loss={best_loss:.4f})", flush=True)

    # Save Stage 1 final weights
    model.save_weights(os.path.join(artifacts_dir, "model_weights_s1.npz"))
    print(f"\n[+] Stage 1 complete. Best loss: {best_loss:.4f}")

    # ═══════════ STAGE 2: SUPERVISED FINE-TUNING ═══════════
    print("\n" + "=" * 60)
    print("STAGE 2: SUPERVISED FINE-TUNING (SFT)")
    print("=" * 60)

    # Reset optimizer with lower LR
    optimizer = AdamW(model.weights, lr=s2_lr, weight_decay=weight_decay * 0.5)

    for epoch in range(1, s2_epochs + 1):
        epoch_start = time.time()
        np.random.shuffle(sequences)
        epoch_loss = 0.0
        accumulated_grads = {}

        for seq_idx, seq in enumerate(sequences):
            input_ids = seq[:-1]
            targets = seq[1:]
            loss, grads = model.compute_loss_and_grads(input_ids, targets)
            epoch_loss += loss
            accumulate_grads(accumulated_grads, grads)

            if (seq_idx + 1) % grad_accum_steps == 0:
                for k in accumulated_grads:
                    accumulated_grads[k] /= grad_accum_steps
                clip_grad_norm(accumulated_grads, gradient_clip)
                optimizer.step(accumulated_grads, lr=s2_lr)
                accumulated_grads = {}

        avg_epoch_loss = epoch_loss / len(sequences)
        elapsed = time.time() - epoch_start
        print(f"[SFT Epoch {epoch}/{s2_epochs}] Avg Loss: {avg_epoch_loss:.4f} | "
              f"Time: {elapsed:.1f}s | Perplexity: {math.exp(min(avg_epoch_loss, 20)):.2f}")

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            model.save_weights(os.path.join(artifacts_dir, "model_weights_best.npz"))

    # Save final weights
    model.save_weights(os.path.join(artifacts_dir, "model_weights.npz"))
    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total parameters: {model.count_parameters():,}")
    print(f"Final loss: {best_loss:.4f}")
    print(f"Final perplexity: {math.exp(min(best_loss, 20)):.2f}")
    print(f"Weights saved to: {os.path.join(artifacts_dir, 'model_weights.npz')}")

    # Quick generation test
    print("\n[*] Generation test:")
    test_prompt = "[SYS] You are VoltForge AI. [USER] What is a resistor?"
    test_tokens = [tokenizer.special_token_to_id.get("[BOS]", 2)] + tokenizer.encode(test_prompt)
    generated = model.generate(test_tokens, max_new_tokens=60, temperature=0.7, top_k=40)
    print(f"  Input:  {test_prompt}")
    print(f"  Output: {tokenizer.decode(generated)}")


if __name__ == "__main__":
    artifacts = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
    train(artifacts)
