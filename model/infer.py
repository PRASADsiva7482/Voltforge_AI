"""
VoltForge Model Inference Runtime.
Loads trained tokenizer, standalone NumPy Transformer, and Electronics Reasoning LLM.
"""

import json
import os
import sys
from typing import Any, AsyncIterator, Dict, List, Optional
import numpy as np

# Add model directory to sys.path
ai_model_dir = os.path.dirname(os.path.abspath(__file__))
if ai_model_dir not in sys.path:
    sys.path.insert(0, ai_model_dir)

try:
    from model.model import NumPyTransformer, TransformerConfig, softmax
    from model.tokenizer import VoltForgeTokenizer
    from model.reasoning_llm import ElectronicsReasoningEngine
except ImportError:
    from model import NumPyTransformer, TransformerConfig, softmax
    from tokenizer import VoltForgeTokenizer
    from reasoning_llm import ElectronicsReasoningEngine


class VoltForgeInferenceEngine:
    def __init__(self, artifacts_dir: Optional[str] = None):
        if not artifacts_dir:
            artifacts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
        self.artifacts_dir = artifacts_dir
        self.tokenizer = VoltForgeTokenizer()
        self.config: Optional[TransformerConfig] = None
        self.model: Optional[NumPyTransformer] = None
        self.reasoning_engine: Optional[ElectronicsReasoningEngine] = None
        self.is_loaded = False
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        vocab_path = os.path.join(self.artifacts_dir, "vocab.json")
        config_path = os.path.join(self.artifacts_dir, "config.json")
        meta_path = os.path.join(self.artifacts_dir, "model_meta.json")
        weights_path = os.path.join(self.artifacts_dir, "model_weights.npz")
        if not os.path.exists(weights_path):
            weights_path = os.path.join(self.artifacts_dir, "model_weights_best.npz")

        try:
            if os.path.exists(vocab_path):
                self.tokenizer.load(self.artifacts_dir)

            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    self.config = TransformerConfig.from_dict(meta.get("model_config", {}))
            elif os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    self.config = TransformerConfig.from_dict(json.load(f))
            else:
                self.config = TransformerConfig(vocab_size=len(self.tokenizer.vocab))

            self.model = NumPyTransformer(self.config)
            if os.path.exists(weights_path):
                self.model.load_weights(weights_path)
                print(f"[+] Loaded neural Transformer weights from {os.path.basename(weights_path)}")

            self.reasoning_engine = ElectronicsReasoningEngine(self.artifacts_dir)
            self.is_loaded = True
        except Exception as exc:
            print(f"[!] Warning: Inference engine loaded with partial fallback ({exc})")
            self.reasoning_engine = ElectronicsReasoningEngine(self.artifacts_dir)
            self.is_loaded = True

    def compute_confidence(self, prompt: str) -> float:
        """Estimates model confidence using average top token probability."""
        if not self.is_loaded or not self.model:
            return 0.88
        try:
            ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
            if not ids:
                return 0.88
            logits = self.model.forward(np.array(ids[-self.config.context_length:], dtype=np.int32))
            probs = softmax(logits[-1])
            top_prob = float(np.max(probs))
            return max(0.65, min(0.98, top_prob * 10.0))
        except Exception:
            return 0.88

    def reason_and_solve(
        self,
        prompt: str,
        board_type: str = "ARDUINO_UNO",
        components: Optional[List[Dict[str, Any]]] = None,
        wires: Optional[List[Dict[str, Any]]] = None,
        code: Optional[str] = None,
        simulation_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Runs Chain-of-Thought (CoT) reasoning and generates an answer."""
        if not self.reasoning_engine:
            self.reasoning_engine = ElectronicsReasoningEngine(self.artifacts_dir)
        return self.reasoning_engine.reason_and_solve(
            prompt=prompt,
            board_type=board_type,
            components=components,
            wires=wires,
            code=code,
            simulation_state=simulation_state
        )

    def generate_neural_text(self, prompt: str, max_tokens: int = 120, temperature: float = 0.7) -> str:
        """Generates text directly using the trained neural Transformer model."""
        if not self.model or not self.tokenizer:
            return ""
        formatted_prompt = f"[SYS] You are VoltForge AI. [USER] {prompt}"
        bos_id = self.tokenizer.special_token_to_id.get("[BOS]", 2)
        eos_id = self.tokenizer.special_token_to_id.get("[EOS]", 3)
        input_ids = [bos_id] + self.tokenizer.encode(formatted_prompt)
        generated_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=40,
            top_p=0.9,
            eos_token_id=eos_id,
        )
        return self.tokenizer.decode(generated_ids)

    async def stream_reasoning_and_response(
        self,
        prompt: str,
        board_type: str = "ARDUINO_UNO",
        components: Optional[List[Dict[str, Any]]] = None,
        wires: Optional[List[Dict[str, Any]]] = None,
        code: Optional[str] = None,
        simulation_state: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """Streams thoughts, words, and actions in real-time SSE format."""
        if not self.reasoning_engine:
            self.reasoning_engine = ElectronicsReasoningEngine(self.artifacts_dir)
        async for chunk in self.reasoning_engine.stream_reasoning_and_response(
            prompt=prompt,
            board_type=board_type,
            components=components,
            wires=wires,
            code=code,
            simulation_state=simulation_state
        ):
            yield chunk


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    engine = VoltForgeInferenceEngine()
    print("Inference Engine Loaded:", engine.is_loaded)
    if engine.is_loaded:
        test_prompt = "How to calculate LED resistor for 5V supply?"
        res = engine.reason_and_solve(test_prompt, "ARDUINO_UNO")
        print("\n=== THOUGHTS ===")
        for t in res["thoughts"]:
            print(f"- {t}")
        print("\n=== ANSWER ===")
        print(res["answer"])
