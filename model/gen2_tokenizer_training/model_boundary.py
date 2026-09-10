"""Executable tokenizer-bound neural decoder for task024 integration evidence.

Uses reviewed project-owned Gen1 tensor mathematics under a new Gen2 input
binding. This finite integration profile is not task025's architecture sweep,
a trained model, or a serving artifact. No historical model config is modified.
"""
from model.gen2_tokenizer.contract import canonical, require, sha
from context_compiler.gen2 import PromptIds
from . import release


class BoundDecoder:
    def __init__(self, tokenizer):
        require(type(tokenizer) is release.ReleasedTokenizer, "GEN2_MODEL_REQUIRES_RELEASED_TOKENIZER")
        version = release.read_json(release.VERSION_RECORD)
        expected, _ = release.load((release.AI / version["releaseManifest"]["path"]).parent)
        require(canonical(tokenizer.binding()) == canonical(expected.binding()), "GEN2_MODEL_TOKENIZER_MISMATCH")
        # Import the pinned native runtime only after tokenizer admission passes.
        import torch
        from model.gen1.config import Gen1Config
        from model.gen1.model import VoltForgeGen1
        self.tokenizer_binding = tokenizer.binding()
        self.config = Gen1Config(vocab_size=tokenizer.vocab_size, max_sequence_length=4096, d_model=32, n_layers=1, n_heads=4, n_kv_heads=2, d_ff=64, seed=24017)
        self.decoder = VoltForgeGen1.for_training(self.config, device="cpu", dtype=torch.float32, allocation_limit=2_000_000)
        self.decoder.eval()
        self.specification = {"modelFamily": "vfdlm-g2", "integrationProfileId": "vfdlm-g2-tokenizer-bound-decoder-v1", "tokenizer": self.tokenizer_binding,
                              "kernelConfig": self.config.to_dict(), "kernelImplementation": "reviewed project-owned Gen1 causal decoder mathematics via composition",
                              "randomInitialization": True, "weightsLoaded": False, "trainedModel": False, "servingAllowed": False}
        self.assert_binding(tokenizer.binding())

    def assert_binding(self, binding):
        require(canonical(binding) == canonical(self.tokenizer_binding) == canonical(self.specification["tokenizer"]), "GEN2_MODEL_TOKENIZER_MISMATCH")
        require(self.decoder.token_embedding.num_embeddings == binding["vocabSize"] and self.decoder.lm_head.out_features == binding["vocabSize"] and self.decoder.config.vocab_size == binding["vocabSize"], "GEN2_MODEL_VOCAB_SHAPE_MISMATCH")
        require(self.decoder.token_embedding.weight.shape == self.decoder.lm_head.weight.shape == (binding["vocabSize"], self.config.d_model), "GEN2_MODEL_VOCAB_SHAPE_MISMATCH")
        return dict(binding)

    def forward(self, prompt):
        import torch
        require(isinstance(prompt, PromptIds), "GEN2_MODEL_PROMPT_REQUIRED")
        self.assert_binding(prompt.binding)
        require(len(prompt.token_ids) + prompt.reserved_output_tokens <= prompt.context_limit <= self.config.max_sequence_length and len(prompt.token_ids) == prompt.content_tokens + prompt.framing_tokens, "GEN2_MODEL_CONTEXT_BUDGET_MISMATCH")
        require(all(type(token) is int and 0 <= token < self.config.vocab_size for token in prompt.token_ids), "GEN2_MODEL_INPUT_TOKEN_INVALID")
        with torch.inference_mode():
            return self.decoder(torch.tensor([prompt.token_ids], dtype=torch.long))

    def initialization_fingerprint(self):
        # Exact in-memory tensor bytes; no checkpoint serialization or loading.
        records = [{"name": name, "shape": list(tensor.shape), "sha256": sha(tensor.detach().cpu().contiguous().numpy().tobytes())} for name, tensor in self.decoder.state_dict().items()]
        return sha(canonical(records))
