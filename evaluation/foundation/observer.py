"""Evaluator-owned decoding and process-local, request-bound invocation evidence.

Self-reported token counts and serialized proof dictionaries are never admitted.
The random Gen1 probe is numerical instrumentation evidence only, never a release.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import secrets
import re
import inspect
from pathlib import Path

from .suite import canonical, sha


@dataclass(frozen=True)
class _Witness:
    nonce: str
    request_sha256: str
    output_sha256: str
    token_ids: tuple[int, ...]
    forward_calls: int
    logits_elements: int
    logits_sha256: str
    artifact_id: str | None
    release_eligible: bool


class InvocationLedger:
    def __init__(self):
        self._issued = {}
        self._admitted_models = {}

    def _decode(self, *, model, input_ids, decode, request, max_new_tokens, eos_id=None):
        import torch
        if not isinstance(model, torch.nn.Module) or not input_ids or max_new_tokens < 1:
            raise ValueError("A real native model, nonempty token input and finite token budget are required")
        if isinstance(max_new_tokens, bool) or max_new_tokens > 256:
            raise ValueError("Evaluation generation limit exceeded")
        model.eval()
        device = next(model.parameters()).device
        ids = list(input_ids)
        generated, hashes = [], []
        logits_elements = 0
        random = torch.Generator(device="cpu")
        random.manual_seed(int(request.get("seed", 17018)))
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                result = model(torch.tensor([ids], dtype=torch.long, device=device), use_cache=False)
                logits = result.logits
                if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != len(ids) or logits.shape[2] < 2 or not torch.isfinite(logits).all().item():
                    raise ValueError("Invalid observed model logits")
                last = logits[0, -1].detach().float().cpu().contiguous()
                hashes.append(sha(last.numpy().tobytes()))
                logits_elements += last.numel()
                top_values, top_ids = torch.topk(last, min(20, last.numel()))
                probabilities = torch.softmax(top_values / 0.8, dim=-1)
                token_id = int(top_ids[torch.multinomial(probabilities, 1, generator=random)].item())
                generated.append(token_id)
                ids.append(token_id)
                if token_id == eos_id:
                    break
        output = decode(generated)
        if not isinstance(output, str):
            raise ValueError("Tokenizer decoding must return text")
        witness = _Witness(secrets.token_hex(16), sha(canonical(request).encode()), sha(output.encode()), tuple(generated),
                           len(generated), logits_elements, sha(canonical(hashes).encode()),
                           self._admitted_models.get(id(model)), id(model) in self._admitted_models)
        self._issued[witness.nonce] = witness
        return {"text": output, "witness": witness}

    def inspect(self, response, request):
        witness = response.get("witness")
        valid = (isinstance(witness, _Witness) and self._issued.get(witness.nonce) is witness
                 and witness.request_sha256 == sha(canonical(request).encode())
                 and witness.output_sha256 == sha(response.get("text", "").encode())
                 and witness.forward_calls > 0 and witness.logits_elements > 0 and bool(witness.token_ids))
        return {"observed": bool(valid), "releaseEligible": bool(valid and witness.release_eligible),
                "forwardCalls": witness.forward_calls if valid else 0,
                "logitsElements": witness.logits_elements if valid else 0,
                "generatedTokens": len(witness.token_ids) if valid else 0,
                "artifactId": witness.artifact_id if valid else None,
                "outputSha256": witness.output_sha256 if valid else None,
                "logitsSha256": witness.logits_sha256 if valid else None}

    def random_probe(self, request):
        # Import only the verified local Gen1 runtime; no checkpoint load or optimizer.
        from model.gen1.config import Gen1Config
        from model.gen1.model import VoltForgeGen1
        config = Gen1Config(vocab_size=32, d_model=16, n_layers=1, n_heads=2, n_kv_heads=1,
                            d_ff=32, max_sequence_length=16, seed=17018)
        model = VoltForgeGen1.for_training(config, device="cpu", allocation_limit=100_000)
        return self._decode(model=model, input_ids=[1, 4, 7], decode=lambda ids: " ".join(map(str, ids)),
                            request=request, max_new_tokens=3)

    def candidate_adapter(self, model, tokenizer, attestation, trust):
        """Admit an inactive owned training candidate without production activation.

        Tasks 024/027 supply the owned Gen2 classes and signed training lineage;
        task 028 supplies the validated checkpoint runtime. The evaluator checks
        actual tensor bytes and owns formatting/decoding, not an answer callback.
        """
        from model.runtime_service import checkpoint_runtime_security_block
        from .grading import verify_signed_receipt
        from .suite import verify_suite
        block = checkpoint_runtime_security_block()
        if block is not None:
            raise ValueError(block[0])
        expected_model = importlib.import_module("model.gen2.model").VoltForgeGen2
        expected_tokenizer = importlib.import_module("model.gen2.tokenizer").Gen2Tokenizer
        if type(model) is not expected_model or type(tokenizer) is not expected_tokenizer:
            raise ValueError("Only audited owned Gen2 model/tokenizer classes are evaluation candidates")
        if getattr(model.forward, "__func__", None) is not expected_model.forward or getattr(tokenizer.decode, "__func__", None) is not expected_tokenizer.decode:
            raise ValueError("Overridden forward/decode callback is not an owned candidate")
        tensors = []
        for name, value in sorted(model.state_dict().items()):
            raw = value.detach().cpu().contiguous()
            # Byte view supports all pinned floating dtypes, including bfloat16.
            import torch
            tensors.append({"name": name, "shape": list(raw.shape), "dtype": str(raw.dtype), "sha256": sha(raw.view(torch.uint8).numpy().tobytes())})
        suite = verify_suite()
        required = {"kind": "owned-gen2-evaluation-candidate", "familyNamespace": "vfdlm-g2", "pretrainedWeightsUsed": False,
                    "tensorManifestSha256": sha(canonical(tensors).encode()), "tokenizerSha256": tokenizer.evaluation_fingerprint(),
                    "modelCodeSha256": sha(Path(inspect.getsourcefile(expected_model)).read_bytes()),
                    "tokenizerCodeSha256": sha(Path(inspect.getsourcefile(expected_tokenizer)).read_bytes()),
                    "suiteSha256": suite["suiteSha256"], "policySha256": suite["policySha256"]}
        payload, _ = verify_signed_receipt(attestation, trust, "candidate-lineage", required)
        for field in ("checkpointSha256", "randomInitializationRootSha256", "trainingRunManifestSha256", "codeSnapshotSha256", "dataReleaseSha256"):
            if not isinstance(payload.get(field), str) or re.fullmatch(r"[0-9a-f]{64}", payload[field]) is None:
                raise ValueError("Missing owned candidate lineage: " + field)
        from model.identity import parse_artifact_id
        if parse_artifact_id(payload["artifactId"]).generation != 2:
            raise ValueError("Wrong candidate family")
        def adapter(request):
            # Evaluation IDs, targets, grades and self-reported counters never enter the prompt.
            prompt = canonical({"context": request["context"], "history": request["history"], "message": request["message"]})
            ids = tokenizer.encode(prompt)
            budget = min(256, model.config.max_sequence_length - len(ids))
            if budget < 1:
                return {"text": "", "unavailable": "EVALUATION_CONTEXT_BUDGET_EXHAUSTED"}
            self._admitted_models[id(model)] = payload["artifactId"]
            try:
                return self._decode(model=model, input_ids=ids, decode=tokenizer.decode, request=request,
                                    max_new_tokens=budget, eos_id=tokenizer.eos_token_id)
            finally:
                self._admitted_models.pop(id(model), None)
        return adapter

    def from_supervised_runtime(self, request, mode):
        """Narrow integration contract for task 042; absent implementation stays blocked."""
        if mode not in {"raw_model", "model_tools"}:
            raise ValueError("Only native evaluation lanes may use this adapter")
        from model.runtime_service import get_model_runtime_service
        service = get_model_runtime_service()
        health = service.start()
        if not health.get("ready"):
            return {"text": "", "unavailable": health.get("code", "MODEL_UNAVAILABLE")}
        artifact_id = health.get("artifactId")
        if not isinstance(artifact_id, str) or not artifact_id.startswith("vfdlm-g2-"):
            return {"text": "", "unavailable": "GEN2_APPROVED_ARTIFACT_REQUIRED"}
        runtime = service.runtime_for_generation()
        try:
            expected_type = importlib.import_module("model.gen2.model").VoltForgeGen2
        except (ImportError, AttributeError):
            return {"text": "", "unavailable": "GEN2_EVALUATION_ADAPTER_NOT_IMPLEMENTED"}
        if type(getattr(runtime, "_model", None)) is not expected_type:
            return {"text": "", "unavailable": "UNVERIFIED_NATIVE_MODEL_CLASS"}
        encode = getattr(runtime, "encode_evaluation_request", None)
        tokenizer = getattr(runtime, "_tokenizer", None)
        if not callable(encode) or tokenizer is None:
            return {"text": "", "unavailable": "GEN2_PROMPT_ADAPTER_NOT_IMPLEMENTED"}
        # Encoding is part of the audited runtime, not a function supplied by an evaluated answer.
        # model_tools may add typed evidence to context, but cannot supply response text.
        ids = encode(request, with_tools=mode == "model_tools")
        eos_id = getattr(runtime, "evaluation_eos_token_id", None)
        if type(eos_id) is not int or sum(p.numel() for p in runtime._model.parameters()) != health.get("parameterCount"):
            return {"text": "", "unavailable": "GEN2_EVALUATION_IDENTITY_OR_TOKENIZER_MISMATCH"}
        budget = min(256, int(health["contextLength"]) - len(ids))
        if budget < 1:
            return {"text": "", "unavailable": "EVALUATION_CONTEXT_BUDGET_EXHAUSTED"}
        self._admitted_models[id(runtime._model)] = artifact_id
        try:
            return self._decode(model=runtime._model, input_ids=ids, decode=tokenizer.decode,
                                request=request, max_new_tokens=budget, eos_id=eos_id)
        finally:
            self._admitted_models.pop(id(runtime._model), None)
