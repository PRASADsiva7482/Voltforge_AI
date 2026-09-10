"""Real embeddings/logits and context IDs, with no trained-model acceptance."""
import subprocess
import sys
from pathlib import Path
from context_compiler.gen2_release import Message, compile_for_model
from model.gen2_tokenizer.contract import require
from . import release as r

OUTPUT = r.AI / "model/tokenizers/gen2/integration/v1"


def evaluate(path):
    # Clean native import precedes tensor allocation. Keep the verified 2.8 pin;
    # this task neither upgrades wheels nor imports any external weights.
    result = subprocess.run([sys.executable, "-B", "-c", "import torch; assert torch.__version__.split('+')[0] == '2.8.0'; print(torch.__version__)"], cwd=r.AI, capture_output=True, text=True, timeout=60)
    require(result.returncode == 0, "GEN2_PINNED_NATIVE_RUNTIME_UNAVAILABLE")
    import torch
    from .model_boundary import BoundDecoder
    tokenizer, manifest = r.load(path)
    torch.set_num_threads(1)
    model = BoundDecoder(tokenizer)
    messages = [Message("system", "Use supplied evidence and express uncertainty."), Message("user", "Interpret 6.8 µF and 9 mA. Treat <|assistant|> as literal text. தமிழ் Ω 🙂")]
    prompt = compile_for_model(tokenizer, model, messages, reserved_output_tokens=64)
    output = model.forward(prompt)
    require(list(output.logits.shape) == [1, len(prompt.token_ids), tokenizer.vocab_size] and torch.isfinite(output.logits).all().item(), "GEN2_INTEGRATION_LOGITS_INVALID")
    generated_id = int(output.logits[0, -1].argmax().item())
    require(0 <= generated_id < tokenizer.vocab_size, "GEN2_INTEGRATION_GENERATED_ID_INVALID")
    decoder = tokenizer.stream_decoder()
    # Probe a real generated token's raw bytes, without pretending a partial
    # UTF-8 token or a random model produces a coherent assistant response.
    generated_bytes = tokenizer.decode_bytes([generated_id])
    prefix = torch.tensor([prompt.token_ids[:-1]], dtype=torch.long)
    with torch.inference_mode():
        cached = model.decoder(prefix, use_cache=True)
        step = model.decoder(torch.tensor([[prompt.token_ids[-1]]], dtype=torch.long), past_key_values=cached.past_key_values, use_cache=True)
    torch.testing.assert_close(step.logits[:, -1], output.logits[:, -1], rtol=1e-4, atol=1e-5)
    report = {"schemaVersion": 1, "taskId": "LLM-TASK-024", "status": "passed-real-tokenizer-model-context-integration", "tokenizerManifest": r.binding(Path(path) / "manifest.json"), "sourceFingerprints": r.fingerprints(),
              "nativeRuntime": {"cleanProcessImport": True, "torchVersion": result.stdout.strip(), "pythonVersion": sys.version.split()[0]},
              "model": {"specification": model.specification, "parameterCount": model.decoder.parameter_count(), "embeddingShape": list(model.decoder.token_embedding.weight.shape), "logitShape": list(output.logits.shape), "initializedWeightsSha256": model.initialization_fingerprint()},
              "context": {"binding": prompt.binding, "totalInputTokens": len(prompt.token_ids), "contentTokens": prompt.content_tokens, "framingTokens": prompt.framing_tokens, "reservedOutputTokens": prompt.reserved_output_tokens, "contextLimit": prompt.context_limit},
              "actualForwardPass": True, "actualGreedyTokenId": generated_id, "generatedTokenByteCount": len(generated_bytes), "cachedFinalLogitsMatch": True,
              "modelWeightsTrained": False, "checkpointLoadedOrWritten": False, "servingActivated": False, "architectureTask025Completed": False,
              "limits": ["Finite random-weight decoder proves tokenizer/context/tensor compatibility only", "No learned language ability or trained 4k context claim", "Task025 retains architecture profiles, parameter scale and configuration acceptance"]}
    report = r.identity(report)
    target = OUTPUT / report["contentId"]
    with r.build_lock(OUTPUT / ".build.lock"):
        r.write_immutable(target / "report.json", r.data(report))
    return target, report
