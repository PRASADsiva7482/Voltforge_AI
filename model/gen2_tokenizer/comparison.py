"""Diagnostic metrics only; fixture results cannot select a production vocabulary."""
import time
from model.tokenizer import VoltForgeTokenizer
from .contract import BYTE_OFFSET, CANDIDATE_SIZES, SPECIAL_TOKENS, require, sha
from .fixtures import MEASURE, fit_fixture, lineage
from .release import AI, build_fixture, fingerprints, verify_fixture

BASELINE = AI / "model/tokenizers/vfdlm-byte-bpe-v1.1.0"


def measure(tokenizer, text):
    raw = text.encode("utf-8", "surrogatepass")
    encoded = tokenizer.encode(text)
    require(tokenizer.decode_bytes(encoded) == raw, "GEN2_COMPARISON_ROUNDTRIP_FAILED")
    iterations = 3
    start = time.perf_counter_ns()
    for _ in range(iterations):
        tokenizer.encode(text)
    encode_seconds = (time.perf_counter_ns() - start) / 1e9
    start = time.perf_counter_ns()
    for _ in range(iterations):
        tokenizer.decode_bytes(encoded)
    decode_seconds = (time.perf_counter_ns() - start) / 1e9
    return {
        "bytes": len(raw), "unicodeCodePoints": len(text), "tokens": len(encoded),
        "whitespaceUnits": len(text.split()), "tokensPerWhitespaceUnit": len(encoded) / max(1, len(text.split())),
        "tokensPerCodePoint": len(encoded) / max(1, len(text)),
        "bytesPerToken": len(raw) / max(1, len(encoded)),
        "reductionVersusByteTokens": 1 - len(encoded) / max(1, len(raw)),
        "roundtrip": True, "measurementIterations": iterations,
        "encodeSeconds": encode_seconds, "decodeSeconds": decode_seconds,
        "encodeBytesPerSecond": len(raw) * iterations / max(1e-9, encode_seconds),
        "decodeBytesPerSecond": len(raw) * iterations / max(1e-9, decode_seconds),
    }


def compare_fixtures(*, publish=True):
    baseline = VoltForgeTokenizer()
    baseline.load(BASELINE)
    require(baseline.vocab_size == 3072, "GEN2_BASELINE_CHANGED")
    baseline_metrics = {category: measure(baseline, text) for category, text in MEASURE}
    candidates = []
    for target in CANDIDATE_SIZES:
        path = build_fixture(target) if publish else None
        tokenizer, manifest = verify_fixture(path, recompute=True) if publish else (fit_fixture(target), None)
        metrics = {category: measure(tokenizer, text) for category, text in MEASURE}
        for category, item in metrics.items():
            item["tokenCountRatioTo3072Baseline"] = item["tokens"] / baseline_metrics[category]["tokens"]
        marker_costs = [{"marker": marker, "literalContentTokens": len(tokenizer.encode(marker)), "structuralTokens": 1,
                         "literalOverheadTokens": len(tokenizer.encode(marker)) - 1} for marker in SPECIAL_TOKENS]
        require(all(all(token >= BYTE_OFFSET for token in tokenizer.encode(marker)) for marker in SPECIAL_TOKENS), "GEN2_MARKER_CONTENT_BOUNDARY_FAILED")
        candidates.append({
            "targetVocabSize": target, "actualVocabSize": tokenizer.vocab_size,
            "targetReached": target == tokenizer.vocab_size, "binding": tokenizer.binding(),
            "fixtureRelease": path.relative_to(AI).as_posix() if path else None,
            "manifestSha256": sha((path / "manifest.json").read_bytes()) if path else None,
            "metrics": metrics, "literalMarkerCosts": marker_costs,
            "chatFramingCost": "BOS + 2 per message + one assistant prefix or EOS; content costs remain exact",
        })
    return {
        "schemaVersion": 1, "taskId": "LLM-TASK-024", "status": "passed-fixture-comparison-only",
        "lineage": lineage(), "sourceFingerprints": fingerprints(),
        "baseline": {"vocabSize": baseline.vocab_size, "manifestSha256": sha((BASELINE / "tokenizer_manifest.json").read_bytes()), "metrics": baseline_metrics},
        "candidates": candidates, "productionVocabularySelected": None,
        "approvedCorpusFitted": False, "tokenizerReleaseAccepted": False, "modelTrainingPerformed": False,
        "limitations": [
            "Target sizes are ceilings; these tiny fixtures do not populate 16k or 32k vocabularies.",
            "Whitespace fertility is a proxy, not a linguistic word segmentation metric.",
            "Fixture throughput is this host's short codec measurement, not server or large-corpus capacity.",
            "Baseline and fixtures have different fitting data; this is a diagnostic, not a production ranking.",
            "Real admitted-corpus comparison, release and model/runtime binding remain pending.",
        ],
    }
