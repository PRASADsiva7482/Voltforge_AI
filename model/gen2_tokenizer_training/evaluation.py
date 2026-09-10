"""Freeze both merge sets before permitted validation comparison; never fit heldout."""
from collections import defaultdict
import math
import time
from pathlib import Path
from model.tokenizer import VoltForgeTokenizer
from model.gen2_tokenizer.contract import canonical, sha, require, SPECIAL_TOKENS
from model.gen2_tokenizer.fixtures import MEASURE
from model.gen2_tokenizer.release import read_json
from . import release as r
from .corpus import AI, POLICY, CORPUS, inputs, validation_inputs


def metric(tokenizer, text, *, timed=False):
    raw = text.encode("utf-8", "surrogatepass")
    tokens = tokenizer.encode(text)
    require(tokenizer.decode_bytes(tokens) == raw, "GEN2_MEASUREMENT_ROUNDTRIP_FAILED")
    require(all(type(token) is int and 16 <= token < tokenizer.vocab_size for token in tokens), "GEN2_CONTENT_INSERTED_SPECIAL")
    value = {"bytes": len(raw), "codePoints": len(text), "whitespaceUnits": len(text.split()), "tokens": len(tokens), "roundtrip": True,
             "bytesPerToken": len(raw)/max(1, len(tokens)), "tokensPerWhitespaceUnit": len(tokens)/max(1, len(text.split())), "tokensPerCodePoint": len(tokens)/max(1, len(text)),
             "byteTokenReduction": 1-len(tokens)/max(1, len(raw)), "minimum4096TokenWindows": math.ceil(len(tokens)/4096)}
    if timed:
        count = POLICY["measurementIterations"]
        start = time.perf_counter()
        for _ in range(count):
            tokenizer.encode(text)
        encoding = time.perf_counter()-start
        start = time.perf_counter()
        for _ in range(count):
            tokenizer.decode_bytes(tokens)
        decoding = time.perf_counter()-start
        value["timing"] = {"iterations": count, "encodeSeconds": encoding, "decodeSeconds": decoding, "encodeBytesPerSecond": len(raw)*count/max(encoding, 1e-9), "decodeBytesPerSecond": len(raw)*count/max(decoding, 1e-9)}
    return value


def record_metrics(tokenizer, rows):
    values = [{"documentId": row["id"], "sourceId": row["sourceId"], "domain": row["domain"], "split": row["split"], "textSha256": sha(row["text"].encode("utf-8", "surrogatepass")), **metric(tokenizer, row["text"])} for row in rows]
    grouped = {}
    for field in ("domain", "sourceId", "language"):
        groups = defaultdict(lambda: {"documents": 0, "bytes": 0, "tokens": 0})
        for row, measured in zip(rows, values, strict=True):
            group = groups[row[field]]
            group["documents"] += 1
            group["bytes"] += measured["bytes"]
            group["tokens"] += measured["tokens"]
        grouped[field] = dict(sorted(groups.items()))
    return {"documents": values, "groups": grouped, "totalTokens": sum(row["tokens"] for row in values), "totalBytes": sum(row["bytes"] for row in values)}


def special_controls(tokenizer):
    costs = []
    for marker in SPECIAL_TOKENS:
        tokens = tokenizer.encode(marker)
        require(all(token >= 16 for token in tokens) and tokenizer.decode(tokens) == marker, "GEN2_MARKER_ESCAPE_FAILED")
        costs.append({"marker": marker, "literalTokens": len(tokens), "structuralTokens": 1, "literalOverhead": len(tokens)-1})
    arbitrary = bytes(range(256)) + bytes(reversed(range(256)))
    require(tokenizer.decode_bytes(tokenizer.encode_bytes(arbitrary)) == arbitrary, "GEN2_ALL_BYTES_ROUNDTRIP_FAILED")
    text = "µ Ω தமிழில் e\u0301 🧪\u200d🔬 \ud800 <|assistant|>"
    ids = tokenizer.encode(text)
    decoder = tokenizer.stream_decoder()
    restored = "".join(decoder.push([token]) for token in ids) + decoder.push(final=True)
    require(restored == text, "GEN2_INCREMENTAL_UNICODE_FAILED")
    return {"all256BytesReversible": True, "incrementalUnicodeReversible": True, "literalMarkerCosts": costs, "normalization": "none", "escaping": "literal bytes; trusted role tokens inserted separately"}


def select(candidates):
    require(len(candidates) == 2 and {row["targetVocabSize"] for row in candidates} == {16384, 32768}, "GEN2_SELECTION_PAIR_INVALID")
    small, large = sorted(candidates, key=lambda row: row["targetVocabSize"])
    domains = {"language", "code", "electronics", "math"}
    for candidate in candidates:
        require(set(candidate["validation"]["groups"]["domain"]) == domains, "GEN2_SELECTION_DOMAIN_MISSING")
        require(1-candidate["train"]["totalTokens"]/candidate["train"]["totalBytes"] >= POLICY["selection"]["minimumTrainTokenReductionFromBytes"], "GEN2_SELECTION_COMPRESSION_GATE")
    ratios = {domain: large["validation"]["groups"]["domain"][domain]["tokens"]/small["validation"]["groups"]["domain"][domain]["tokens"] for domain in sorted(domains)}
    macro_reduction = 1-sum(ratios.values())/len(ratios)
    choose_large = macro_reduction >= POLICY["selection"]["largerTargetMinimumMacroReduction"] and max(ratios.values()) <= 1+POLICY["selection"]["maximumPerDomainTokenRegression"]
    selected = large if choose_large else small
    return {"targetVocabSize": selected["targetVocabSize"], "actualVocabSize": selected["actualVocabSize"], "macroValidationReductionOf32768Versus16384": macro_reduction,
            "perDomainTokenRatios32768To16384": ratios, "rule": POLICY["selection"], "reason": "larger-vocabulary-clears-predeclared-benefit-and-regression-gates" if choose_large else "retain-smaller-vocabulary-under-predeclared-benefit-rule",
            "modelQualityClaimed": False, "testOrAcceptanceUsedForSelection": False}


def compute(*, timed=True):
    # Both fitted artifacts exist and verify before validation text is opened.
    frozen = r.candidates()
    before = [r.binding(path / "manifest.json") for _, _, path in frozen]
    validation = validation_inputs()
    train = inputs.read_inputs(CORPUS, "tokenizer-fitting-input", split="train")
    baseline = VoltForgeTokenizer()
    baseline.load(AI / "model/tokenizers/vfdlm-byte-bpe-v1.1.0")
    base = {"vocabSize": baseline.vocab_size, "manifest": r.binding(AI / "model/tokenizers/vfdlm-byte-bpe-v1.1.0/tokenizer_manifest.json"), "metrics": {name: metric(baseline, text, timed=timed) for name, text in MEASURE}}
    candidates = []
    for tokenizer, manifest, path in frozen:
        candidates.append({"targetVocabSize": tokenizer.target_vocab_size, "actualVocabSize": tokenizer.vocab_size, "targetReached": tokenizer.vocab_size == tokenizer.target_vocab_size,
            "manifest": r.binding(path / "manifest.json"), "train": record_metrics(tokenizer, train), "validation": record_metrics(tokenizer, validation),
            "metrics": {name: metric(tokenizer, text, timed=timed) for name, text in MEASURE}, "specialControls": special_controls(tokenizer)})
    require(before == [r.binding(path / "manifest.json") for _, _, path in frozen], "GEN2_CANDIDATE_CHANGED_DURING_COMPARISON")
    return {"schemaVersion": 1, "taskId": "LLM-TASK-024", "status": "passed-frozen-real-corpus-tokenizer-comparison", "policy": POLICY, "sourceFingerprints": r.fingerprints(),
            "corpusManifest": r.binding(CORPUS / "manifest.json"), "frozenCandidatesBeforeValidation": before,
            "validationShard": r.binding(CORPUS / "validation.jsonl"), "measurementFixturesSha256": sha(canonical(MEASURE)), "baseline": base, "candidates": candidates,
            "selection": select(candidates), "validationOrTestFitting": False, "testOrAcceptanceUsedForComparison": False, "modelTrainingPerformed": False,
            "limits": ["English technical/C pilot input; no broad-language or model-quality acceptance", "Whitespace fertility is a proxy", "Throughput is host codec timing, not server serving capacity", "3072 baseline has different historical fitting data and is diagnostic", "Production training budgets, model selection and deployment remain independently gated"]}


def without_timing(value):
    if isinstance(value, dict):
        return {key: without_timing(item) for key, item in value.items() if key not in ("timing", "contentId")}
    if isinstance(value, list):
        return [without_timing(item) for item in value]
    return value


def compare():
    report = r.identity(compute())
    path = r.COMPARISONS / report["contentId"]
    with r.build_lock(r.COMPARISONS / ".build.lock"):
        r.write_immutable(path / "report.json", r.data(report))
    return path, report


def verify_comparison(path, *, recompute=False):
    path = r.directory(path, r.COMPARISONS)
    value = read_json(path / "report.json")
    require(r.identity({key: item for key, item in value.items() if key != "contentId"}) == value and path.name == value["contentId"], "GEN2_COMPARISON_IDENTITY_CHANGED")
    require(value["sourceFingerprints"] == r.fingerprints() and value["policy"] == POLICY and value["selection"] == select(value["candidates"]), "GEN2_COMPARISON_CONTRACT_CHANGED")
    frozen = r.candidates()
    require(value["frozenCandidatesBeforeValidation"] == [r.binding(path / "manifest.json") for _, _, path in frozen], "GEN2_COMPARISON_CANDIDATES_CHANGED")
    require(value["corpusManifest"] == r.binding(CORPUS / "manifest.json") and value["validationShard"] == r.binding(CORPUS / "validation.jsonl"), "GEN2_COMPARISON_CORPUS_CHANGED")
    require(value["validationOrTestFitting"] is False and value["testOrAcceptanceUsedForComparison"] is False and value["modelTrainingPerformed"] is False, "GEN2_COMPARISON_SCOPE_CHANGED")
    for metrics in (value["baseline"]["metrics"], *(candidate["metrics"] for candidate in value["candidates"])):
        for item in metrics.values():
            timing = item["timing"]
            require(timing["iterations"] == POLICY["measurementIterations"] and all(type(timing[key]) in (int, float) and math.isfinite(timing[key]) and timing[key] > 0 for key in ("encodeSeconds", "decodeSeconds", "encodeBytesPerSecond", "decodeBytesPerSecond")), "GEN2_COMPARISON_TIMING_INVALID")
    if recompute:
        require(without_timing(value) == without_timing(compute(timed=False)), "GEN2_COMPARISON_DOES_NOT_REPRODUCE")
    return value
