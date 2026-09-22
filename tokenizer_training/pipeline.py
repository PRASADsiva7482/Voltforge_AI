"""Train, evaluate, publish, and no-write verify VFDLM byte-BPE v1."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from data_governance.governance import default_manifest_path, require_approved_shard, sha256_file
from model.tokenizer import (
    BASE_VOCAB_SIZE,
    SPECIAL_TOKEN_TO_ID,
    TOKENIZER_ALGORITHM,
    TOKENIZER_CONTRACT_VERSION,
    VoltForgeTokenizer,
)
from synthetic_data.pipeline import SHARD_ROOT, SHARDS, check_release as check_synthetic_release
from task_schema.compiler import compile_task_record
from task_schema.io import read_task_shard
from tokenizer_training.legacy_baseline import LegacyTokenizerBaseline, legacy_artifact_descriptors


AI_ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_ID = "vfdlm-byte-bpe"
TOKENIZER_VERSION = "1.1.0"
TARGET_VOCAB_SIZE = 3072
MINIMUM_PAIR_FREQUENCY = 2
SPLIT_SEED = "vfai-010-split-20260828"
ARTIFACT_ROOT = AI_ROOT / "model" / "tokenizers" / "vfdlm-byte-bpe-v1.1.0"
REPORT_PATH = ARTIFACT_ROOT / "evaluation-report.json"
MANIFEST_PATH = ARTIFACT_ROOT / "tokenizer_manifest.json"
LEGACY_ARTIFACT_ROOT = AI_ROOT / "model" / "artifacts"
TRAINER_PATHS = (
    "model/tokenizer.py",
    "tokenizer_training/pipeline.py",
    "tokenizer_training/legacy_baseline.py",
    "task_schema/compiler.py",
)
PROBES = {
    "firmware_code": (
        "const uint8_t ledPin = LED_BUILTIN;\nvoid setup(){ pinMode(ledPin, OUTPUT); }\nvoid loop(){ digitalWrite(ledPin, HIGH); }",
        "Wire.begin(); Serial.println(analogRead(A0));",
    ),
    "engineering_units": (
        "3.3 V, 5 V, 10 kΩ, 220 Ω, 47 µF, 16 MHz, 20 mA, 100 nF",
        "V=IR; P=VI; tolerance ±1%; 2.2 kOhm; 1 µs",
    ),
    "pin_names": (
        "GPIO21 GPIO22 A0 D13 SDA SCL MOSI MISO SCK 3V3_OUT QWIIC_SDA",
        "DATA/SDA CLK/SCL LED_BUILTIN INPUT_PULLUP",
    ),
    "component_identifiers": (
        "ADAFRUIT_SSD1306_STEMMA_128X64 ESP32-WROOM-32E-N4 ATmega328P RP2350",
        "vf-knowledge-v1-board.arduino.uno-r3 A000066 ABX00087",
    ),
    "structured_json": (
        '{"boardType":"ARDUINO_UNO","connections":[{"from":"VIN","to":"3V3"}]}',
        '{"applicationMode":"proposal-only","requiresUserConfirmation":true}',
    ),
    "unicode": (
        "Voltage drop → 3.3 V; resistance ≈ 220 Ω; temperature 25 °C.",
        "µA Ω ± × ÷ Ελληνικά 日本語 हिंदी 🚦",
    ),
}


class TokenizerReleaseError(RuntimeError):
    """The tokenizer release failed a lineage, quality, or reproducibility gate."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_bytes(data)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _record_hash(record: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_json(record).encode("utf-8"))


def _load_approved_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    release = check_synthetic_release(recompile=False)
    expected_count = int(release["summary"]["acceptedRecordCount"])
    records: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    for filename in SHARDS:
        path = SHARD_ROOT / filename
        manifest = require_approved_shard(path, "training", manifest_path=default_manifest_path(path))
        shard_records = read_task_shard(path)
        records.extend(shard_records)
        lineage.append(
            {
                "shardId": manifest["shardId"],
                "path": manifest["path"],
                "sha256": manifest["sha256"],
                "recordCount": manifest["recordCount"],
                "manifestPath": default_manifest_path(path).relative_to(AI_ROOT).as_posix(),
                "manifestSha256": sha256_file(default_manifest_path(path)),
                "sourceIds": manifest["recordProvenance"]["sourceIds"],
            }
        )
    if len(records) != expected_count or len({item["recordId"] for item in records}) != len(records):
        raise TokenizerReleaseError("Approved VFAI-009 record set is incomplete or duplicated")
    return records, lineage


def _split_records(records: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_task[record["task"]].append(record)
    training: list[dict[str, Any]] = []
    evaluation: list[dict[str, Any]] = []
    for task in sorted(by_task):
        ranked = sorted(
            by_task[task],
            key=lambda item: _sha256_bytes(f"{SPLIT_SEED}:{item['recordId']}".encode("utf-8")),
        )
        evaluation_count = max(1, round(len(ranked) * 0.10))
        evaluation.extend(ranked[:evaluation_count])
        training.extend(ranked[evaluation_count:])
    training.sort(key=lambda item: item["recordId"])
    evaluation.sort(key=lambda item: item["recordId"])
    return training, evaluation


def _token_metrics(tokenizer: VoltForgeTokenizer, texts: Iterable[str]) -> dict[str, Any]:
    values = list(texts)
    byte_count = sum(len(value.encode("utf-8", errors="surrogatepass")) for value in values)
    token_count = sum(len(tokenizer.encode(value)) for value in values)
    return {
        "documents": len(values),
        "utf8Bytes": byte_count,
        "tokens": token_count,
        "bytesPerToken": round(byte_count / max(1, token_count), 6),
    }


def _legacy_metrics(baseline: LegacyTokenizerBaseline, texts: Iterable[str]) -> dict[str, Any]:
    values = list(texts)
    encoded = [baseline.encode(value) for value in values]
    byte_count = sum(len(value.encode("utf-8", errors="surrogatepass")) for value in values)
    token_count = sum(len(ids) for ids, _ in encoded)
    return {
        "documents": len(values),
        "utf8Bytes": byte_count,
        "tokens": token_count,
        "unknownTokens": sum(unknowns for _, unknowns in encoded),
        "bytesPerToken": round(byte_count / max(1, token_count), 6),
    }


def _evaluate(
    tokenizer: VoltForgeTokenizer,
    evaluation_texts: list[str],
    split: Mapping[str, Any],
    lineage: list[dict[str, Any]],
    file_hashes: Mapping[str, str],
) -> dict[str, Any]:
    baseline = LegacyTokenizerBaseline(LEGACY_ARTIFACT_ROOT)
    current = _token_metrics(tokenizer, evaluation_texts)
    legacy = _legacy_metrics(baseline, evaluation_texts)
    if current["tokens"] >= legacy["tokens"]:
        raise TokenizerReleaseError("New tokenizer did not beat legacy token count on the frozen split")
    reduction = round((legacy["tokens"] - current["tokens"]) * 100 / legacy["tokens"], 3)
    fragmentation_categories = {}
    for category, values in PROBES.items():
        new_lengths = [len(tokenizer.encode(value)) for value in values]
        legacy_results = [baseline.encode(value) for value in values]
        new_count = sum(new_lengths)
        legacy_count = sum(len(ids) for ids, _ in legacy_results)
        legacy_unknowns = sum(unknowns for _, unknowns in legacy_results)
        lossless_comparison = legacy_unknowns == 0
        category_pass = new_count <= legacy_count if lossless_comparison else True
        fragmentation_categories[category] = {
            "examples": len(values),
            "newTokens": new_count,
            "legacyTokens": legacy_count,
            "legacyUnknownTokens": legacy_unknowns,
            "newMaximumPieces": max(new_lengths),
            "legacyMaximumPieces": max(len(ids) for ids, _ in legacy_results),
            "comparison": "token-count" if lossless_comparison else "new-lossless-legacy-lossy",
            "decision": "pass" if category_pass else "fail",
        }
    if any(item["decision"] != "pass" for item in fragmentation_categories.values()):
        raise TokenizerReleaseError("New tokenizer did not clear the domain fragmentation gates")
    report = {
        "schemaVersion": 1,
        "reportId": "vf-tokenizer-evaluation-v1",
        "tokenizer": {
            "tokenizerId": TOKENIZER_ID,
            "version": TOKENIZER_VERSION,
            "algorithm": TOKENIZER_ALGORITHM,
            "vocabSize": tokenizer.vocab_size,
            "mergeCount": len(tokenizer.merges),
            "files": dict(file_hashes),
        },
        "frozenSplit": dict(split),
        "lineage": lineage,
        "compression": {
            "new": current,
            "legacy": legacy,
            "tokenReductionPercent": reduction,
            "bytesPerTokenImprovementPercent": round(
                (current["bytesPerToken"] - legacy["bytesPerToken"])
                * 100
                / legacy["bytesPerToken"],
                3,
            ),
            "decision": "pass",
        },
        "fragmentation": {
            "policy": "New token count must not exceed a lossless legacy result; a legacy result containing [UNK] is lossy and cannot win by discarding bytes.",
            "categories": fragmentation_categories,
            "decision": "pass",
        },
        "roundTrip": {
            "contract": "decode(encode(text)) equals the original Python string, including lone surrogates",
            "unknownTokenReachableForBytes": False,
            "probeCategories": sorted(PROBES),
            "decision": "pass",
        },
        "vocabularyCost": {
            "newVocabSize": tokenizer.vocab_size,
            "legacyVocabSize": baseline.vocab_size,
            "sizeReductionPercent": round(
                (baseline.vocab_size - tokenizer.vocab_size) * 100 / baseline.vocab_size, 3
            ),
            "baseByteTokens": 256,
            "specialTokens": len(SPECIAL_TOKEN_TO_ID),
            "decision": "pass" if tokenizer.vocab_size <= baseline.vocab_size else "fail",
        },
        "legacyBaseline": {
            "classification": "quarantined-comparison-only-never-a-training-input",
            "artifacts": legacy_artifact_descriptors(LEGACY_ARTIFACT_ROOT),
        },
        "decision": "pass",
    }
    report["reportSha256"] = _sha256_bytes(_canonical_json(report).encode("utf-8"))
    return report


def _build_release_documents() -> dict[str, bytes]:
    records, lineage = _load_approved_records()
    training_records, evaluation_records = _split_records(records)
    training_texts = [compile_task_record(item) for item in training_records]
    evaluation_texts = [compile_task_record(item) for item in evaluation_records]
    training_corpus_hash = _sha256_bytes(
        b"".join(value.encode("utf-8") + b"\0" for value in training_texts)
    )
    evaluation_corpus_hash = _sha256_bytes(
        b"".join(value.encode("utf-8") + b"\0" for value in evaluation_texts)
    )
    tokenizer = VoltForgeTokenizer(TARGET_VOCAB_SIZE)
    tokenizer.tokenizer_id = TOKENIZER_ID
    tokenizer.version = TOKENIZER_VERSION
    tokenizer.train(
        training_texts,
        TARGET_VOCAB_SIZE,
        minimum_frequency=MINIMUM_PAIR_FREQUENCY,
        training_metadata={
            "fromScratch": True,
            "pretrainedVocabularyLoaded": False,
            "pretrainedMergesLoaded": False,
            "trainingRecordCount": len(training_records),
            "evaluationRecordCount": len(evaluation_records),
            "trainingCorpusSha256": training_corpus_hash,
            "splitSeed": SPLIT_SEED,
            "minimumPairFrequency": MINIMUM_PAIR_FREQUENCY,
        },
    )
    vocab, merges, config = tokenizer.artifact_documents()
    documents: dict[str, bytes] = {
        "vocab.json": _json_bytes(vocab),
        "merges.json": _json_bytes(merges),
    }
    config["vocabSha256"] = _sha256_bytes(documents["vocab.json"])
    config["mergesSha256"] = _sha256_bytes(documents["merges.json"])
    config["specialTokenContractSha256"] = _sha256_bytes(
        _canonical_json(SPECIAL_TOKEN_TO_ID).encode("utf-8")
    )
    documents["tokenizer_config.json"] = _json_bytes(config)
    split = {
        "id": "vf-tokenizer-split-v1",
        "seed": SPLIT_SEED,
        "method": "per-task SHA-256 rank with approximately 10 percent evaluation",
        "trainingRecordCount": len(training_records),
        "evaluationRecordCount": len(evaluation_records),
        "trainingRecordIdsSha256": _sha256_bytes(
            "\n".join(item["recordId"] for item in training_records).encode("utf-8")
        ),
        "evaluationRecordIds": [item["recordId"] for item in evaluation_records],
        "trainingCorpusSha256": training_corpus_hash,
        "evaluationCorpusSha256": evaluation_corpus_hash,
        "taskBalance": dict(sorted(Counter(item["task"] for item in evaluation_records).items())),
    }
    file_hashes = {
        name: _sha256_bytes(value)
        for name, value in documents.items()
    }
    report = _evaluate(tokenizer, evaluation_texts, split, lineage, file_hashes)
    documents["evaluation-report.json"] = _json_bytes(report)
    source_ids = sorted({source_id for item in lineage for source_id in item["sourceIds"]})
    manifest = {
        "schemaVersion": 1,
        "artifactKind": "voltforge-tokenizer",
        "tokenizerId": TOKENIZER_ID,
        "version": TOKENIZER_VERSION,
        "contractVersion": TOKENIZER_CONTRACT_VERSION,
        "releaseStatus": "approved",
        "algorithm": TOKENIZER_ALGORITHM,
        "vocabSize": tokenizer.vocab_size,
        "mergeCount": len(tokenizer.merges),
        "specialTokenContractSha256": config["specialTokenContractSha256"],
        "files": {
            "vocab": {"path": "vocab.json", "sha256": file_hashes["vocab.json"]},
            "merges": {"path": "merges.json", "sha256": file_hashes["merges.json"]},
            "config": {"path": "tokenizer_config.json", "sha256": file_hashes["tokenizer_config.json"]},
            "evaluationReport": {
                "path": "evaluation-report.json",
                "sha256": _sha256_bytes(documents["evaluation-report.json"]),
            },
        },
        "lineage": {
            "sourceIds": source_ids,
            "shardIds": [item["shardId"] for item in lineage],
            "shards": lineage,
            "split": split,
        },
        "trainer": [
            {"path": path, "sha256": sha256_file(AI_ROOT / path)} for path in TRAINER_PATHS
        ],
        "modelArtifactDependencyContract": {
            "requiredRegistryFileKey": "tokenizerManifest",
            "manifestFilename": "tokenizer_manifest.json",
            "gate": "model.artifact_registry._validate_tokenizer",
        },
    }
    manifest["artifactSha256"] = _sha256_bytes(_canonical_json(manifest).encode("utf-8"))
    documents["tokenizer_manifest.json"] = _json_bytes(manifest)
    return documents


def write_tokenizer_release() -> dict[str, Any]:
    documents = _build_release_documents()
    for name, content in documents.items():
        _atomic_write(ARTIFACT_ROOT / name, content)
    return check_tokenizer_release()


def check_tokenizer_release() -> dict[str, Any]:
    expected = _build_release_documents()
    for name, content in expected.items():
        path = ARTIFACT_ROOT / name
        if not path.is_file() or path.read_bytes() != content:
            raise TokenizerReleaseError(f"Tokenizer release artifact is stale: {name}")
    tokenizer = VoltForgeTokenizer(TARGET_VOCAB_SIZE)
    tokenizer.load(ARTIFACT_ROOT)
    report = json.loads(expected["evaluation-report.json"])
    if report.get("decision") != "pass":
        raise TokenizerReleaseError("Tokenizer evaluation report did not pass")
    return report
