from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from model.tokenizer import (
    BASE_VOCAB_SIZE,
    SPECIAL_TOKEN_TO_ID,
    TOKENIZER_ALGORITHM,
    TokenizerContractError,
    VoltForgeTokenizer,
)
from tokenizer_training.pipeline import (
    ARTIFACT_ROOT,
    PROBES,
    TARGET_VOCAB_SIZE,
    check_tokenizer_release,
)


def released_tokenizer() -> VoltForgeTokenizer:
    tokenizer = VoltForgeTokenizer(TARGET_VOCAB_SIZE)
    tokenizer.load(ARTIFACT_ROOT)
    return tokenizer


def test_fixed_special_contract_and_complete_byte_alphabet() -> None:
    tokenizer = released_tokenizer()
    assert tokenizer.special_token_to_id == SPECIAL_TOKEN_TO_ID
    assert SPECIAL_TOKEN_TO_ID["[PAD]"] == 0
    assert SPECIAL_TOKEN_TO_ID["[UNK]"] == 1
    assert SPECIAL_TOKEN_TO_ID["[BOS]"] == 2
    assert SPECIAL_TOKEN_TO_ID["[EOS]"] == 3
    all_bytes = bytes(range(256))
    encoded = tokenizer.encode_bytes(all_bytes)
    assert SPECIAL_TOKEN_TO_ID["[UNK]"] not in encoded
    assert tokenizer.decode_bytes(encoded) == all_bytes
    assert len(tokenizer.vocab) == TARGET_VOCAB_SIZE
    assert len(tokenizer.merges) == TARGET_VOCAB_SIZE - BASE_VOCAB_SIZE


@pytest.mark.parametrize(
    "value",
    [
        "const uint8_t pin = GPIO21; digitalWrite(pin, HIGH);",
        "3.3 V | 10 kΩ | 47 µF | 16 MHz | ±1% | 20 mA",
        "ADAFRUIT_SSD1306_STEMMA_128X64 / ESP32-WROOM-32E-N4 / A000066",
        '{"board":"ARDUINO_UNO","pins":["A0","SDA","SCL"],"safe":true}',
        "Voltage → current; Ω µA °C Ελληνικά 日本語 हिंदी 🚦",
        "literal [BOS] text is ordinary input unless special parsing is explicitly enabled",
        "\ud800 lone-high \udfff lone-low",
    ],
)
def test_lossless_round_trip_for_domain_json_unicode_and_malformed_unicode(value: str) -> None:
    tokenizer = released_tokenizer()
    assert tokenizer.decode(tokenizer.encode(value)) == value
    assert SPECIAL_TOKEN_TO_ID["[UNK]"] not in tokenizer.encode(value)


def test_special_parsing_is_explicit_and_bos_eos_are_stable() -> None:
    tokenizer = released_tokenizer()
    literal = tokenizer.encode("[BOS]")
    assert literal != [SPECIAL_TOKEN_TO_ID["[BOS]"]]
    assert tokenizer.decode(literal) == "[BOS]"
    assert tokenizer.encode("[BOS]", allowed_special=True) == [SPECIAL_TOKEN_TO_ID["[BOS]"]]
    framed = tokenizer.encode("GPIO21", add_bos=True, add_eos=True)
    assert framed[0] == 2 and framed[-1] == 3
    assert tokenizer.decode(framed) == "GPIO21"


def test_training_is_deterministic_and_does_not_seed_domain_vocabulary() -> None:
    documents = [
        "digitalWrite(GPIO21, HIGH); digitalWrite(GPIO21, LOW);",
        "GPIO21 SDA SCL 3.3 V 10 kOhm",
        '{"pin":"GPIO21","mode":"OUTPUT"}',
    ]
    first = VoltForgeTokenizer(512)
    second = VoltForgeTokenizer(512)
    metadata = {"pretrainedVocabularyLoaded": False, "pretrainedMergesLoaded": False}
    first.train(documents, 512, training_metadata=metadata)
    second.train(list(reversed(documents)), 512, training_metadata=metadata)
    assert first.artifact_documents() == second.artifact_documents()
    assert first.training_metadata == metadata


def test_release_retrains_exactly_and_beats_legacy_with_lower_vocab_cost() -> None:
    report = check_tokenizer_release()
    assert report["decision"] == "pass"
    assert report["tokenizer"]["algorithm"] == TOKENIZER_ALGORITHM
    assert report["tokenizer"]["vocabSize"] == TARGET_VOCAB_SIZE
    assert report["compression"]["tokenReductionPercent"] >= 70.0
    assert report["compression"]["new"]["tokens"] < report["compression"]["legacy"]["tokens"]
    assert report["vocabularyCost"]["newVocabSize"] < report["vocabularyCost"]["legacyVocabSize"]
    assert report["vocabularyCost"]["decision"] == "pass"
    assert report["fragmentation"]["decision"] == "pass"
    assert set(report["fragmentation"]["categories"]) == set(PROBES)
    assert all(
        item["decision"] == "pass"
        for item in report["fragmentation"]["categories"].values()
    )
    assert report["legacyBaseline"]["classification"] == (
        "quarantined-comparison-only-never-a-training-input"
    )


def test_manifest_binds_vocab_merges_config_report_lineage_and_trainer() -> None:
    manifest = json.loads(
        (ARTIFACT_ROOT / "tokenizer_manifest.json").read_text(encoding="utf-8")
    )
    config = json.loads(
        (ARTIFACT_ROOT / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    assert manifest["releaseStatus"] == "approved"
    assert manifest["vocabSize"] == TARGET_VOCAB_SIZE
    assert set(manifest["files"]) == {"vocab", "merges", "config", "evaluationReport"}
    assert len(manifest["lineage"]["sourceIds"]) == 2
    assert len(manifest["lineage"]["shardIds"]) == 4
    assert len(manifest["trainer"]) == 4
    assert config["training"]["fromScratch"] is True
    assert config["training"]["pretrainedVocabularyLoaded"] is False
    assert config["training"]["pretrainedMergesLoaded"] is False
    assert config["unknownTokenReachableForBytes"] is False


def test_tampered_tokenizer_file_fails_closed(tmp_path: Path) -> None:
    copied = tmp_path / "tokenizer"
    shutil.copytree(ARTIFACT_ROOT, copied)
    vocab_path = copied / "vocab.json"
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    changed = deepcopy(vocab)
    changed["hex:00"], changed["hex:01"] = changed["hex:01"], changed["hex:00"]
    vocab_path.write_text(json.dumps(changed), encoding="utf-8")
    tokenizer = VoltForgeTokenizer(TARGET_VOCAB_SIZE)
    with pytest.raises(TokenizerContractError) as raised:
        tokenizer.load(copied)
    assert raised.value.code == "TOKENIZER_FILE_CHECKSUM_MISMATCH"
