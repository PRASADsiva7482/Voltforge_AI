from __future__ import annotations

from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
import json

import pytest

try:
    installed_torch = version("torch")
except PackageNotFoundError:
    pytest.skip("Gen1 pinned PyTorch runtime is not installed", allow_module_level=True)
if installed_torch.partition("+")[0] != "2.8.0":
    pytest.skip(
        f"Gen1 training tests require pinned PyTorch 2.8.0, found {installed_torch}",
        allow_module_level=True,
    )

import torch

from gen1_training import (
    Gen1Trainer,
    Gen1TrainingConfig,
    PackedBatchStream,
    PackedCorpus,
    TrainingCheckpointError,
    TrainingRunError,
    load_approved_corpus,
    pack_token_sequences,
)
from model.gen1 import Gen1Config


def model_config(*, dropout: float = 0.1) -> Gen1Config:
    return Gen1Config(
        vocab_size=16,
        max_sequence_length=8,
        d_model=16,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        d_ff=32,
        dropout=dropout,
        seed=13,
    )


def toy_corpus() -> PackedCorpus:
    training = [
        [2, 4, 5, 6, 7, 3],
        [2, 4, 5, 8, 9, 3],
        [2, 10, 11, 6, 7, 3],
        [2, 10, 11, 8, 9, 3],
    ] * 3
    validation = [[2, 4, 5, 6, 7, 3], [2, 10, 11, 8, 9, 3]]
    return PackedCorpus.for_testing(
        training_sequences=training,
        validation_sequences=validation,
        block_size=8,
    )


def training_config(*, max_steps: int = 4, dropout: float = 0.1) -> Gen1TrainingConfig:
    return Gen1TrainingConfig(
        model=model_config(dropout=dropout),
        seed=101,
        max_steps=max_steps,
        micro_batch_size=2,
        gradient_accumulation_steps=2,
        peak_learning_rate=0.02,
        minimum_learning_rate=0.002,
        warmup_steps=1,
        weight_decay=0.0,
        max_gradient_norm=1.0,
        validation_interval=2,
        checkpoint_interval=2,
        precision="auto",
        device="cpu",
    )


def assert_nested_equal(first, second) -> None:
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        assert set(first) == set(second)
        for key in first:
            assert_nested_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for left, right in zip(first, second, strict=True):
            assert_nested_equal(left, right)
    else:
        assert first == second


class OverflowOnceGradScaler:
    """CPU-safe GradScaler double that skips exactly one optimizer update."""

    def __init__(self, *args, **kwargs) -> None:
        self._scale = 65_536.0
        self._overflow_remaining = 1
        self._skipped = False

    def is_enabled(self) -> bool:
        return True

    def scale(self, value):
        return value

    def unscale_(self, optimizer) -> None:
        return None

    def step(self, optimizer) -> None:
        if self._overflow_remaining:
            self._overflow_remaining -= 1
            self._skipped = True
            return None
        self._skipped = False
        optimizer.step()
        return None

    def update(self) -> None:
        if self._skipped:
            self._scale *= 0.5

    def get_scale(self) -> float:
        return self._scale

    def state_dict(self) -> dict[str, object]:
        return {
            "scale": self._scale,
            "overflowRemaining": self._overflow_remaining,
        }

    def load_state_dict(self, state) -> None:
        self._scale = float(state["scale"])
        self._overflow_remaining = int(state["overflowRemaining"])
        self._skipped = False


def test_packing_preserves_every_next_token_transition_exactly_once() -> None:
    sequences = [[2, 4, 5, 3], [2, 6, 3]]
    packed = pack_token_sequences(sequences, block_size=4, pad_token_id=0)
    assert packed.token_count == 7
    assert packed.predicted_token_count == 6
    assert packed.block_count == 2
    targets = []
    for labels in packed.labels:
        targets.extend(value for value in labels[1:].tolist() if value != -100)
    assert targets == [4, 5, 3, 2, 6, 3]


def test_batch_cursor_state_reproduces_future_batches_exactly() -> None:
    corpus = toy_corpus()
    first = PackedBatchStream(corpus.train, batch_size=3, seed=77)
    for _ in range(4):
        first.next_batch(torch.device("cpu"))
    state = first.state_dict()
    resumed = PackedBatchStream(corpus.train, batch_size=3, seed=77)
    resumed.load_state_dict(state)
    for _ in range(6):
        expected = first.next_batch(torch.device("cpu"))
        actual = resumed.next_batch(torch.device("cpu"))
        for expected_tensor, actual_tensor in zip(expected, actual, strict=True):
            torch.testing.assert_close(expected_tensor, actual_tensor, rtol=0, atol=0)


def test_training_config_round_trip_and_schedule_are_deterministic() -> None:
    config = training_config(max_steps=5)
    assert Gen1TrainingConfig.from_dict(config.to_dict()) == config
    assert config.fingerprint() == Gen1TrainingConfig.from_dict(config.to_dict()).fingerprint()
    rates = [config.learning_rate_for_step(index) for index in range(config.max_steps)]
    assert rates[0] == config.peak_learning_rate
    assert rates[-1] == pytest.approx(config.minimum_learning_rate)
    assert all(left >= right for left, right in zip(rates, rates[1:]))


def test_approved_loader_revalidates_frozen_split_lineage_and_tokenizer() -> None:
    corpus = load_approved_corpus(Gen1Config())
    assert corpus.corpus_kind == "approved-vfai009"
    assert corpus.manifest["approvalStatus"] == "approved"
    assert corpus.manifest["trainingRecordCount"] == 227
    assert corpus.manifest["validationRecordCount"] == 23
    assert len(corpus.manifest["shards"]) == 4
    assert corpus.manifest["tokenizer"]["tokenizerId"] == "vfdlm-byte-bpe"
    assert len(corpus.manifest["tokenizer"]["artifactSha256"]) == 64
    assert corpus.train.predicted_token_count == corpus.train.token_count - 1
    assert corpus.validation.predicted_token_count == corpus.validation.token_count - 1


def test_training_uses_real_loss_updates_parameters_and_writes_complete_manifest(
    tmp_path,
) -> None:
    config = training_config(max_steps=10, dropout=0.0)
    corpus = toy_corpus()
    trainer = Gen1Trainer.create(
        config,
        corpus,
        tmp_path / "real-run",
        run_id="real-run",
        allow_test_corpus=True,
    )
    before = {name: value.detach().clone() for name, value in trainer.model.state_dict().items()}
    initial_validation = trainer.validate()["validationLoss"]
    result = trainer.train()
    final_validation = trainer.validate()["validationLoss"]

    assert result["status"] == "completed"
    assert result["globalStep"] == 10
    assert final_validation < initial_validation
    assert any(
        not torch.equal(before[name], value)
        for name, value in trainer.model.state_dict().items()
    )
    assert all(metric["trainingLoss"] > 0 for metric in trainer.metrics)
    manifest = json.loads(
        (tmp_path / "real-run" / "run-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "completed"
    assert manifest["trainingConfigFingerprint"] == config.fingerprint()
    assert manifest["datasetFingerprint"] == corpus.fingerprint
    assert len(manifest["codeRevision"]["sourceFingerprint"]) == 64
    assert manifest["hardware"]["precision"] == "float32"
    assert manifest["metrics"][-1]["validationLoss"] == trainer.metrics[-1]["validationLoss"]
    assert manifest["checkpoints"][-1]["globalStep"] == 10
    assert len(manifest["checkpoints"][-1]["modelWeightsSha256"]) == 64
    assert len(manifest["manifestSha256"]) == 64


def test_interrupted_resume_matches_uninterrupted_model_optimizer_cursor_and_rng(
    tmp_path,
) -> None:
    config = training_config(max_steps=4, dropout=0.1)
    corpus = toy_corpus()

    uninterrupted = Gen1Trainer.create(
        config,
        corpus,
        tmp_path / "uninterrupted",
        run_id="uninterrupted",
        allow_test_corpus=True,
    )
    uninterrupted.train()

    interrupted = Gen1Trainer.create(
        config,
        corpus,
        tmp_path / "resumed",
        run_id="resumed",
        allow_test_corpus=True,
    )
    interrupted_result = interrupted.train(until_step=2)
    assert interrupted_result["status"] == "interrupted"
    resumed = Gen1Trainer.resume_latest(
        tmp_path / "resumed", corpus, allow_test_corpus=True
    )
    resumed.train()

    for name, expected in uninterrupted.model.state_dict().items():
        torch.testing.assert_close(expected, resumed.model.state_dict()[name], rtol=0, atol=0)
    assert_nested_equal(uninterrupted.optimizer.state_dict(), resumed.optimizer.state_dict())
    assert uninterrupted.scheduler.state_dict() == resumed.scheduler.state_dict()
    assert uninterrupted.stream.state_dict()["epoch"] == resumed.stream.state_dict()["epoch"]
    assert uninterrupted.stream.state_dict()["position"] == resumed.stream.state_dict()["position"]
    assert uninterrupted.stream.state_dict()["permutation"] == resumed.stream.state_dict()["permutation"]
    torch.testing.assert_close(
        uninterrupted.stream.state_dict()["generatorState"],
        resumed.stream.state_dict()["generatorState"],
        rtol=0,
        atol=0,
    )
    uninterrupted_state = torch.load(
        tmp_path
        / "uninterrupted"
        / "checkpoints"
        / "step-00000004"
        / "training-state.pt",
        map_location="cpu",
        weights_only=True,
    )
    resumed_state = torch.load(
        tmp_path / "resumed" / "checkpoints" / "step-00000004" / "training-state.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert_nested_equal(uninterrupted_state["rng"], resumed_state["rng"])
    assert_nested_equal(uninterrupted_state["scaler"], resumed_state["scaler"])
    assert uninterrupted_state["accumulationMicroStep"] == 0
    assert resumed_state["accumulationMicroStep"] == 0
    for expected, actual in zip(uninterrupted.metrics, resumed.metrics, strict=True):
        for key in ("step", "trainingLoss", "learningRate", "gradientNormBeforeClip"):
            assert expected[key] == actual[key]
        if "validationLoss" in expected:
            assert expected["validationLoss"] == actual["validationLoss"]


def test_synthetic_scaler_overflow_retries_exact_batch_and_resumes_safely(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.amp, "GradScaler", OverflowOnceGradScaler)
    config = training_config(max_steps=4, dropout=0.1)
    corpus = toy_corpus()

    uninterrupted = Gen1Trainer.create(
        config,
        corpus,
        tmp_path / "overflow-uninterrupted",
        run_id="overflow-uninterrupted",
        allow_test_corpus=True,
    )
    uninterrupted.train()

    interrupted = Gen1Trainer.create(
        config,
        corpus,
        tmp_path / "overflow-resumed",
        run_id="overflow-resumed",
        allow_test_corpus=True,
    )
    interrupted.train(until_step=2)
    resumed = Gen1Trainer.resume_latest(
        tmp_path / "overflow-resumed", corpus, allow_test_corpus=True
    )
    resumed.train()

    assert uninterrupted.metrics[0]["mixedPrecisionOverflowRetries"] == 1
    assert resumed.metrics[0]["mixedPrecisionOverflowRetries"] == 1
    assert uninterrupted.metrics[0]["gradientScaleBefore"] == 32_768.0
    assert resumed.metrics[0]["gradientScaleAfter"] == 32_768.0
    for name, expected in uninterrupted.model.state_dict().items():
        torch.testing.assert_close(expected, resumed.model.state_dict()[name], rtol=0, atol=0)
    assert_nested_equal(uninterrupted.optimizer.state_dict(), resumed.optimizer.state_dict())
    assert uninterrupted.scheduler.state_dict() == resumed.scheduler.state_dict()
    assert_nested_equal(uninterrupted.stream.state_dict(), resumed.stream.state_dict())
    assert uninterrupted.scaler.state_dict() == resumed.scaler.state_dict()
    uninterrupted_state = torch.load(
        tmp_path
        / "overflow-uninterrupted"
        / "checkpoints"
        / "step-00000004"
        / "training-state.pt",
        map_location="cpu",
        weights_only=True,
    )
    resumed_state = torch.load(
        tmp_path
        / "overflow-resumed"
        / "checkpoints"
        / "step-00000004"
        / "training-state.pt",
        map_location="cpu",
        weights_only=True,
    )
    for key in ("rng", "scaler", "dataCursor"):
        assert_nested_equal(uninterrupted_state[key], resumed_state[key])
    for expected, actual in zip(uninterrupted.metrics, resumed.metrics, strict=True):
        for key in (
            "step",
            "trainingLoss",
            "learningRate",
            "gradientNormBeforeClip",
            "mixedPrecisionOverflowRetries",
            "gradientScaleBefore",
            "gradientScaleAfter",
        ):
            assert expected[key] == actual[key]


def test_checkpoint_tampering_and_unapproved_production_corpus_fail_closed(tmp_path) -> None:
    config = training_config(max_steps=2)
    corpus = toy_corpus()
    with pytest.raises(TrainingRunError, match="approved VFAI-009 corpus"):
        Gen1Trainer.create(config, corpus, tmp_path / "denied", run_id="denied")

    trainer = Gen1Trainer.create(
        config,
        corpus,
        tmp_path / "tampered",
        run_id="tampered",
        allow_test_corpus=True,
    )
    trainer.train(until_step=1)
    checkpoint = tmp_path / "tampered" / "checkpoints" / "step-00000001"
    state_path = checkpoint / "training-state.pt"
    state_path.write_bytes(state_path.read_bytes() + b"tamper")
    with pytest.raises(TrainingCheckpointError, match="checksum/size mismatch"):
        Gen1Trainer.resume(checkpoint, corpus, allow_test_corpus=True)


def test_unsupported_cpu_mixed_precision_fails_explicitly(tmp_path) -> None:
    if torch.cpu._is_avx512_bf16_supported():
        pytest.skip("CPU has native BF16 support")
    config = replace(training_config(max_steps=1), precision="bfloat16")
    with pytest.raises(TrainingRunError, match="AVX-512 BF16"):
        Gen1Trainer.create(
            config,
            toy_corpus(),
            tmp_path / "unsupported-bf16",
            run_id="unsupported-bf16",
            allow_test_corpus=True,
        )
