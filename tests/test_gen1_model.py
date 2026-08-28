from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version

import pytest

try:
    installed_torch = version("torch")
except PackageNotFoundError:
    pytest.skip("Gen1 pinned PyTorch runtime is not installed", allow_module_level=True)
if installed_torch.partition("+")[0] != "2.8.0":
    pytest.skip(
        f"Gen1 tests require pinned PyTorch 2.8.0, found {installed_torch}",
        allow_module_level=True,
    )
try:
    import torch
except (ImportError, OSError) as exc:  # Host Python may not have the pinned native runtime.
    pytest.skip(f"Gen1 PyTorch runtime unavailable: {exc}", allow_module_level=True)

from model.gen1 import (  # noqa: E402
    DEFAULT_ALLOCATION_LIMIT,
    CheckpointContractError,
    Gen1Config,
    Gen1ConfigError,
    ParameterAllocationError,
    VoltForgeGen1,
)


def tiny_config(**overrides: object) -> Gen1Config:
    values: dict[str, object] = {
        "vocab_size": 32,
        "max_sequence_length": 16,
        "d_model": 32,
        "n_layers": 2,
        "n_heads": 4,
        "n_kv_heads": 2,
        "d_ff": 64,
        "dropout": 0.0,
        "seed": 29,
    }
    values.update(overrides)
    return Gen1Config(**values)


def test_default_is_safe_and_parameter_formula_is_exact() -> None:
    config = Gen1Config()
    assert config.parameter_count() == 289_088
    assert config.parameter_count() < DEFAULT_ALLOCATION_LIMIT

    model = VoltForgeGen1.for_training(config)
    assert model.parameter_count() == config.parameter_count()
    assert sum(parameter.numel() for parameter in model.parameters()) == 289_088
    assert model.token_embedding.weight.data_ptr() == model.lm_head.weight.data_ptr()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"d_model": 30, "n_heads": 4}, "divisible by n_heads"),
        ({"n_heads": 4, "n_kv_heads": 3}, "divisible by n_kv_heads"),
        ({"d_model": 12, "n_heads": 4}, "head_dim must be even"),
        ({"dropout": 1.0}, "dropout"),
    ],
)
def test_invalid_architecture_configs_fail_closed(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(Gen1ConfigError, match=message):
        tiny_config(**overrides)


def test_tensor_shapes_gqa_cache_and_weight_tying() -> None:
    config = tiny_config()
    model = VoltForgeGen1.for_training(config)
    input_ids = torch.tensor([[2, 5, 8, 3], [2, 7, 9, 3]], dtype=torch.long)
    output = model(input_ids, use_cache=True)

    assert output.logits.shape == (2, 4, config.vocab_size)
    assert output.loss is None
    assert output.past_key_values is not None
    assert len(output.past_key_values) == config.n_layers
    for key, value in output.past_key_values:
        assert key.shape == (2, config.n_kv_heads, 4, config.head_dim)
        assert value.shape == key.shape
    assert model.token_embedding.weight.data_ptr() == model.lm_head.weight.data_ptr()


def test_causal_mask_prevents_future_tokens_from_changing_prefix_logits() -> None:
    model = VoltForgeGen1.for_training(tiny_config()).eval()
    first = torch.tensor([[2, 4, 6, 8, 10]], dtype=torch.long)
    changed_future = torch.tensor([[2, 4, 6, 13, 15]], dtype=torch.long)

    first_logits = model(first).logits
    changed_logits = model(changed_future).logits
    torch.testing.assert_close(first_logits[:, :3], changed_logits[:, :3], rtol=0, atol=0)


def test_loss_honors_ignore_index_and_all_masked_batches_are_finite() -> None:
    model = VoltForgeGen1.for_training(tiny_config()).eval()
    input_ids = torch.tensor([[2, 4, 6, 8, 3]], dtype=torch.long)
    labels = input_ids.clone()
    labels[:, 3] = -100
    first_loss = model(input_ids, labels=labels).loss
    assert first_loss is not None
    logits = model(input_ids).logits[:, :-1].float()
    shifted_labels = labels[:, 1:]
    valid = shifted_labels != -100
    expected_loss = torch.nn.functional.cross_entropy(logits[valid], shifted_labels[valid])
    torch.testing.assert_close(first_loss, expected_loss, rtol=0, atol=0)

    all_masked = torch.full_like(input_ids, -100)
    all_masked_loss = model(input_ids, labels=all_masked).loss
    assert all_masked_loss is not None
    assert torch.isfinite(all_masked_loss)
    assert all_masked_loss.item() == 0.0


def test_kv_cache_token_by_token_logits_equal_full_causal_forward() -> None:
    model = VoltForgeGen1.for_training(tiny_config()).eval()
    input_ids = torch.tensor([[2, 5, 7, 11, 13, 3]], dtype=torch.long)
    full_logits = model(input_ids).logits

    cache = None
    incremental_logits = []
    for index in range(input_ids.shape[1]):
        output = model(
            input_ids[:, index : index + 1],
            past_key_values=cache,
            use_cache=True,
        )
        incremental_logits.append(output.logits)
        cache = output.past_key_values

    torch.testing.assert_close(
        torch.cat(incremental_logits, dim=1),
        full_logits,
        rtol=1e-5,
        atol=1e-6,
    )


def test_training_initialization_is_seed_deterministic_without_global_rng_leak() -> None:
    config = tiny_config(seed=101)
    torch.manual_seed(999)
    state_before = torch.random.get_rng_state().clone()
    first = VoltForgeGen1.for_training(config)
    state_after = torch.random.get_rng_state()
    second = VoltForgeGen1.for_training(config)

    torch.testing.assert_close(state_before, state_after, rtol=0, atol=0)
    for first_tensor, second_tensor in zip(
        first.state_dict().values(), second.state_dict().values(), strict=True
    ):
        torch.testing.assert_close(first_tensor, second_tensor, rtol=0, atol=0)


def test_checkpoint_round_trip_is_exact_and_inference_never_initializes(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tiny_config()
    training_model = VoltForgeGen1.for_training(config).eval()
    input_ids = torch.tensor([[2, 7, 12, 3]], dtype=torch.long)
    expected = training_model(input_ids).logits.detach()
    checkpoint = training_model.save_checkpoint(tmp_path / "checkpoint")

    def fail_if_initialized(*args, **kwargs):
        raise AssertionError("checkpoint inference attempted random initialization")

    monkeypatch.setattr(VoltForgeGen1, "_initialize_training_weights", fail_if_initialized)
    loaded = VoltForgeGen1.from_checkpoint(checkpoint)

    assert loaded.training is False
    assert loaded.parameter_count() == config.parameter_count()
    assert loaded.token_embedding.weight.data_ptr() == loaded.lm_head.weight.data_ptr()
    torch.testing.assert_close(loaded(input_ids).logits, expected, rtol=0, atol=0)
    for expected_tensor, loaded_tensor in zip(
        training_model.state_dict().values(), loaded.state_dict().values(), strict=True
    ):
        torch.testing.assert_close(expected_tensor, loaded_tensor, rtol=0, atol=0)


def test_checkpoint_checksum_and_missing_files_fail_before_model_materialization(
    tmp_path,
) -> None:
    with pytest.raises(CheckpointContractError, match="required checkpoint file is missing"):
        VoltForgeGen1.from_checkpoint(tmp_path / "absent")

    model = VoltForgeGen1.for_training(tiny_config())
    checkpoint = model.save_checkpoint(tmp_path / "tampered")
    config_path = checkpoint / "config.json"
    document = json.loads(config_path.read_text(encoding="utf-8"))
    document["seed"] += 1
    config_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(CheckpointContractError, match="checksum/size mismatch"):
        VoltForgeGen1.from_checkpoint(checkpoint)


def test_billion_scale_configuration_is_rejected_before_allocation() -> None:
    billion_scale = Gen1Config(
        vocab_size=3_072,
        max_sequence_length=2_048,
        d_model=2_048,
        n_layers=24,
        n_heads=16,
        n_kv_heads=4,
        d_ff=5_504,
    )
    assert billion_scale.parameter_count() > 900_000_000
    with pytest.raises(ParameterAllocationError, match="no tensors were allocated"):
        VoltForgeGen1.for_training(billion_scale)


def test_tiny_model_overfits_and_reproduces_a_controlled_sequence() -> None:
    config = Gen1Config(
        vocab_size=16,
        max_sequence_length=8,
        d_model=16,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        d_ff=32,
        dropout=0.0,
        seed=7,
    )
    model = VoltForgeGen1.for_training(config)
    sequence = torch.tensor([[2, 4, 5, 6, 7, 3]], dtype=torch.long).repeat(8, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.0)

    initial_loss = model(sequence, labels=sequence).loss
    assert initial_loss is not None
    for _ in range(180):
        optimizer.zero_grad(set_to_none=True)
        loss = model(sequence, labels=sequence).loss
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.eval()
    final = model(sequence, labels=sequence)
    assert final.loss is not None
    assert final.loss.item() < 0.02
    assert final.loss.item() < initial_loss.item() * 0.01
    predicted_next = final.logits[:, :-1].argmax(dim=-1)
    assert torch.equal(predicted_next, sequence[:, 1:])
