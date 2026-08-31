"""PyTorch implementation of the production VoltForge Gen1 decoder contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
from typing import Any, Literal, TypeAlias


PINNED_TORCH_VERSION = "2.8.0"
try:
    _installed_torch_version = version("torch")
except PackageNotFoundError as exc:
    raise RuntimeError(
        "VFDLM Gen1 requires the pinned PyTorch runtime; install requirements-gen1.txt"
    ) from exc
if _installed_torch_version.partition("+")[0] != PINNED_TORCH_VERSION:
    raise RuntimeError(
        "VFDLM Gen1 requires PyTorch "
        f"{PINNED_TORCH_VERSION}, found {_installed_torch_version}; "
        "install requirements-gen1.txt before importing the native model"
    )

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import (
    ARCHITECTURE_ID,
    DEFAULT_ALLOCATION_LIMIT,
    Gen1Config,
)


LayerKVCache: TypeAlias = tuple[Tensor, Tensor]
KVCache: TypeAlias = tuple[LayerKVCache, ...]
AttentionBackend: TypeAlias = Literal["manual", "sdpa"]


class CheckpointContractError(RuntimeError):
    """Raised when a Gen1 checkpoint is absent, incomplete, or incompatible."""


@dataclass(slots=True)
class Gen1Output:
    logits: Tensor
    loss: Tensor | None = None
    past_key_values: KVCache | None = None


class RMSNorm(nn.Module):
    def __init__(
        self,
        width: int,
        epsilon: float,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.empty(width, device=device, dtype=dtype))

    def forward(self, value: Tensor) -> Tensor:
        normalized = value.float() * torch.rsqrt(
            value.float().pow(2).mean(dim=-1, keepdim=True) + self.epsilon
        )
        return normalized.to(value.dtype) * self.weight


def _rotate_half(value: Tensor) -> Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rope(value: Tensor, positions: Tensor, theta: float) -> Tensor:
    head_dim = value.shape[-1]
    frequency_steps = torch.arange(0, head_dim, 2, device=value.device, dtype=torch.float32)
    inverse_frequency = theta ** (-frequency_steps / head_dim)
    frequencies = torch.outer(positions.float(), inverse_frequency)
    angles = torch.cat((frequencies, frequencies), dim=-1)
    cosine = angles.cos().to(value.dtype)[None, None, :, :]
    sine = angles.sin().to(value.dtype)[None, None, :, :]
    return value * cosine + _rotate_half(value) * sine


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        config: Gen1Config,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        factory = {"device": device, "dtype": dtype, "bias": False}
        self.config = config
        self.q_proj = nn.Linear(config.d_model, config.d_model, **factory)
        self.k_proj = nn.Linear(config.d_model, config.kv_width, **factory)
        self.v_proj = nn.Linear(config.d_model, config.kv_width, **factory)
        self.o_proj = nn.Linear(config.d_model, config.d_model, **factory)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.attention_backend: AttentionBackend = "manual"

    def _shape_query(self, value: Tensor) -> Tensor:
        batch, length, _ = value.shape
        return value.view(batch, length, self.config.n_heads, self.config.head_dim).transpose(1, 2)

    def _shape_key_value(self, value: Tensor) -> Tensor:
        batch, length, _ = value.shape
        return value.view(
            batch, length, self.config.n_kv_heads, self.config.head_dim
        ).transpose(1, 2)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None,
        past_key_value: LayerKVCache | None,
        use_cache: bool,
    ) -> tuple[Tensor, LayerKVCache | None]:
        batch, query_length, _ = hidden_states.shape
        past_length = 0
        if past_key_value is not None:
            past_key, past_value = past_key_value
            expected_prefix = (batch, self.config.n_kv_heads)
            if past_key.ndim != 4 or past_value.ndim != 4:
                raise ValueError("cached keys and values must be rank-4 tensors")
            if past_key.shape != past_value.shape:
                raise ValueError("cached key/value shapes must match")
            if tuple(past_key.shape[:2]) != expected_prefix or past_key.shape[-1] != self.config.head_dim:
                raise ValueError("cached key/value shape is incompatible with the Gen1 config")
            if past_key.device != hidden_states.device or past_value.device != hidden_states.device:
                raise ValueError("cached keys and values must be on the model device")
            past_length = past_key.shape[2]

        positions = torch.arange(
            past_length,
            past_length + query_length,
            device=hidden_states.device,
        )
        query = _apply_rope(self._shape_query(self.q_proj(hidden_states)), positions, self.config.rope_theta)
        key = _apply_rope(self._shape_key_value(self.k_proj(hidden_states)), positions, self.config.rope_theta)
        value = self._shape_key_value(self.v_proj(hidden_states))

        if past_key_value is not None:
            key = torch.cat((past_key_value[0], key), dim=2)
            value = torch.cat((past_key_value[1], value), dim=2)

        total_length = key.shape[2]
        repeat_factor = self.config.n_heads // self.config.n_kv_heads
        expanded_key = key.repeat_interleave(repeat_factor, dim=1)
        expanded_value = value.repeat_interleave(repeat_factor, dim=1)
        query_positions = torch.arange(
            past_length,
            past_length + query_length,
            device=hidden_states.device,
        )[:, None]
        key_positions = torch.arange(total_length, device=hidden_states.device)[None, :]
        allowed = (key_positions <= query_positions)[None, None, :, :]
        if attention_mask is not None:
            if attention_mask.ndim != 2 or tuple(attention_mask.shape) != (batch, total_length):
                raise ValueError(
                    "attention_mask must have shape [batch, cached_length + input_length]"
                )
            allowed = allowed & attention_mask.to(device=hidden_states.device, dtype=torch.bool)[
                :, None, None, :
            ]
        if self.attention_backend == "sdpa":
            attended = F.scaled_dot_product_attention(
                query,
                expanded_key,
                expanded_value,
                attn_mask=allowed,
                dropout_p=self.config.dropout if self.training else 0.0,
                is_causal=False,
            )
        elif self.attention_backend == "manual":
            scores = torch.matmul(query, expanded_key.transpose(-2, -1)) * (
                self.config.head_dim ** -0.5
            )
            probabilities = torch.softmax(
                scores.float().masked_fill(~allowed, torch.finfo(torch.float32).min),
                dim=-1,
            )
            probabilities = probabilities * allowed
            probabilities = probabilities / probabilities.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)
            probabilities = self.attention_dropout(probabilities.to(hidden_states.dtype))
            attended = torch.matmul(probabilities, expanded_value)
        else:
            raise ValueError(f"unsupported attention backend: {self.attention_backend}")
        attended = attended.transpose(1, 2).contiguous().view(
            batch, query_length, self.config.d_model
        )
        present = (key, value) if use_cache else None
        return self.o_proj(attended), present


class SwiGLU(nn.Module):
    def __init__(
        self,
        config: Gen1Config,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        factory = {"device": device, "dtype": dtype, "bias": False}
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, **factory)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, **factory)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, **factory)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.dropout(
            self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        )


class DecoderBlock(nn.Module):
    def __init__(
        self,
        config: Gen1Config,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(
            config.d_model, config.norm_epsilon, device=device, dtype=dtype
        )
        self.attention = GroupedQueryAttention(config, device=device, dtype=dtype)
        self.feed_forward_norm = RMSNorm(
            config.d_model, config.norm_epsilon, device=device, dtype=dtype
        )
        self.feed_forward = SwiGLU(config, device=device, dtype=dtype)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None,
        past_key_value: LayerKVCache | None,
        use_cache: bool,
    ) -> tuple[Tensor, LayerKVCache | None]:
        attention_output, present = self.attention(
            self.attention_norm(hidden_states),
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + attention_output
        hidden_states = hidden_states + self.feed_forward(
            self.feed_forward_norm(hidden_states)
        )
        return hidden_states, present


class VoltForgeGen1(nn.Module):
    """Decoder-only Transformer trained solely from VoltForge-owned data.

    Use :meth:`for_training` to initialize deterministic random weights and
    :meth:`from_checkpoint` for inference. Direct construction requires an
    internal device argument and therefore cannot silently allocate a model.
    """

    def __init__(
        self,
        config: Gen1Config,
        *,
        _construction_device: torch.device,
        _dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.config = config
        self._materialized = False
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            device=_construction_device,
            dtype=_dtype,
        )
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            DecoderBlock(config, device=_construction_device, dtype=_dtype)
            for _ in range(config.n_layers)
        )
        self.final_norm = RMSNorm(
            config.d_model,
            config.norm_epsilon,
            device=_construction_device,
            dtype=_dtype,
        )
        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
            device=_construction_device,
            dtype=_dtype,
        )
        self._tie_weights()

    def _tie_weights(self) -> None:
        if self.config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    @classmethod
    def for_training(
        cls,
        config: Gen1Config | None = None,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        allocation_limit: int = DEFAULT_ALLOCATION_LIMIT,
    ) -> "VoltForgeGen1":
        resolved_config = config or Gen1Config()
        resolved_config.enforce_allocation_limit(allocation_limit)
        resolved_device = torch.device(device)
        model = cls(
            resolved_config,
            _construction_device=torch.device("meta"),
            _dtype=dtype,
        )
        model.to_empty(device=resolved_device)
        model._tie_weights()
        model._initialize_training_weights(resolved_device)
        model._materialized = True
        model.train()
        model._assert_parameter_contract()
        return model

    def _initialize_training_weights(self, device: torch.device) -> None:
        fork_devices: list[int] = []
        if device.type == "cuda":
            fork_devices = [device.index if device.index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(self.config.seed)
            for module in self.modules():
                if isinstance(module, RMSNorm):
                    nn.init.ones_(module.weight)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0.0, std=self.config.initialization_std)
                elif isinstance(module, nn.Linear):
                    if module is self.lm_head and self.config.tie_embeddings:
                        continue
                    nn.init.normal_(module.weight, mean=0.0, std=self.config.initialization_std)
        self._tie_weights()

    def _assert_parameter_contract(self) -> None:
        actual = sum(parameter.numel() for parameter in self.parameters())
        expected = self.config.parameter_count()
        if actual != expected:
            raise RuntimeError(
                f"Gen1 parameter contract mismatch: config={expected:,}, module={actual:,}"
            )

    def parameter_count(self) -> int:
        self._assert_parameter_contract()
        return self.config.parameter_count()

    def set_attention_backend(self, backend: AttentionBackend) -> None:
        if backend not in {"manual", "sdpa"}:
            raise ValueError("attention backend must be manual or sdpa")
        for layer in self.layers:
            layer.attention.attention_backend = backend

    def _ensure_materialized(self) -> None:
        if not self._materialized or any(parameter.is_meta for parameter in self.parameters()):
            raise RuntimeError(
                "Gen1 weights are not materialized; use for_training() or from_checkpoint()"
            )

    def forward(
        self,
        input_ids: Tensor,
        *,
        labels: Tensor | None = None,
        attention_mask: Tensor | None = None,
        past_key_values: KVCache | None = None,
        use_cache: bool = False,
    ) -> Gen1Output:
        self._ensure_materialized()
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be a rank-2 torch.long tensor")
        batch, input_length = input_ids.shape
        if input_length == 0:
            raise ValueError("input_ids cannot be empty")
        if torch.any(input_ids < 0) or torch.any(input_ids >= self.config.vocab_size):
            raise ValueError("input_ids contain a token outside the configured vocabulary")
        if past_key_values is not None and len(past_key_values) != self.config.n_layers:
            raise ValueError("past_key_values must contain exactly one entry per decoder layer")
        layer_cache: tuple[LayerKVCache | None, ...] = (
            tuple(past_key_values)
            if past_key_values is not None
            else tuple(None for _ in range(self.config.n_layers))
        )
        past_length = layer_cache[0][0].shape[2] if layer_cache[0] is not None else 0
        if any(
            cache is not None and cache[0].shape[2] != past_length
            for cache in layer_cache
        ):
            raise ValueError("every decoder layer cache must have the same sequence length")
        total_length = past_length + input_length
        if total_length > self.config.max_sequence_length:
            raise ValueError(
                f"sequence length {total_length} exceeds max_sequence_length "
                f"{self.config.max_sequence_length}"
            )
        if attention_mask is not None and tuple(attention_mask.shape) != (batch, total_length):
            raise ValueError(
                "attention_mask must have shape [batch, cached_length + input_length]"
            )
        if labels is not None:
            if past_key_values is not None:
                raise ValueError("labels cannot be combined with a KV cache")
            if labels.ndim != 2 or tuple(labels.shape) != (batch, input_length):
                raise ValueError("labels must have the same rank-2 shape as input_ids")
            if labels.dtype != torch.long:
                raise ValueError("labels must be a torch.long tensor")
            invalid_labels = (labels != -100) & (
                (labels < 0) | (labels >= self.config.vocab_size)
            )
            if torch.any(invalid_labels):
                raise ValueError("labels contain a token outside the configured vocabulary")

        hidden_states = self.embedding_dropout(self.token_embedding(input_ids))
        present_key_values: list[LayerKVCache] = []
        for block, cached in zip(self.layers, layer_cache, strict=True):
            hidden_states, present = block(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=cached,
                use_cache=use_cache,
            )
            if present is not None:
                present_key_values.append(present)
        logits = self.lm_head(self.final_norm(hidden_states))

        loss: Tensor | None = None
        if labels is not None:
            if input_length < 2:
                raise ValueError("next-token loss requires at least two input tokens")
            shifted_logits = logits[:, :-1, :].float()
            shifted_labels = labels[:, 1:]
            valid = shifted_labels != -100
            if torch.any(valid):
                loss = F.cross_entropy(shifted_logits[valid], shifted_labels[valid])
            else:
                loss = shifted_logits.sum() * 0.0

        cache_output: KVCache | None = tuple(present_key_values) if use_cache else None
        return Gen1Output(logits=logits, loss=loss, past_key_values=cache_output)

    def save_checkpoint(self, directory: str | Path) -> Path:
        self._ensure_materialized()
        self._assert_parameter_contract()
        target = Path(directory)
        if target.exists() and (not target.is_dir() or any(target.iterdir())):
            raise CheckpointContractError(
                f"checkpoint directory must be absent or empty: {target}"
            )
        target.mkdir(parents=True, exist_ok=True)
        config_path = target / "config.json"
        weights_path = target / "weights.pt"
        manifest_path = target / "manifest.json"

        config_path.write_text(self.config.canonical_json() + "\n", encoding="utf-8")
        state = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in self.state_dict().items()
        }
        torch.save(state, weights_path)
        manifest = {
            "schemaVersion": 1,
            "architectureId": ARCHITECTURE_ID,
            "checkpointFormat": "pytorch-weights-only-state-dict-v1",
            "configFingerprint": self.config.fingerprint(),
            "parameterCount": self.config.parameter_count(),
            "parameterBreakdown": self.config.parameter_breakdown(),
            "weightDtype": _dtype_name(next(self.parameters()).dtype),
            "files": {
                "config.json": _file_descriptor(config_path),
                "weights.pt": _file_descriptor(weights_path),
            },
        }
        manifest_path.write_text(_canonical_json(manifest) + "\n", encoding="utf-8")
        return target

    @classmethod
    def from_checkpoint(
        cls,
        directory: str | Path,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        mmap_weights: bool = False,
        allocation_limit: int = DEFAULT_ALLOCATION_LIMIT,
    ) -> "VoltForgeGen1":
        target = Path(directory)
        config_path = target / "config.json"
        weights_path = target / "weights.pt"
        manifest_path = target / "manifest.json"
        for path in (config_path, weights_path, manifest_path):
            if not path.is_file():
                raise CheckpointContractError(f"required checkpoint file is missing: {path}")

        manifest = _read_json_object(manifest_path, "checkpoint manifest")
        if manifest.get("schemaVersion") != 1:
            raise CheckpointContractError("unsupported checkpoint manifest schemaVersion")
        if manifest.get("architectureId") != ARCHITECTURE_ID:
            raise CheckpointContractError("checkpoint architectureId is incompatible")
        if manifest.get("checkpointFormat") != "pytorch-weights-only-state-dict-v1":
            raise CheckpointContractError("checkpoint format is incompatible")
        files = manifest.get("files")
        if not isinstance(files, Mapping) or set(files) != {"config.json", "weights.pt"}:
            raise CheckpointContractError("checkpoint manifest file set is invalid")
        _verify_file_descriptor(config_path, files["config.json"])
        _verify_file_descriptor(weights_path, files["weights.pt"])

        config_document = _read_json_object(config_path, "Gen1 config")
        try:
            config = Gen1Config.from_dict(config_document)
        except ValueError as exc:
            raise CheckpointContractError(f"invalid Gen1 config: {exc}") from exc
        if manifest.get("configFingerprint") != config.fingerprint():
            raise CheckpointContractError("checkpoint config fingerprint mismatch")
        if manifest.get("parameterCount") != config.parameter_count():
            raise CheckpointContractError("checkpoint parameter count mismatch")
        if manifest.get("parameterBreakdown") != config.parameter_breakdown():
            raise CheckpointContractError("checkpoint parameter breakdown mismatch")
        config.enforce_allocation_limit(allocation_limit)

        checkpoint_dtype = _dtype_from_name(manifest.get("weightDtype"))
        model = cls(
            config,
            _construction_device=torch.device("meta"),
            _dtype=checkpoint_dtype,
        )
        expected_state = model.state_dict()
        try:
            loaded_state = torch.load(
                weights_path,
                map_location="cpu",
                weights_only=True,
                mmap=mmap_weights,
            )
        except Exception as exc:
            raise CheckpointContractError(f"unable to load checkpoint weights: {exc}") from exc
        if not isinstance(loaded_state, Mapping):
            raise CheckpointContractError("checkpoint weights must be a tensor mapping")
        if set(loaded_state) != set(expected_state):
            missing = sorted(set(expected_state) - set(loaded_state))
            extra = sorted(set(loaded_state) - set(expected_state))
            raise CheckpointContractError(
                f"checkpoint tensor names mismatch: missing={missing}, extra={extra}"
            )
        for name, expected in expected_state.items():
            loaded = loaded_state[name]
            if not isinstance(loaded, Tensor):
                raise CheckpointContractError(f"checkpoint value is not a tensor: {name}")
            if tuple(loaded.shape) != tuple(expected.shape):
                raise CheckpointContractError(
                    f"checkpoint tensor shape mismatch for {name}: "
                    f"expected={tuple(expected.shape)}, actual={tuple(loaded.shape)}"
                )
            if loaded.dtype != expected.dtype:
                raise CheckpointContractError(
                    f"checkpoint tensor dtype mismatch for {name}: "
                    f"expected={expected.dtype}, actual={loaded.dtype}"
                )
        if config.tie_embeddings and not torch.equal(
            loaded_state["token_embedding.weight"], loaded_state["lm_head.weight"]
        ):
            raise CheckpointContractError("tied embedding tensors disagree in checkpoint")

        model.load_state_dict(loaded_state, strict=True, assign=True)
        model._tie_weights()
        resolved_dtype = dtype or checkpoint_dtype
        model.to(device=torch.device(device), dtype=resolved_dtype)
        model._tie_weights()
        model._materialized = True
        model.eval()
        model._assert_parameter_contract()
        return model


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_descriptor(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def _verify_file_descriptor(path: Path, descriptor: Any) -> None:
    if not isinstance(descriptor, Mapping):
        raise CheckpointContractError(f"checkpoint descriptor is invalid for {path.name}")
    if set(descriptor) != {"bytes", "sha256"}:
        raise CheckpointContractError(f"checkpoint descriptor keys are invalid for {path.name}")
    if descriptor["bytes"] != path.stat().st_size or descriptor["sha256"] != _sha256(path):
        raise CheckpointContractError(f"checkpoint checksum/size mismatch for {path.name}")


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointContractError(f"unable to read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise CheckpointContractError(f"{label} must be a JSON object")
    return value


def _dtype_name(dtype: torch.dtype) -> str:
    names = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
    }
    try:
        return names[dtype]
    except KeyError as exc:
        raise CheckpointContractError(f"unsupported checkpoint weight dtype: {dtype}") from exc


def _dtype_from_name(value: Any) -> torch.dtype:
    names = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return names[value]
    except (KeyError, TypeError) as exc:
        raise CheckpointContractError(f"unsupported checkpoint weightDtype: {value!r}") from exc
