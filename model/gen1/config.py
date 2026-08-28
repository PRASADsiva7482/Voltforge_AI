"""Deterministic configuration and parameter accounting for VFDLM Gen1."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping


ARCHITECTURE_ID = "vfdlm-gen1-decoder-v1"
CONFIG_SCHEMA_VERSION = 1

# Edge/core experiments fit below this ceiling. Larger experiments must make their
# memory intent explicit, and an accidentally resurrected 1B config fails before
# any tensor is allocated.
DEFAULT_ALLOCATION_LIMIT = 75_000_000


class Gen1ConfigError(ValueError):
    """Raised when an architecture configuration violates the Gen1 contract."""


class ParameterAllocationError(RuntimeError):
    """Raised before a model whose parameter count exceeds the caller's limit."""


@dataclass(frozen=True, slots=True)
class Gen1Config:
    """Immutable Gen1 decoder configuration.

    Defaults are a small executable contract model, not a release-size profile.
    VFAI-013 owns evidence-based edge/core/server profile selection.
    """

    vocab_size: int = 3_072
    max_sequence_length: int = 128
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int = 2
    d_ff: int = 176
    rope_theta: float = 10_000.0
    norm_epsilon: float = 1e-5
    dropout: float = 0.0
    tie_embeddings: bool = True
    initialization_std: float = 0.02
    seed: int = 17

    def __post_init__(self) -> None:
        integer_fields = {
            "vocab_size": self.vocab_size,
            "max_sequence_length": self.max_sequence_length,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "d_ff": self.d_ff,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise Gen1ConfigError(f"{name} must be a positive integer")
        if self.vocab_size < 8:
            raise Gen1ConfigError("vocab_size must leave room for the fixed tokenizer contract")
        if self.max_sequence_length < 2:
            raise Gen1ConfigError("max_sequence_length must be at least 2")
        if self.d_model % self.n_heads != 0:
            raise Gen1ConfigError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise Gen1ConfigError("n_heads must be divisible by n_kv_heads")
        if self.head_dim % 2 != 0:
            raise Gen1ConfigError("attention head_dim must be even for RoPE")
        numeric_fields = {
            "rope_theta": self.rope_theta,
            "norm_epsilon": self.norm_epsilon,
            "dropout": self.dropout,
            "initialization_std": self.initialization_std,
        }
        for name, value in numeric_fields.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise Gen1ConfigError(f"{name} must be numeric")
        if not isinstance(self.tie_embeddings, bool):
            raise Gen1ConfigError("tie_embeddings must be a boolean")
        if not 0.0 <= self.dropout < 1.0:
            raise Gen1ConfigError("dropout must be in [0, 1)")
        if self.rope_theta <= 0.0:
            raise Gen1ConfigError("rope_theta must be positive")
        if self.norm_epsilon <= 0.0:
            raise Gen1ConfigError("norm_epsilon must be positive")
        if self.initialization_std <= 0.0:
            raise Gen1ConfigError("initialization_std must be positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise Gen1ConfigError("seed must be a non-negative integer")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def kv_width(self) -> int:
        return self.n_kv_heads * self.head_dim

    def parameter_breakdown(self) -> dict[str, int]:
        embeddings = self.vocab_size * self.d_model
        attention_per_layer = (
            self.d_model * self.d_model
            + 2 * self.d_model * self.kv_width
            + self.d_model * self.d_model
        )
        feed_forward_per_layer = 3 * self.d_model * self.d_ff
        norms_per_layer = 2 * self.d_model
        output_head = 0 if self.tie_embeddings else self.vocab_size * self.d_model
        return {
            "tokenEmbedding": embeddings,
            "layerAttention": self.n_layers * attention_per_layer,
            "layerFeedForward": self.n_layers * feed_forward_per_layer,
            "layerNorms": self.n_layers * norms_per_layer,
            "finalNorm": self.d_model,
            "untiedOutputHead": output_head,
        }

    def parameter_count(self) -> int:
        return sum(self.parameter_breakdown().values())

    def enforce_allocation_limit(self, limit: int = DEFAULT_ALLOCATION_LIMIT) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ParameterAllocationError("allocation limit must be a positive integer")
        count = self.parameter_count()
        if count > limit:
            raise ParameterAllocationError(
                f"Gen1 config requires {count:,} parameters, above the explicit "
                f"allocation limit of {limit:,}; no tensors were allocated"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": CONFIG_SCHEMA_VERSION,
            "architectureId": ARCHITECTURE_ID,
            "vocabSize": self.vocab_size,
            "maxSequenceLength": self.max_sequence_length,
            "dModel": self.d_model,
            "nLayers": self.n_layers,
            "nHeads": self.n_heads,
            "nKvHeads": self.n_kv_heads,
            "dFf": self.d_ff,
            "ropeTheta": self.rope_theta,
            "normEpsilon": self.norm_epsilon,
            "dropout": self.dropout,
            "tieEmbeddings": self.tie_embeddings,
            "initializationStd": self.initialization_std,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Gen1Config":
        expected = {
            "schemaVersion",
            "architectureId",
            "vocabSize",
            "maxSequenceLength",
            "dModel",
            "nLayers",
            "nHeads",
            "nKvHeads",
            "dFf",
            "ropeTheta",
            "normEpsilon",
            "dropout",
            "tieEmbeddings",
            "initializationStd",
            "seed",
        }
        if not isinstance(value, Mapping):
            raise Gen1ConfigError("Gen1 config must be a JSON object")
        keys = set(value)
        if keys != expected:
            missing = sorted(expected - keys)
            extra = sorted(keys - expected)
            raise Gen1ConfigError(f"Gen1 config keys mismatch: missing={missing}, extra={extra}")
        if value["schemaVersion"] != CONFIG_SCHEMA_VERSION:
            raise Gen1ConfigError(
                f"unsupported Gen1 config schemaVersion {value['schemaVersion']!r}"
            )
        if value["architectureId"] != ARCHITECTURE_ID:
            raise Gen1ConfigError(
                f"unsupported architectureId {value['architectureId']!r}"
            )
        return cls(
            vocab_size=value["vocabSize"],
            max_sequence_length=value["maxSequenceLength"],
            d_model=value["dModel"],
            n_layers=value["nLayers"],
            n_heads=value["nHeads"],
            n_kv_heads=value["nKvHeads"],
            d_ff=value["dFf"],
            rope_theta=value["ropeTheta"],
            norm_epsilon=value["normEpsilon"],
            dropout=value["dropout"],
            tie_embeddings=value["tieEmbeddings"],
            initialization_std=value["initializationStd"],
            seed=value["seed"],
        )

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
