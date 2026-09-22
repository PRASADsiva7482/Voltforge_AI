"""Immutable configuration for reproducible Gen1 optimization runs."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Mapping

from model.gen1.config import Gen1Config


TRAINING_CONFIG_SCHEMA_VERSION = 1
TRAINING_PIPELINE_ID = "vfdlm-gen1-training-v1"
SUPPORTED_PRECISIONS = frozenset({"auto", "float32", "bfloat16", "float16"})
SUPPORTED_DEVICES = frozenset({"auto", "cpu", "cuda"})


class TrainingConfigError(ValueError):
    """Raised when a training configuration is incomplete or unsafe."""


@dataclass(frozen=True, slots=True)
class Gen1TrainingConfig:
    model: Gen1Config = field(default_factory=Gen1Config)
    seed: int = 41
    max_steps: int = 100
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    peak_learning_rate: float = 3e-4
    minimum_learning_rate: float = 3e-5
    warmup_steps: int = 10
    beta1: float = 0.9
    beta2: float = 0.95
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.1
    max_gradient_norm: float = 1.0
    validation_interval: int = 25
    checkpoint_interval: int = 25
    precision: str = "auto"
    device: str = "auto"
    deterministic_algorithms: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.model, Gen1Config):
            raise TrainingConfigError("model must be a Gen1Config")
        integer_fields = {
            "seed": self.seed,
            "max_steps": self.max_steps,
            "micro_batch_size": self.micro_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "validation_interval": self.validation_interval,
            "checkpoint_interval": self.checkpoint_interval,
        }
        for name, value in integer_fields.items():
            minimum = 0 if name == "seed" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise TrainingConfigError(f"{name} must be an integer >= {minimum}")
        if isinstance(self.warmup_steps, bool) or not isinstance(self.warmup_steps, int):
            raise TrainingConfigError("warmup_steps must be a non-negative integer")
        if self.warmup_steps < 0 or self.warmup_steps > self.max_steps:
            raise TrainingConfigError("warmup_steps must be between 0 and max_steps")
        numeric_fields = {
            "peak_learning_rate": self.peak_learning_rate,
            "minimum_learning_rate": self.minimum_learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "adam_epsilon": self.adam_epsilon,
            "weight_decay": self.weight_decay,
            "max_gradient_norm": self.max_gradient_norm,
        }
        for name, value in numeric_fields.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TrainingConfigError(f"{name} must be numeric")
        if self.peak_learning_rate <= 0 or self.minimum_learning_rate < 0:
            raise TrainingConfigError("learning rates must be non-negative and peak must be positive")
        if self.minimum_learning_rate > self.peak_learning_rate:
            raise TrainingConfigError("minimum_learning_rate cannot exceed peak_learning_rate")
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise TrainingConfigError("Adam beta values must be in [0, 1)")
        if self.adam_epsilon <= 0 or self.weight_decay < 0 or self.max_gradient_norm <= 0:
            raise TrainingConfigError(
                "adam_epsilon and max_gradient_norm must be positive; weight_decay cannot be negative"
            )
        if self.precision not in SUPPORTED_PRECISIONS:
            raise TrainingConfigError(f"unsupported precision policy: {self.precision}")
        if self.device not in SUPPORTED_DEVICES:
            raise TrainingConfigError(f"unsupported device policy: {self.device}")
        if not isinstance(self.deterministic_algorithms, bool):
            raise TrainingConfigError("deterministic_algorithms must be a boolean")

    def learning_rate_for_step(self, step_index: int) -> float:
        """Return the LR for one zero-based optimizer update."""

        if step_index < 0:
            raise TrainingConfigError("step_index cannot be negative")
        if self.warmup_steps and step_index < self.warmup_steps:
            return self.peak_learning_rate * (step_index + 1) / self.warmup_steps
        cosine_steps = self.max_steps - self.warmup_steps
        if cosine_steps <= 1:
            progress = 1.0
        else:
            progress = min(1.0, (step_index - self.warmup_steps) / (cosine_steps - 1))
        if progress <= 0.0:
            return self.peak_learning_rate
        if progress >= 1.0:
            return self.minimum_learning_rate
        import math

        return self.minimum_learning_rate + 0.5 * (
            self.peak_learning_rate - self.minimum_learning_rate
        ) * (1.0 + math.cos(math.pi * progress))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": TRAINING_CONFIG_SCHEMA_VERSION,
            "pipelineId": TRAINING_PIPELINE_ID,
            "model": self.model.to_dict(),
            "seed": self.seed,
            "maxSteps": self.max_steps,
            "microBatchSize": self.micro_batch_size,
            "gradientAccumulationSteps": self.gradient_accumulation_steps,
            "peakLearningRate": self.peak_learning_rate,
            "minimumLearningRate": self.minimum_learning_rate,
            "warmupSteps": self.warmup_steps,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "adamEpsilon": self.adam_epsilon,
            "weightDecay": self.weight_decay,
            "maxGradientNorm": self.max_gradient_norm,
            "validationInterval": self.validation_interval,
            "checkpointInterval": self.checkpoint_interval,
            "precision": self.precision,
            "device": self.device,
            "deterministicAlgorithms": self.deterministic_algorithms,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Gen1TrainingConfig":
        expected = {
            "schemaVersion",
            "pipelineId",
            "model",
            "seed",
            "maxSteps",
            "microBatchSize",
            "gradientAccumulationSteps",
            "peakLearningRate",
            "minimumLearningRate",
            "warmupSteps",
            "beta1",
            "beta2",
            "adamEpsilon",
            "weightDecay",
            "maxGradientNorm",
            "validationInterval",
            "checkpointInterval",
            "precision",
            "device",
            "deterministicAlgorithms",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            keys = set(value) if isinstance(value, Mapping) else set()
            raise TrainingConfigError(
                f"training config keys mismatch: missing={sorted(expected - keys)}, "
                f"extra={sorted(keys - expected)}"
            )
        if value["schemaVersion"] != TRAINING_CONFIG_SCHEMA_VERSION:
            raise TrainingConfigError("unsupported training config schemaVersion")
        if value["pipelineId"] != TRAINING_PIPELINE_ID:
            raise TrainingConfigError("unsupported training pipelineId")
        return cls(
            model=Gen1Config.from_dict(value["model"]),
            seed=value["seed"],
            max_steps=value["maxSteps"],
            micro_batch_size=value["microBatchSize"],
            gradient_accumulation_steps=value["gradientAccumulationSteps"],
            peak_learning_rate=value["peakLearningRate"],
            minimum_learning_rate=value["minimumLearningRate"],
            warmup_steps=value["warmupSteps"],
            beta1=value["beta1"],
            beta2=value["beta2"],
            adam_epsilon=value["adamEpsilon"],
            weight_decay=value["weightDecay"],
            max_gradient_norm=value["maxGradientNorm"],
            validation_interval=value["validationInterval"],
            checkpoint_interval=value["checkpointInterval"],
            precision=value["precision"],
            device=value["device"],
            deterministic_algorithms=value["deterministicAlgorithms"],
        )

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
