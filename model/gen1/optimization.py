"""Portable, explicit Gen1 inference optimization profiles.

Profiles are data contracts, not ambient tuning.  They are selected from a
signed artifact policy, expose every behavior-changing switch, and retain an
FP32/manual-attention fallback.  Quantization is local weight-only inference;
it never changes or rewrites the signed source checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence, TypeAlias

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .model import AttentionBackend, VoltForgeGen1


QuantizationMode: TypeAlias = Literal["fp32", "int8-weight-only", "int4-weight-only"]


class OptimizationProfileError(RuntimeError):
    """Stable local failure for an invalid or unavailable optimization."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class InferenceOptimizationProfile:
    profile_id: str
    kv_cache_enabled: bool
    attention_backend: AttentionBackend
    torch_threads: int
    mmap_weights: bool
    quantization: QuantizationMode
    dynamic_batch_max_size: int = 1

    def validate(self) -> None:
        if not self.profile_id or len(self.profile_id) > 96:
            raise OptimizationProfileError(
                "MODEL_OPTIMIZATION_PROFILE_INVALID", "Optimization profile ID is invalid."
            )
        if self.attention_backend not in {"manual", "sdpa"}:
            raise OptimizationProfileError(
                "MODEL_OPTIMIZATION_PROFILE_INVALID", "Attention backend is invalid."
            )
        if (
            isinstance(self.torch_threads, bool)
            or not isinstance(self.torch_threads, int)
            or not 1 <= self.torch_threads <= 64
        ):
            raise OptimizationProfileError(
                "MODEL_OPTIMIZATION_PROFILE_INVALID",
                "Torch thread count must be between 1 and 64.",
            )
        if self.quantization not in {"fp32", "int8-weight-only", "int4-weight-only"}:
            raise OptimizationProfileError(
                "MODEL_OPTIMIZATION_PROFILE_INVALID", "Quantization mode is invalid."
            )
        if (
            isinstance(self.dynamic_batch_max_size, bool)
            or not isinstance(self.dynamic_batch_max_size, int)
            or not 1 <= self.dynamic_batch_max_size <= 32
        ):
            raise OptimizationProfileError(
                "MODEL_OPTIMIZATION_PROFILE_INVALID",
                "Dynamic batch maximum must be between 1 and 32.",
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "profileId": self.profile_id,
            "kvCacheEnabled": self.kv_cache_enabled,
            "attentionBackend": self.attention_backend,
            "torchThreads": self.torch_threads,
            "mmapWeights": self.mmap_weights,
            "quantization": self.quantization,
            "dynamicBatchMaxSize": self.dynamic_batch_max_size,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "InferenceOptimizationProfile":
        expected = {
            "profileId",
            "kvCacheEnabled",
            "attentionBackend",
            "torchThreads",
            "mmapWeights",
            "quantization",
            "dynamicBatchMaxSize",
        }
        if set(value) != expected:
            raise OptimizationProfileError(
                "MODEL_OPTIMIZATION_PROFILE_INVALID",
                "Optimization profile fields do not match the runtime contract.",
            )
        if (
            not isinstance(value["profileId"], str)
            or not isinstance(value["attentionBackend"], str)
            or not isinstance(value["quantization"], str)
            or not isinstance(value["kvCacheEnabled"], bool)
            or not isinstance(value["mmapWeights"], bool)
        ):
            raise OptimizationProfileError(
                "MODEL_OPTIMIZATION_PROFILE_INVALID",
                "Optimization profile value types are invalid.",
            )
        profile = cls(
            profile_id=value["profileId"],
            kv_cache_enabled=value["kvCacheEnabled"],
            attention_backend=value["attentionBackend"],
            torch_threads=value["torchThreads"],
            mmap_weights=value["mmapWeights"],
            quantization=value["quantization"],
            dynamic_batch_max_size=value["dynamicBatchMaxSize"],
        )
        profile.validate()
        return profile


REFERENCE_PROFILE = InferenceOptimizationProfile(
    profile_id="vfai018-reference-fp32-manual-no-kv",
    kv_cache_enabled=False,
    attention_backend="manual",
    torch_threads=4,
    mmap_weights=False,
    quantization="fp32",
    dynamic_batch_max_size=1,
)

VFAI017_SAFE_PROFILE = InferenceOptimizationProfile(
    profile_id="vfai017-safe-fp32-manual-kv",
    kv_cache_enabled=True,
    attention_backend="manual",
    torch_threads=4,
    mmap_weights=False,
    quantization="fp32",
    dynamic_batch_max_size=1,
)


def profiles_from_policy(
    policy: Mapping[str, Any],
) -> tuple[InferenceOptimizationProfile, InferenceOptimizationProfile]:
    if policy.get("schemaVersion") != 1:
        raise OptimizationProfileError(
            "MODEL_OPTIMIZATION_POLICY_INVALID", "Optimization policy schema is unsupported."
        )
    values = policy.get("profiles")
    if not isinstance(values, list) or not values:
        raise OptimizationProfileError(
            "MODEL_OPTIMIZATION_POLICY_INVALID", "Optimization policy has no profiles."
        )
    profiles = [
        InferenceOptimizationProfile.from_mapping(item)
        for item in values
        if isinstance(item, Mapping)
    ]
    if len(profiles) != len(values) or len({item.profile_id for item in profiles}) != len(
        profiles
    ):
        raise OptimizationProfileError(
            "MODEL_OPTIMIZATION_POLICY_INVALID", "Optimization profile IDs are invalid."
        )
    by_id = {item.profile_id: item for item in profiles}
    try:
        selected = by_id[policy["selectedProfileId"]]
        fallback = by_id[policy["fallbackProfileId"]]
    except (KeyError, TypeError) as error:
        raise OptimizationProfileError(
            "MODEL_OPTIMIZATION_POLICY_INVALID",
            "Selected or fallback optimization profile is missing.",
        ) from error
    if (
        fallback.quantization != "fp32"
        or fallback.attention_backend != "manual"
        or fallback.dynamic_batch_max_size != 1
    ):
        raise OptimizationProfileError(
            "MODEL_OPTIMIZATION_POLICY_INVALID",
            "Fallback must be an FP32 manual-attention single-request profile.",
        )
    return selected, fallback


def validate_profile_for_device(
    profile: InferenceOptimizationProfile, device: torch.device
) -> None:
    profile.validate()
    if profile.attention_backend == "sdpa" and not hasattr(
        F, "scaled_dot_product_attention"
    ):
        raise OptimizationProfileError(
            "MODEL_OPTIMIZATION_UNAVAILABLE", "SDPA is unavailable in the local Torch runtime."
        )
    if profile.quantization != "fp32" and device.type != "cpu":
        raise OptimizationProfileError(
            "MODEL_OPTIMIZATION_UNAVAILABLE",
            "Portable weight-only int8/int4 inference is currently CPU-only.",
        )


def configure_torch_threads(profile: InferenceOptimizationProfile) -> int:
    profile.validate()
    previous = torch.get_num_threads()
    torch.set_num_threads(profile.torch_threads)
    return previous


class WeightOnlyLinear(nn.Module):
    """Deterministic per-output-channel symmetric int8/int4 linear layer."""

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        bits: Literal[4, 8],
        quantized_weight: Tensor,
        scales: Tensor,
        bias: Tensor | None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.register_buffer("quantized_weight", quantized_weight.contiguous())
        self.register_buffer("scales", scales.contiguous())
        self.register_buffer("bias", None if bias is None else bias.detach().float().contiguous())

    @classmethod
    def from_linear(cls, linear: nn.Linear, bits: Literal[4, 8]) -> "WeightOnlyLinear":
        weight = linear.weight.detach().float().cpu()
        maximum = weight.abs().amax(dim=1).clamp_min(torch.finfo(torch.float32).eps)
        quantization_max = 127 if bits == 8 else 7
        scales = maximum / quantization_max
        quantized = torch.round(weight / scales[:, None]).clamp(
            -quantization_max, quantization_max
        ).to(torch.int8)
        if bits == 4:
            if quantized.shape[1] % 2:
                quantized = F.pad(quantized, (0, 1))
            unsigned = (quantized + 8).to(torch.uint8)
            quantized = unsigned[:, 0::2] | (unsigned[:, 1::2] << 4)
        return cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bits=bits,
            quantized_weight=quantized,
            scales=scales,
            bias=linear.bias,
        )

    def _dequantized_weight(self, dtype: torch.dtype, device: torch.device) -> Tensor:
        if self.bits == 8:
            quantized = self.quantized_weight
        else:
            packed = self.quantized_weight
            low = (packed & 0x0F).to(torch.int8) - 8
            high = ((packed >> 4) & 0x0F).to(torch.int8) - 8
            quantized = torch.stack((low, high), dim=-1).flatten(start_dim=1)[
                :, : self.in_features
            ]
        return quantized.to(device=device, dtype=dtype) * self.scales.to(
            device=device, dtype=dtype
        )[:, None]

    def forward(self, value: Tensor) -> Tensor:
        weight = self._dequantized_weight(value.dtype, value.device)
        bias = None if self.bias is None else self.bias.to(device=value.device, dtype=value.dtype)
        return F.linear(value, weight, bias)


def _replace_linears(module: nn.Module, bits: Literal[4, 8]) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, WeightOnlyLinear.from_linear(child, bits))
            replaced += 1
        else:
            replaced += _replace_linears(child, bits)
    return replaced


def apply_model_optimizations(
    model: VoltForgeGen1,
    profile: InferenceOptimizationProfile,
    device: torch.device,
) -> dict[str, Any]:
    validate_profile_for_device(profile, device)
    model.set_attention_backend(profile.attention_backend)
    replaced = 0
    if profile.quantization == "int8-weight-only":
        replaced = _replace_linears(model, 8)
    elif profile.quantization == "int4-weight-only":
        replaced = _replace_linears(model, 4)
    return {
        "profileId": profile.profile_id,
        "quantizedLinearCount": replaced,
        "residentTensorBytes": model_tensor_bytes(model),
    }


def model_tensor_bytes(model: nn.Module) -> int:
    """Count unique resident parameter/buffer tensor storage bytes."""

    seen: set[tuple[str, int]] = set()
    total = 0
    tensors: Sequence[Tensor] = tuple(model.parameters()) + tuple(model.buffers())
    for tensor in tensors:
        if tensor.device.type == "meta":
            continue
        storage = tensor.untyped_storage()
        key = (str(tensor.device), int(storage.data_ptr()))
        if key in seen:
            continue
        seen.add(key)
        total += storage.nbytes()
    return total
