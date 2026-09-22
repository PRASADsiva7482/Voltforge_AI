"""Local-only inference runtime for signed VoltForge Gen1 artifacts.

This module performs no downloads, training, retrieval, or service integration.
It is imported lazily by the process supervisor only when a signed approved
artifact is active, or explicitly by the offline experimental smoke tool.
"""

from __future__ import annotations

import codecs
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import threading
import time
from typing import Any, Iterator, Literal, Mapping

import torch

from model.identity import parse_artifact_id
from model.registry_manager import (
    DEFAULT_TRUST_STORE_PATH,
    RegistryManagerError,
    VerifiedArtifact,
    verify_artifact_directory,
)
from model.tokenizer import SPECIAL_TOKEN_TO_ID, TokenizerContractError, VoltForgeTokenizer

from .model import CheckpointContractError, VoltForgeGen1
from .optimization import (
    InferenceOptimizationProfile,
    OptimizationProfileError,
    VFAI017_SAFE_PROFILE,
    apply_model_optimizations,
    configure_torch_threads,
    profiles_from_policy,
)


GenerationMode = Literal["greedy", "sample"]


class RuntimeState(str, Enum):
    UNAVAILABLE = "unavailable"
    LOADING = "loading"
    READY = "ready"
    DEGRADED = "degraded"
    FAILED = "failed"


class LocalRuntimeError(RuntimeError):
    """Public-safe local runtime failure with a stable machine code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class GenerationCancelled(LocalRuntimeError):
    def __init__(self, generated_tokens: int):
        super().__init__("MODEL_GENERATION_CANCELLED", "Local generation was cancelled.")
        self.generated_tokens = generated_tokens


class GenerationDeadlineExceeded(LocalRuntimeError):
    def __init__(self, generated_tokens: int):
        super().__init__(
            "MODEL_GENERATION_DEADLINE_EXCEEDED", "Local generation exceeded its deadline."
        )
        self.generated_tokens = generated_tokens


class CancellationToken:
    """Thread-safe cooperative cancellation signal for one generation."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


@dataclass(frozen=True, slots=True)
class GenerationOptions:
    max_new_tokens: int = 16
    mode: GenerationMode = "greedy"
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    seed: int = 17

    def validate(self) -> None:
        if isinstance(self.max_new_tokens, bool) or not isinstance(self.max_new_tokens, int):
            raise LocalRuntimeError(
                "MODEL_GENERATION_OPTIONS_INVALID", "max_new_tokens must be an integer."
            )
        if self.max_new_tokens <= 0:
            raise LocalRuntimeError(
                "MODEL_GENERATION_OPTIONS_INVALID", "max_new_tokens must be positive."
            )
        if self.mode not in {"greedy", "sample"}:
            raise LocalRuntimeError(
                "MODEL_GENERATION_OPTIONS_INVALID", "mode must be greedy or sample."
            )
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float)):
            raise LocalRuntimeError(
                "MODEL_GENERATION_OPTIONS_INVALID", "temperature must be numeric."
            )
        if not 0.05 <= float(self.temperature) <= 2.0:
            raise LocalRuntimeError(
                "MODEL_GENERATION_OPTIONS_INVALID", "temperature must be between 0.05 and 2.0."
            )
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 0:
            raise LocalRuntimeError(
                "MODEL_GENERATION_OPTIONS_INVALID", "top_k must be a non-negative integer."
            )
        if isinstance(self.top_p, bool) or not isinstance(self.top_p, (int, float)):
            raise LocalRuntimeError(
                "MODEL_GENERATION_OPTIONS_INVALID", "top_p must be numeric."
            )
        if not 0.05 <= float(self.top_p) <= 1.0:
            raise LocalRuntimeError(
                "MODEL_GENERATION_OPTIONS_INVALID", "top_p must be between 0.05 and 1.0."
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise LocalRuntimeError(
                "MODEL_GENERATION_OPTIONS_INVALID", "seed must be a non-negative integer."
            )


@dataclass(frozen=True, slots=True)
class GenerationChunk:
    delta: str
    token_id: int | None
    token_index: int
    done: bool
    finish_reason: str | None = None


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    token_ids: tuple[int, ...]
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    elapsed_ms: float
    first_token_ms: float | None
    artifact_id: str
    device: str
    mode: GenerationMode
    seed: int


def select_local_device(requested: str = "auto") -> torch.device:
    normalized = requested.strip().lower()
    if normalized not in {"auto", "cpu", "cuda", "mps"}:
        raise LocalRuntimeError(
            "MODEL_DEVICE_INVALID", "Device must be auto, cpu, cuda, or mps."
        )
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise LocalRuntimeError("MODEL_DEVICE_UNAVAILABLE", "CUDA is not available locally.")
    if normalized == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise LocalRuntimeError("MODEL_DEVICE_UNAVAILABLE", "MPS is not available locally.")
    return torch.device(normalized)


class LocalGen1Runtime:
    """One signed Gen1 model/tokenizer instance with bounded generation."""

    def __init__(
        self,
        *,
        trust_store_path: Path = DEFAULT_TRUST_STORE_PATH,
        requested_device: str = "auto",
        optimization_profile: InferenceOptimizationProfile | None = None,
    ) -> None:
        self.trust_store_path = trust_store_path.resolve()
        self.requested_device = requested_device
        self.requested_optimization_profile = optimization_profile
        self._lifecycle_lock = threading.RLock()
        self._generation_lock = threading.Lock()
        self._state = RuntimeState.UNAVAILABLE
        self._code = "MODEL_RUNTIME_UNLOADED"
        self._message = "No local Gen1 artifact is loaded."
        self._artifact: VerifiedArtifact | None = None
        self._model: VoltForgeGen1 | None = None
        self._tokenizer: VoltForgeTokenizer | None = None
        self._policy: dict[str, Any] | None = None
        self._device: torch.device | None = None
        self._experimental = False
        self._optimization_profile: InferenceOptimizationProfile | None = None
        self._optimization_metadata: dict[str, Any] | None = None
        self._optimization_fallback_code: str | None = None
        self._previous_torch_threads: int | None = None
        self._active_requests = 0
        self._load_count = 0
        self._loaded_at_monotonic: float | None = None
        self._last_error_code: str | None = None

    @property
    def state(self) -> RuntimeState:
        with self._lifecycle_lock:
            return self._state

    @property
    def artifact_id(self) -> str | None:
        with self._lifecycle_lock:
            return self._artifact.artifact_id if self._artifact else None

    def load(self, artifact_root: str | Path, *, allow_experimental: bool = False) -> None:
        root = Path(artifact_root).resolve()
        with self._lifecycle_lock:
            if self._state == RuntimeState.LOADING:
                raise LocalRuntimeError(
                    "MODEL_RUNTIME_LOADING", "A local model load is already in progress."
                )
            if self._model is not None:
                if self._artifact is not None and self._artifact.root == root:
                    return
                raise LocalRuntimeError(
                    "MODEL_RUNTIME_ALREADY_LOADED",
                    "This process already owns a different local model instance.",
                )
            self._state = RuntimeState.LOADING
            self._code = "MODEL_RUNTIME_LOADING"
            self._message = "A signed local Gen1 artifact is loading."
            self._last_error_code = None

        try:
            artifact = verify_artifact_directory(
                root,
                trust_store_path=self.trust_store_path,
                require_activation=not allow_experimental,
            )
            manifest = dict(artifact.manifest)
            policy = self._read_declared_json(root, manifest, "generationPolicyPath")
            experimental = artifact.release_status != "approved"
            if experimental:
                if not allow_experimental:
                    raise LocalRuntimeError(
                        "MODEL_ARTIFACT_NOT_APPROVED",
                        "Experimental artifacts require an explicit offline-only load.",
                    )
                if policy.get("offlineExperimentalInferenceEnabled") is not True:
                    raise LocalRuntimeError(
                        "MODEL_EXPERIMENTAL_INFERENCE_DISABLED",
                        "Artifact policy does not permit offline experimental inference.",
                    )
                if policy.get("servingEnabled") is not False:
                    raise LocalRuntimeError(
                        "MODEL_ARTIFACT_POLICY_INVALID",
                        "An experimental artifact must not permit service serving.",
                    )
            elif policy.get("servingEnabled") is not True:
                raise LocalRuntimeError(
                    "MODEL_ARTIFACT_SERVING_DISABLED",
                    "Approved artifact policy does not permit serving.",
                )

            device = select_local_device(self.requested_device)
            if self.requested_optimization_profile is not None and not allow_experimental:
                raise LocalRuntimeError(
                    "MODEL_OPTIMIZATION_OVERRIDE_DENIED",
                    "Runtime optimization overrides are restricted to offline evaluation.",
                )
            optimization_policy = None
            if manifest.get("optimizationPolicyPath") is not None:
                optimization_policy = self._read_declared_json(
                    root, manifest, "optimizationPolicyPath"
                )
                selected_profile, fallback_profile = profiles_from_policy(
                    optimization_policy
                )
            else:
                selected_profile = VFAI017_SAFE_PROFILE
                fallback_profile = VFAI017_SAFE_PROFILE
            if self.requested_optimization_profile is not None:
                selected_profile = self.requested_optimization_profile
            model_metadata = manifest.get("model")
            tokenizer_metadata = manifest.get("tokenizer")
            if not isinstance(model_metadata, Mapping) or not isinstance(
                tokenizer_metadata, Mapping
            ):
                raise LocalRuntimeError(
                    "MODEL_ARTIFACT_RUNTIME_METADATA_INVALID",
                    "Artifact model/tokenizer runtime metadata is missing.",
                )
            model_directory = self._declared_parent(root, manifest, model_metadata.get("configPath"))
            checkpoint_manifest = model_metadata.get(
                "checkpointManifestPath", "model/manifest.json"
            )
            if self._declared_parent(root, manifest, checkpoint_manifest) != model_directory:
                raise LocalRuntimeError(
                    "MODEL_ARTIFACT_RUNTIME_METADATA_INVALID",
                    "Checkpoint files must share one declared model directory.",
                )
            if Path(str(checkpoint_manifest)).name != "manifest.json":
                raise LocalRuntimeError(
                    "MODEL_ARTIFACT_RUNTIME_METADATA_INVALID",
                    "Runtime checkpoint manifest must use the canonical manifest.json name.",
                )
            weights_path = model_metadata.get("weightsPath")
            if self._declared_parent(root, manifest, weights_path) != model_directory:
                raise LocalRuntimeError(
                    "MODEL_ARTIFACT_RUNTIME_METADATA_INVALID",
                    "Checkpoint files must share one declared model directory.",
                )

            tokenizer_manifest_path = tokenizer_metadata.get("manifestPath")
            tokenizer_directory = self._declared_parent(
                root, manifest, tokenizer_manifest_path
            )
            if Path(str(tokenizer_manifest_path)).name != "tokenizer_manifest.json":
                raise LocalRuntimeError(
                    "MODEL_ARTIFACT_RUNTIME_METADATA_INVALID",
                    "Tokenizer manifest must use the canonical tokenizer_manifest.json name.",
                )
            tokenizer = VoltForgeTokenizer(vocab_size=int(tokenizer_metadata["vocabSize"]))
            tokenizer.load(tokenizer_directory)
            previous_torch_threads = torch.get_num_threads()
            configure_torch_threads(selected_profile)
            model = VoltForgeGen1.from_checkpoint(
                model_directory,
                device=device,
                mmap_weights=selected_profile.mmap_weights,
            )
            if model.parameter_count() != artifact.parameter_count:
                raise LocalRuntimeError(
                    "MODEL_PARAMETER_COUNT_MISMATCH",
                    "Loaded Gen1 tensors do not match the signed parameter count.",
                )
            if model.config.vocab_size != tokenizer.vocab_size:
                raise LocalRuntimeError(
                    "MODEL_TOKENIZER_VOCAB_MISMATCH",
                    "Loaded model and tokenizer vocabulary sizes differ.",
                )
            if model.config.max_sequence_length != artifact.context_length:
                raise LocalRuntimeError(
                    "MODEL_CONTEXT_LENGTH_MISMATCH",
                    "Loaded model context does not match the signed artifact.",
                )
            self._validate_policy(policy, model.config.max_sequence_length)
            optimization_fallback_code = None
            try:
                optimization_metadata = apply_model_optimizations(
                    model, selected_profile, device
                )
            except OptimizationProfileError as error:
                if selected_profile == fallback_profile:
                    raise
                del model
                configure_torch_threads(fallback_profile)
                model = VoltForgeGen1.from_checkpoint(
                    model_directory,
                    device=device,
                    mmap_weights=fallback_profile.mmap_weights,
                )
                optimization_metadata = apply_model_optimizations(
                    model, fallback_profile, device
                )
                selected_profile = fallback_profile
                optimization_fallback_code = error.code
        except LocalRuntimeError as error:
            self._restore_torch_threads(locals().get("previous_torch_threads"))
            self._mark_failed(error.code, error.message)
            raise
        except OptimizationProfileError as error:
            self._restore_torch_threads(locals().get("previous_torch_threads"))
            self._mark_failed(error.code, error.message)
            raise LocalRuntimeError(error.code, error.message) from error
        except RegistryManagerError as error:
            self._restore_torch_threads(locals().get("previous_torch_threads"))
            self._mark_failed(error.code, error.message)
            raise LocalRuntimeError(error.code, error.message) from error
        except (CheckpointContractError, TokenizerContractError, KeyError, TypeError, ValueError) as error:
            self._restore_torch_threads(locals().get("previous_torch_threads"))
            code = getattr(error, "code", "MODEL_RUNTIME_LOAD_FAILED")
            message = "The signed Gen1 artifact could not be loaded by the local runtime."
            self._mark_failed(code, message)
            raise LocalRuntimeError(code, message) from error
        except Exception as error:
            self._restore_torch_threads(locals().get("previous_torch_threads"))
            self._mark_failed(
                "MODEL_RUNTIME_LOAD_FAILED",
                "The signed Gen1 artifact failed during local model materialization.",
            )
            raise LocalRuntimeError(
                "MODEL_RUNTIME_LOAD_FAILED",
                "The signed Gen1 artifact failed during local model materialization.",
            ) from error

        with self._lifecycle_lock:
            self._artifact = artifact
            self._model = model
            self._tokenizer = tokenizer
            self._policy = policy
            self._device = device
            self._experimental = experimental
            self._optimization_profile = selected_profile
            self._optimization_metadata = optimization_metadata
            self._optimization_fallback_code = optimization_fallback_code
            self._previous_torch_threads = previous_torch_threads
            self._state = RuntimeState.DEGRADED if experimental else RuntimeState.READY
            self._code = (
                "MODEL_RUNTIME_EXPERIMENTAL_OFFLINE"
                if experimental
                else "MODEL_RUNTIME_READY"
            )
            self._message = (
                "The signed experimental model is loaded for offline evaluation only."
                if experimental
                else "The signed approved local model runtime is ready."
            )
            self._load_count += 1
            self._loaded_at_monotonic = time.monotonic()

    def unload(self) -> None:
        with self._lifecycle_lock:
            if self._state == RuntimeState.LOADING:
                raise LocalRuntimeError(
                    "MODEL_RUNTIME_LOADING", "Cannot unload while model materialization is in progress."
                )
        with self._generation_lock:
            with self._lifecycle_lock:
                model = self._model
                device = self._device
                self._artifact = None
                self._model = None
                self._tokenizer = None
                self._policy = None
                self._device = None
                self._experimental = False
                self._optimization_profile = None
                self._optimization_metadata = None
                self._optimization_fallback_code = None
                self._active_requests = 0
                self._loaded_at_monotonic = None
                self._state = RuntimeState.UNAVAILABLE
                self._code = "MODEL_RUNTIME_UNLOADED"
                self._message = "No local Gen1 artifact is loaded."
                self._last_error_code = None
                previous_torch_threads = self._previous_torch_threads
                self._previous_torch_threads = None
            del model
            self._restore_torch_threads(previous_torch_threads)
            if device is not None and device.type == "cuda":
                torch.cuda.empty_cache()

    def generate(
        self,
        prompt: str,
        *,
        options: GenerationOptions | None = None,
        cancellation: CancellationToken | None = None,
        deadline_monotonic: float | None = None,
    ) -> GenerationResult:
        selected = options or GenerationOptions()
        started = time.perf_counter()
        chunks = []
        token_ids: list[int] = []
        prompt_tokens = 0
        first_token_ms: float | None = None
        finish_reason = "length"
        for chunk in self.stream_generate(
            prompt,
            options=selected,
            cancellation=cancellation,
            deadline_monotonic=deadline_monotonic,
        ):
            if chunk.token_id is not None:
                token_ids.append(chunk.token_id)
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - started) * 1_000
            if chunk.delta:
                chunks.append(chunk.delta)
            if chunk.done and chunk.finish_reason:
                finish_reason = chunk.finish_reason
                prompt_tokens = chunk.token_index
        # The final marker stores prompt count in token_index; generated tokens
        # remain explicitly counted from token IDs.
        return GenerationResult(
            text="".join(chunks),
            token_ids=tuple(token_ids),
            prompt_tokens=prompt_tokens,
            completion_tokens=len(token_ids),
            finish_reason=finish_reason,
            elapsed_ms=(time.perf_counter() - started) * 1_000,
            first_token_ms=first_token_ms,
            artifact_id=self.artifact_id or "",
            device=str(self._device),
            mode=selected.mode,
            seed=selected.seed,
        )

    def stream_generate(
        self,
        prompt: str,
        *,
        options: GenerationOptions | None = None,
        cancellation: CancellationToken | None = None,
        deadline_monotonic: float | None = None,
    ) -> Iterator[GenerationChunk]:
        selected = options or GenerationOptions()
        selected.validate()
        if not isinstance(prompt, str) or not prompt:
            raise LocalRuntimeError(
                "MODEL_PROMPT_INVALID", "Prompt must be a non-empty string."
            )
        token = cancellation or CancellationToken()
        self._acquire_generation(token, deadline_monotonic)
        with self._lifecycle_lock:
            model = self._model
            tokenizer = self._tokenizer
            policy = dict(self._policy or {})
            device = self._device
            experimental = self._experimental
            self._active_requests += 1
        if model is None or tokenizer is None or device is None:
            with self._lifecycle_lock:
                self._active_requests -= 1
            self._generation_lock.release()
            raise LocalRuntimeError(
                "MODEL_RUNTIME_UNAVAILABLE", "No local Gen1 artifact is loaded."
            )
        try:
            yield from self._stream_locked(
                prompt,
                selected,
                token,
                deadline_monotonic,
                model,
                tokenizer,
                policy,
                device,
            )
        except (GenerationCancelled, GenerationDeadlineExceeded, LocalRuntimeError):
            raise
        except Exception as error:
            with self._lifecycle_lock:
                self._state = RuntimeState.DEGRADED
                self._code = "MODEL_RUNTIME_GENERATION_FAILED"
                self._message = "The local model encountered a generation failure."
                self._last_error_code = "MODEL_RUNTIME_GENERATION_FAILED"
            raise LocalRuntimeError(
                "MODEL_RUNTIME_GENERATION_FAILED",
                "The local model encountered a generation failure.",
            ) from error
        finally:
            with self._lifecycle_lock:
                self._active_requests -= 1
                if (
                    not experimental
                    and self._model is not None
                    and self._last_error_code is None
                ):
                    self._state = RuntimeState.READY
            self._generation_lock.release()

    def health(self) -> dict[str, Any]:
        with self._lifecycle_lock:
            artifact = self._artifact
            identity = parse_artifact_id(artifact.artifact_id).health_fields() if artifact else {
                "productName": "VoltForge AI",
                "familyName": "VoltForge Domain Language Model",
                "familySlug": "vfdlm",
                "artifactId": None,
                "generation": None,
                "deploymentProfile": None,
                "semanticVersion": None,
            }
            identity.update(
                {
                    "ready": self._state == RuntimeState.READY,
                    "runtimeOperational": self._model is not None,
                    "state": self._state.value,
                    "code": self._code,
                    "message": self._message,
                    "runtime": "pytorch-gen1-v1",
                    "device": str(self._device) if self._device else None,
                    "dtype": (
                        str(next(self._model.parameters()).dtype).removeprefix("torch.")
                        if self._model is not None
                        else None
                    ),
                    "parameterCount": artifact.parameter_count if artifact else None,
                    "contextLength": artifact.context_length if artifact else None,
                    "releaseStatus": artifact.release_status if artifact else "none",
                    "activationEligible": artifact.activation_eligible if artifact else False,
                    "experimentalOffline": self._experimental,
                    "activeRequests": self._active_requests,
                    "loadCount": self._load_count,
                    "lastErrorCode": self._last_error_code,
                    "generationNetworkAccess": False,
                    "trainingAvailableAtRuntime": False,
                    "downloadAvailableAtRuntime": False,
                    "optimizationProfile": (
                        self._optimization_profile.as_dict()
                        if self._optimization_profile is not None
                        else None
                    ),
                    "optimizationFallbackCode": self._optimization_fallback_code,
                }
            )
            if self._optimization_metadata is not None:
                identity["optimizationRuntime"] = dict(self._optimization_metadata)
            if self._policy is not None:
                identity["generationBudgets"] = {
                    "maximumInputTokens": self._policy["maximumInputTokens"],
                    "maximumNewTokens": self._policy["maximumNewTokens"],
                    "samplingEnabled": self._policy["samplingEnabled"],
                    "streamingEnabled": self._policy["streamingEnabled"],
                }
            return identity

    def _stream_locked(
        self,
        prompt: str,
        options: GenerationOptions,
        cancellation: CancellationToken,
        deadline_monotonic: float | None,
        model: VoltForgeGen1,
        tokenizer: VoltForgeTokenizer,
        policy: Mapping[str, Any],
        device: torch.device,
    ) -> Iterator[GenerationChunk]:
        if options.mode == "sample" and policy["samplingEnabled"] is not True:
            raise LocalRuntimeError(
                "MODEL_SAMPLING_DISABLED", "Artifact generation policy disables sampling."
            )
        if policy["streamingEnabled"] is not True:
            raise LocalRuntimeError(
                "MODEL_STREAMING_DISABLED", "Artifact generation policy disables streaming."
            )
        prompt_ids = tokenizer.encode(prompt, add_bos=True, allowed_special=False)
        maximum_input = int(policy["maximumInputTokens"])
        maximum_new = int(policy["maximumNewTokens"])
        if len(prompt_ids) > maximum_input:
            raise LocalRuntimeError(
                "MODEL_INPUT_TOKEN_BUDGET_EXCEEDED",
                f"Prompt requires {len(prompt_ids)} tokens; the artifact limit is {maximum_input}.",
            )
        generation_limit = min(
            options.max_new_tokens,
            maximum_new,
            model.config.max_sequence_length - len(prompt_ids),
        )
        if generation_limit <= 0:
            raise LocalRuntimeError(
                "MODEL_CONTEXT_BUDGET_EXHAUSTED", "Prompt leaves no context for generation."
            )
        self._check_stop(cancellation, deadline_monotonic, 0)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        random = torch.Generator(device="cpu")
        random.manual_seed(options.seed)
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        cache = None
        finish_reason = "length"
        generated = 0
        generated_context = list(prompt_ids)
        profile = self._optimization_profile or VFAI017_SAFE_PROFILE
        with torch.inference_mode():
            output = model(input_ids, use_cache=profile.kv_cache_enabled)
            cache = output.past_key_values
            logits = output.logits[0, -1]
            for index in range(generation_limit):
                self._check_stop(cancellation, deadline_monotonic, generated)
                next_token = self._select_token(logits, options, random)
                self._check_stop(cancellation, deadline_monotonic, generated)
                generated += 1
                if next_token == SPECIAL_TOKEN_TO_ID["[EOS]"]:
                    finish_reason = "eos"
                    break
                delta = decoder.decode(
                    tokenizer.decode_bytes([next_token], skip_special=True), final=False
                )
                yield GenerationChunk(
                    delta=delta,
                    token_id=next_token,
                    token_index=index,
                    done=False,
                )
                if index + 1 >= generation_limit:
                    break
                generated_context.append(next_token)
                token_tensor = torch.tensor(
                    [[next_token]] if profile.kv_cache_enabled else [generated_context],
                    dtype=torch.long,
                    device=device,
                )
                output = model(
                    token_tensor,
                    past_key_values=cache if profile.kv_cache_enabled else None,
                    use_cache=profile.kv_cache_enabled,
                )
                cache = output.past_key_values
                logits = output.logits[0, -1]
        tail = decoder.decode(b"", final=True)
        if tail:
            yield GenerationChunk(
                delta=tail,
                token_id=None,
                token_index=generated,
                done=False,
            )
        yield GenerationChunk(
            delta="",
            token_id=None,
            token_index=len(prompt_ids),
            done=True,
            finish_reason=finish_reason,
        )

    @staticmethod
    def _select_token(
        logits: torch.Tensor,
        options: GenerationOptions,
        random: torch.Generator,
    ) -> int:
        scores = logits.detach().float().cpu()
        if not torch.all(torch.isfinite(scores)):
            raise LocalRuntimeError(
                "MODEL_LOGITS_INVALID", "The model produced non-finite logits."
            )
        if options.mode == "greedy":
            return int(torch.argmax(scores).item())
        scores = scores / float(options.temperature)
        if options.top_k > 0 and options.top_k < scores.numel():
            threshold = torch.topk(scores, options.top_k).values[-1]
            scores = scores.masked_fill(scores < threshold, float("-inf"))
        probabilities = torch.softmax(scores, dim=-1)
        if options.top_p < 1.0:
            sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
            cumulative = torch.cumsum(sorted_probabilities, dim=-1)
            remove = cumulative - sorted_probabilities >= float(options.top_p)
            sorted_probabilities = sorted_probabilities.masked_fill(remove, 0.0)
            probabilities = torch.zeros_like(probabilities).scatter(
                0, sorted_indices, sorted_probabilities
            )
            probabilities = probabilities / probabilities.sum().clamp_min(1e-12)
        return int(torch.multinomial(probabilities, 1, generator=random).item())

    def _acquire_generation(
        self,
        cancellation: CancellationToken,
        deadline_monotonic: float | None,
    ) -> None:
        while not self._generation_lock.acquire(timeout=0.05):
            self._check_stop(cancellation, deadline_monotonic, 0)

    @staticmethod
    def _check_stop(
        cancellation: CancellationToken,
        deadline_monotonic: float | None,
        generated_tokens: int,
    ) -> None:
        if cancellation.cancelled:
            raise GenerationCancelled(generated_tokens)
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            raise GenerationDeadlineExceeded(generated_tokens)

    @staticmethod
    def _validate_policy(policy: Mapping[str, Any], context_length: int) -> None:
        required = {
            "maximumInputTokens": int,
            "maximumNewTokens": int,
            "samplingEnabled": bool,
            "streamingEnabled": bool,
            "promptLoggingAllowed": bool,
        }
        for key, expected_type in required.items():
            value = policy.get(key)
            if not isinstance(value, expected_type) or (
                expected_type is int and isinstance(value, bool)
            ):
                raise LocalRuntimeError(
                    "MODEL_ARTIFACT_POLICY_INVALID", f"Generation policy field {key} is invalid."
                )
        if policy["promptLoggingAllowed"] is not False:
            raise LocalRuntimeError(
                "MODEL_ARTIFACT_POLICY_INVALID", "Generation policy must prohibit prompt logging."
            )
        if (
            policy["maximumInputTokens"] <= 0
            or policy["maximumNewTokens"] <= 0
            or policy["maximumInputTokens"] + policy["maximumNewTokens"] > context_length
        ):
            raise LocalRuntimeError(
                "MODEL_ARTIFACT_POLICY_INVALID", "Generation budgets exceed model context."
            )

    @staticmethod
    def _read_declared_json(root: Path, manifest: Mapping[str, Any], key: str) -> dict[str, Any]:
        path = LocalGen1Runtime._declared_path(root, manifest, manifest.get(key))
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise LocalRuntimeError(
                "MODEL_ARTIFACT_RUNTIME_METADATA_INVALID",
                f"Declared runtime JSON is invalid: {path.name}.",
            ) from error
        if not isinstance(value, dict):
            raise LocalRuntimeError(
                "MODEL_ARTIFACT_RUNTIME_METADATA_INVALID", "Runtime JSON must be an object."
            )
        return value

    @staticmethod
    def _declared_parent(root: Path, manifest: Mapping[str, Any], relative: Any) -> Path:
        return LocalGen1Runtime._declared_path(root, manifest, relative).parent

    @staticmethod
    def _declared_path(root: Path, manifest: Mapping[str, Any], relative: Any) -> Path:
        if not isinstance(relative, str):
            raise LocalRuntimeError(
                "MODEL_ARTIFACT_RUNTIME_METADATA_INVALID", "Runtime path is missing."
            )
        declared = {
            item.get("path") for item in manifest.get("files", []) if isinstance(item, Mapping)
        }
        if relative not in declared:
            raise LocalRuntimeError(
                "MODEL_ARTIFACT_RUNTIME_METADATA_INVALID",
                f"Runtime path is not in the signed file inventory: {relative}.",
            )
        path = (root / Path(*relative.split("/"))).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as error:
            raise LocalRuntimeError(
                "MODEL_ARTIFACT_RUNTIME_METADATA_INVALID", "Runtime path escapes artifact root."
            ) from error
        return path

    def _mark_failed(self, code: str, message: str) -> None:
        with self._lifecycle_lock:
            self._artifact = None
            self._model = None
            self._tokenizer = None
            self._policy = None
            self._device = None
            self._experimental = False
            self._optimization_profile = None
            self._optimization_metadata = None
            self._optimization_fallback_code = None
            self._previous_torch_threads = None
            self._state = RuntimeState.FAILED
            self._code = code
            self._message = message
            self._last_error_code = code

    @staticmethod
    def _restore_torch_threads(previous: Any) -> None:
        if isinstance(previous, int) and not isinstance(previous, bool) and previous > 0:
            torch.set_num_threads(previous)
