"""Fail-closed local inference runtime for approved VoltForge artifacts."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

import numpy as np

from model.artifact_registry import (
    ArtifactRegistryError,
    DEFAULT_REGISTRY_PATH,
    ResolvedArtifact,
    get_artifact_health,
    resolve_active_artifact,
)
from model.model import NumPyTransformer, TransformerConfig, softmax
from model.reasoning_llm import ElectronicsReasoningEngine
from model.tokenizer import VoltForgeTokenizer
from task_schema.compiler import compile_task_record
from task_schema.schema import CONTRACT_VERSION, validate_task_record


class VoltForgeInferenceEngine:
    """Load one explicitly approved model; never scan legacy artifacts."""

    def __init__(self, registry_path: Optional[str] = None):
        self.registry_path = Path(registry_path or DEFAULT_REGISTRY_PATH).resolve()
        self.artifacts_dir: Optional[str] = None
        self.artifact: Optional[ResolvedArtifact] = None
        self.tokenizer = VoltForgeTokenizer()
        self.config: Optional[TransformerConfig] = None
        self.model: Optional[NumPyTransformer] = None
        self.reasoning_engine: Optional[ElectronicsReasoningEngine] = None
        self.is_loaded = False
        self.status_code = "MODEL_NOT_LOADED"
        self.load_error = "The local model has not been loaded."
        self._load_approved_artifact()

    def _load_approved_artifact(self) -> None:
        try:
            artifact = resolve_active_artifact(self.registry_path)
            config = TransformerConfig.from_dict(dict(artifact.config))
            tokenizer = VoltForgeTokenizer()
            tokenizer.load(str(artifact.root))

            # Artifact validation completes before model allocation. Production
            # loading therefore cannot silently serve random or partial weights.
            model = NumPyTransformer(config, initialize_weights=False)
            model.load_weights(str(artifact.files["weights"]))
            if model.count_parameters() != artifact.parameter_count:
                raise ValueError(
                    f"Loaded model contains {model.count_parameters()} parameters; "
                    f"manifest declares {artifact.parameter_count}."
                )

            self.artifact = artifact
            self.artifacts_dir = str(artifact.root)
            self.config = config
            self.tokenizer = tokenizer
            self.model = model
            self.reasoning_engine = ElectronicsReasoningEngine(str(artifact.root))
            self.is_loaded = True
            self.status_code = "MODEL_ARTIFACT_READY"
            self.load_error = ""
        except ArtifactRegistryError as error:
            self.status_code = error.code
            self.load_error = error.message
        except Exception as error:
            self.status_code = "MODEL_RUNTIME_LOAD_FAILED"
            self.load_error = f"Approved local model failed runtime loading: {error}"

    def health(self) -> Dict[str, Any]:
        if self.is_loaded and self.artifact is not None:
            return self.artifact.health()
        health = get_artifact_health(self.registry_path)
        if health.get("ready"):
            health.update(
                {
                    "ready": False,
                    "state": "failed",
                    "code": self.status_code,
                    "message": self.load_error,
                }
            )
        return health

    def _require_loaded(self) -> None:
        if not self.is_loaded or self.model is None or self.config is None:
            raise RuntimeError(f"Local model unavailable [{self.status_code}]: {self.load_error}")

    def compute_confidence(self, prompt: str) -> float:
        """Estimate next-token confidence only for a validated loaded model."""
        if not self.is_loaded or self.model is None or self.config is None:
            return 0.0
        try:
            ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
            if not ids:
                return 0.0
            logits = self.model.forward(
                np.array(ids[-self.config.context_length :], dtype=np.int32)
            )
            return float(np.max(softmax(logits[-1])))
        except Exception:
            return 0.0

    def reason_and_solve(
        self,
        prompt: str,
        board_type: str = "ARDUINO_UNO",
        components: Optional[List[Dict[str, Any]]] = None,
        wires: Optional[List[Dict[str, Any]]] = None,
        code: Optional[str] = None,
        simulation_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._require_loaded()
        if self.reasoning_engine is None:
            raise RuntimeError("Approved model reasoning engine is unavailable.")
        return self.reasoning_engine.reason_and_solve(
            prompt=prompt,
            board_type=board_type,
            components=components,
            wires=wires,
            code=code,
            simulation_state=simulation_state,
        )

    def generate_neural_text(
        self, prompt: str, max_tokens: int = 120, temperature: float = 0.7
    ) -> str:
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        task_record = {
            "schemaVersion": 1,
            "contractVersion": CONTRACT_VERSION,
            "recordId": f"vf-task-v1-{prompt_hash[:24]}",
            "recordKind": "inference-request",
            "task": "domain_chat",
            "input": {
                "system": {
                    "type": "system",
                    "text": "You are VoltForge AI, a local electronics assistant.",
                    "policyVersion": "vf-system-policy-v1",
                },
                "user": {"type": "user", "text": prompt},
                "projectContext": {
                    "type": "project-context",
                    "projectId": "direct-local-inference",
                    "sourceProjectRevision": f"snapshot-sha256:{prompt_hash}",
                    "revisionSource": "derived-snapshot",
                    "payload": {},
                },
                "toolEvidence": [],
            },
            "metadata": {"sourceKind": "runtime", "sourceIds": []},
        }
        return self.generate_task_record(task_record, max_tokens=max_tokens, temperature=temperature)

    def generate_task_record(
        self,
        task_record: Dict[str, Any],
        max_tokens: int = 120,
        temperature: float = 0.7,
    ) -> str:
        self._require_loaded()
        assert self.model is not None
        validated = validate_task_record(task_record)
        if validated["recordKind"] != "inference-request":
            raise ValueError("Neural generation requires an inference-request task record.")
        formatted_prompt = compile_task_record(validated)
        bos_id = self.tokenizer.special_token_to_id.get("[BOS]", 2)
        eos_id = self.tokenizer.special_token_to_id.get("[EOS]", 3)
        input_ids = [bos_id] + self.tokenizer.encode(formatted_prompt)
        generated_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=40,
            top_p=0.9,
            eos_token_id=eos_id,
        )
        return self.tokenizer.decode(generated_ids)

    async def stream_reasoning_and_response(
        self,
        prompt: str,
        board_type: str = "ARDUINO_UNO",
        components: Optional[List[Dict[str, Any]]] = None,
        wires: Optional[List[Dict[str, Any]]] = None,
        code: Optional[str] = None,
        simulation_state: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        self._require_loaded()
        if self.reasoning_engine is None:
            raise RuntimeError("Approved model reasoning engine is unavailable.")
        async for chunk in self.reasoning_engine.stream_reasoning_and_response(
            prompt=prompt,
            board_type=board_type,
            components=components,
            wires=wires,
            code=code,
            simulation_state=simulation_state,
        ):
            yield chunk


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    engine = VoltForgeInferenceEngine()
    print(engine.health())
