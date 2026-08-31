"""Torch-free process supervisor for the lazily imported Gen1 runtime."""

from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path
import threading
from typing import Any

from model.artifact_registry import get_artifact_health
from model.registry_manager import (
    DEFAULT_REGISTRY_PATH,
    DEFAULT_TRUST_STORE_PATH,
    RegistryManagerError,
    artifact_root_from_entry,
    verify_registry,
)
from observability import observability


logger = logging.getLogger("voltforge-ai.model-runtime")


class ModelRuntimeService:
    """Own at most one active model runtime for the current process."""

    def __init__(
        self,
        *,
        registry_path: Path = DEFAULT_REGISTRY_PATH,
        trust_store_path: Path = DEFAULT_TRUST_STORE_PATH,
        requested_device: str = "auto",
    ) -> None:
        self.registry_path = registry_path.resolve()
        self.trust_store_path = trust_store_path.resolve()
        self.requested_device = requested_device
        self._lock = threading.RLock()
        self._start_stop_lock = threading.Lock()
        self._runtime: Any | None = None
        self._state = "unavailable"
        self._code = "MODEL_RUNTIME_NOT_STARTED"
        self._message = "The local model runtime has not started."
        self._registry_revision: int | None = None
        self._registry_sha256: str | None = None
        self._active_artifact_id: str | None = None

    def start(self) -> dict[str, Any]:
        """Load only a signed active approved artifact; never train or download."""

        with self._start_stop_lock:
            return self._start_once()

    def _start_once(self) -> dict[str, Any]:

        with self._lock:
            if self._runtime is not None:
                return self.health()
        try:
            registry = verify_registry(
                self.registry_path,
                trust_store_path=self.trust_store_path,
            )
        except RegistryManagerError as error:
            observability.record_model_load_failure(error.code)
            self._set_failure(error.code, error.message)
            return self.health()

        active_id = registry.get("activeArtifactId")
        with self._lock:
            self._registry_revision = registry["revision"]
            self._registry_sha256 = registry["registrySha256"]
            self._active_artifact_id = active_id
        if active_id is None:
            observability.record_model_load_failure("NO_APPROVED_MODEL_ARTIFACT")
            with self._lock:
                self._state = "unavailable"
                self._code = "NO_APPROVED_MODEL_ARTIFACT"
                self._message = str(registry["reason"])
            logger.info(
                "Model runtime unavailable registryRevision=%s activeArtifactId=None",
                registry["revision"],
            )
            return self.health()

        entry = next(
            item for item in registry["artifacts"] if item.get("artifactId") == active_id
        )
        with self._lock:
            self._state = "loading"
            self._code = "MODEL_RUNTIME_LOADING"
            self._message = "The signed active model is loading."
        logger.info(
            "Model runtime loading artifactId=%s registryRevision=%s",
            active_id,
            registry["revision"],
        )
        try:
            runtime_module = importlib.import_module("model.gen1.runtime")
            runtime = runtime_module.LocalGen1Runtime(
                trust_store_path=self.trust_store_path,
                requested_device=self.requested_device,
            )
            runtime.load(artifact_root_from_entry(self.registry_path, entry))
        except Exception as error:
            code = getattr(error, "code", "MODEL_RUNTIME_LOAD_FAILED")
            observability.record_model_load_failure(code)
            self._set_failure(
                str(code), "The signed active local model failed to load."
            )
            logger.error(
                "Model runtime load failed artifactId=%s code=%s",
                active_id,
                code,
            )
            return self.health()
        with self._lock:
            self._runtime = runtime
            runtime_health = runtime.health()
            self._state = str(runtime_health["state"])
            self._code = str(runtime_health["code"])
            self._message = str(runtime_health["message"])
        logger.info(
            "Model runtime ready artifactId=%s device=%s registryRevision=%s",
            active_id,
            runtime.health().get("device"),
            registry["revision"],
        )
        observability.record_model_load_success(active_id)
        return self.health()

    def stop(self) -> None:
        with self._start_stop_lock:
            self._stop_once()

    def _stop_once(self) -> None:
        with self._lock:
            runtime = self._runtime
            self._runtime = None
        if runtime is not None:
            try:
                runtime.unload()
            except Exception:
                logger.exception(
                    "Model runtime unload failed artifactId=%s", self._active_artifact_id
                )
        with self._lock:
            self._state = "unavailable"
            if self._active_artifact_id is None:
                self._code = "NO_APPROVED_MODEL_ARTIFACT"
                self._message = "No release-approved VoltForge neural artifact is active."
            else:
                self._code = "MODEL_RUNTIME_STOPPED"
                self._message = "The local model runtime is stopped."

    def health(self) -> dict[str, Any]:
        with self._lock:
            runtime = self._runtime
            state = self._state
            code = self._code
            message = self._message
            registry_revision = self._registry_revision
            registry_sha256 = self._registry_sha256
            active_artifact_id = self._active_artifact_id
        if runtime is not None:
            health = runtime.health()
            health.update(
                {
                    "registryPath": str(self.registry_path),
                    "registryRevision": registry_revision,
                    "registrySha256": registry_sha256,
                    "activeArtifactId": active_artifact_id,
                    "runtimeState": health["state"],
                }
            )
            return health

        health = get_artifact_health(self.registry_path)
        if state in {"loading", "failed"} or code == "MODEL_RUNTIME_STOPPED":
            health.update(
                {
                    "ready": False,
                    "state": state,
                    "code": code,
                    "message": message,
                }
            )
        health.update(
            {
                "runtimeState": state,
                "runtimeOperational": False,
                "activeArtifactId": active_artifact_id,
                "requestedDevice": self.requested_device,
                "generationNetworkAccess": False,
                "trainingAvailableAtRuntime": False,
                "downloadAvailableAtRuntime": False,
            }
        )
        if registry_revision is not None:
            health["registryRevision"] = registry_revision
        if registry_sha256 is not None:
            health["registrySha256"] = registry_sha256
        return health

    def runtime_for_generation(self):
        with self._lock:
            if self._runtime is None or self._state != "ready":
                raise RuntimeError(f"{self._code}: {self._message}")
            return self._runtime

    def _set_failure(self, code: str, message: str) -> None:
        with self._lock:
            self._runtime = None
            self._state = "failed"
            self._code = code
            self._message = message


_PROCESS_RUNTIME = ModelRuntimeService(
    requested_device=os.environ.get("VOLTFORGE_AI_MODEL_DEVICE", "auto")
)


def get_model_runtime_service() -> ModelRuntimeService:
    return _PROCESS_RUNTIME
