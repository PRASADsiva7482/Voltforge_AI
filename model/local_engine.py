"""Unavailable compatibility facade for the retired unverified model adapter.

The owned model is supervised by ModelRuntimeService. Private transport and
actual neural decoding are later build tasks; this facade never invents output.
"""
from __future__ import annotations

import asyncio
import ipaddress
import os
from typing import Any, AsyncIterator, Optional
import urllib.parse

from model.identity import empty_identity_health

PRIVATE_NETWORKS = tuple(ipaddress.ip_network(value) for value in (
    "127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "::1/128", "fc00::/7", "fe80::/10",
))

class SecurityViolationError(RuntimeError):
    """Raised when an external third-party or public cloud endpoint is targeted."""


def validate_private_server_url(url: str) -> str:
    """Validate that the AI server URL is exclusively private, local, or LAN-hosted."""
    if not url or not url.strip():
        return "http://127.0.0.1:8000"

    parsed = urllib.parse.urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        raise SecurityViolationError(f"Invalid protocol scheme '{parsed.scheme}'; expected http or https.")

    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise SecurityViolationError("Missing hostname in AI model server URL.")

    # 1. Allow local aliases and private network suffixes
    if hostname in ("localhost", "127.0.0.1", "::1") or hostname.endswith((".local", ".lan", ".internal")):
        return url.strip()

    # 2. Check for private IP address (RFC 1918 / loopback)
    try:
        ip = ipaddress.ip_address(hostname)
        if any(ip in net for net in PRIVATE_NETWORKS) or ip.is_private or ip.is_loopback:
            return url.strip()
        raise SecurityViolationError(
            f"Public IP address '{hostname}' violates VoltForge's dedicated private LAN AI server policy."
        )
    except ValueError:
        pass

    # External domain names without .local/.lan/.internal are strictly rejected
    raise SecurityViolationError(
        f"Public domain '{hostname}' violates VoltForge's zero third-party and private LAN policy."
    )


class DedicatedServerConfig:
    """Configuration for dedicated private AI inference server."""

    def __init__(
        self,
        server_url: Optional[str] = None,
        timeout_seconds: float = 30.0,
        connect_timeout_seconds: float = 1.5,
        max_retries: int = 2,
        model_name: str = "vfdlm-domain-server",
    ):
        raw_url = server_url or os.environ.get("VOLTFORGE_AI_MODEL_SERVER_URL", "http://127.0.0.1:8000")
        self.server_url = validate_private_server_url(raw_url)
        self.timeout_seconds = timeout_seconds
        self.connect_timeout_seconds = connect_timeout_seconds
        self.max_retries = max_retries
        self.model_name = os.environ.get("VOLTFORGE_AI_MODEL_NAME", model_name)



class LocalEngine:
    """Deprecated import compatibility; never an active model or rule generator."""

    def __init__(self, config: DedicatedServerConfig | None = None, artifacts_dir: str | None = None):
        self.config = config or DedicatedServerConfig()

    def health(self) -> dict[str, Any]:
        return {**empty_identity_health(), "ready": False, "runtimeOperational": False,
                "state": "unavailable", "code": "MODEL_RUNTIME_NOT_IMPLEMENTED",
                "message": "This unverified adapter cannot serve a neural model.",
                "parameterCount": None, "contextLength": None, "device": None,
                "generationSource": "unavailable", "generationNetworkAccess": False,
                "trainingAvailableAtRuntime": False, "downloadAvailableAtRuntime": False}

    @staticmethod
    def _unavailable(cancellation_token: Any = None) -> None:
        if cancellation_token is not None:
            check = getattr(cancellation_token, "check", None)
            if callable(check):
                check()
            cancelled = getattr(cancellation_token, "is_cancelled", False)
            if (cancelled() if callable(cancelled) else cancelled) or getattr(cancellation_token, "cancelled", False):
                raise asyncio.CancelledError("Inference cancelled.")
        raise RuntimeError("MODEL_RUNTIME_NOT_IMPLEMENTED: use the verified model runtime; no rule substitution is permitted.")

    def generate_task_record(self, task_record: dict, *, cancellation_token: Any = None, **kwargs) -> str:
        self._unavailable(cancellation_token)

    async def generate_text(self, prompt: str, *, cancellation_token: Any = None, **kwargs) -> str:
        self._unavailable(cancellation_token)

    async def generate_stream(self, prompt: str, *, cancellation_token: Any = None, **kwargs) -> AsyncIterator[str]:
        self._unavailable(cancellation_token)
        # Retain the asynchronous iterator interface while producing no output.
        for chunk in ():
            yield chunk

    def unload(self) -> None:
        pass


LocalInferenceEngine = LocalEngine
DedicatedAIClient = LocalEngine
