"""Runtime configuration for the VoltForge AI service.

Secrets are read only from the process environment. Defaults are deliberately
local and non-secret so importing this module cannot leak a production
credential or connect to a remote database unexpectedly.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def _enabled(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer.") from error
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return value


@dataclass(frozen=True)
class AiSettings:
    host: str
    port: int
    allowed_origins: tuple[str, ...]
    max_request_bytes: int
    max_context_characters: int
    internet_retrieval_enabled: bool
    internet_retrieval_timeout_seconds: int
    store_conversations: bool
    api_token: str

    @classmethod
    def from_environment(cls) -> "AiSettings":
        origins = os.environ.get(
            "VOLTFORGE_AI_ALLOWED_ORIGINS",
            os.environ.get("CORS_ALLOWED_ORIGINS", "http://localhost:3000"),
        )
        return cls(
            host=os.environ.get("VOLTFORGE_AI_HOST", "127.0.0.1").strip() or "127.0.0.1",
            port=_bounded_int("VOLTFORGE_AI_PORT", 2002, 1, 65_535),
            allowed_origins=tuple(origin.strip() for origin in origins.split(",") if origin.strip()),
            max_request_bytes=_bounded_int(
                "VOLTFORGE_AI_MAX_REQUEST_BYTES", 2_000_000, 1_024, 10_000_000
            ),
            max_context_characters=_bounded_int(
                "VOLTFORGE_AI_MAX_CONTEXT_CHARACTERS", 48_000, 4_000, 200_000
            ),
            internet_retrieval_enabled=_enabled(
                os.environ.get("VOLTFORGE_AI_INTERNET_RETRIEVAL_ENABLED"), default=False
            ),
            internet_retrieval_timeout_seconds=_bounded_int(
                "VOLTFORGE_AI_INTERNET_RETRIEVAL_TIMEOUT_SECONDS", 2, 1, 10
            ),
            store_conversations=_enabled(os.environ.get("VOLTFORGE_AI_STORE_CONVERSATIONS")),
            api_token=os.environ.get("VOLTFORGE_AI_API_TOKEN", "").strip(),
        )


def get_settings() -> AiSettings:
    """Build settings on demand so environment-based tests remain isolated."""
    return AiSettings.from_environment()


_settings = get_settings()

HOST = _settings.host
PORT = _settings.port
CORS_ALLOWED_ORIGINS = list(_settings.allowed_origins)
MAX_REQUEST_BYTES = _settings.max_request_bytes

# Database configuration. There are intentionally no remote-host or password
# defaults here. Persistence remains opt-in for local development and tests.
DB_HOST = os.getenv("VOLTFORGE_AI_DB_HOST", os.getenv("VOLTFORGE_DB_HOST", "127.0.0.1"))
DB_PORT = int(os.getenv("VOLTFORGE_AI_DB_PORT", os.getenv("VOLTFORGE_DB_PORT", "3306")))
DB_USER = os.getenv(
    "VOLTFORGE_AI_DB_USERNAME",
    os.getenv("VOLTFORGE_DB_USERNAME", os.getenv("VOLTFORGE_DB_USER", "root")),
)
DB_PASSWORD = os.getenv("VOLTFORGE_AI_DB_PASSWORD", os.getenv("VOLTFORGE_DB_PASSWORD", ""))
DB_NAME = os.getenv("VOLTFORGE_AI_DB_NAME", "voltforge_ai")
DB_ENABLED = _enabled(os.getenv("VOLTFORGE_AI_DB_ENABLED"), default=False)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("voltforge-ai")
logger.info(
    "VoltForge AI configuration loaded (host=%s, port=%s, databaseEnabled=%s, localGeneration=true, internetRetrievalEnabled=%s)",
    HOST,
    PORT,
    DB_ENABLED,
    _settings.internet_retrieval_enabled,
)
