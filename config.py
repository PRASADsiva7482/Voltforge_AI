"""Runtime configuration for the VoltForge AI service.

Secrets are read only from the process environment. Defaults are deliberately
local and non-secret so importing this module cannot leak a production
credential or connect to a remote database unexpectedly.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
import sys
from typing import Any


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

CONFIG_FILE = os.path.join(BASE_DIR, "config.json")


def _load_config_file() -> dict[str, Any]:
    if os.path.isfile(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

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


def _choice(name: str, default: str, allowed: set[str]) -> str:
    value = os.environ.get(name, default).strip().lower()
    if value not in allowed:
        raise ValueError(f"{name} must be one of: {', '.join(sorted(allowed))}.")
    return value


@dataclass(frozen=True)
class AiSettings:
    environment: str
    host: str
    port: int
    allowed_origins: tuple[str, ...]
    max_request_bytes: int
    max_context_characters: int
    chat_request_timeout_seconds: int
    internet_retrieval_enabled: bool
    internet_retrieval_timeout_seconds: int
    internet_retrieval_provider: str
    internet_retrieval_max_response_bytes: int
    internet_retrieval_cache_ttl_seconds: int
    store_conversations: bool
    bounded_memory_enabled: bool
    memory_database_path: str
    api_token: str
    local_model_device: str
    runtime_directory: str

    @property
    def service_authentication_required(self) -> bool:
        return self.environment == "production"

    @classmethod
    def from_environment(cls) -> "AiSettings":
        cfg = _load_config_file()
        origins = os.environ.get("VOLTFORGE_AI_ALLOWED_ORIGINS")
        if origins is None:
            cfg_origins = cfg.get("allowed_origins")
            if isinstance(cfg_origins, (list, tuple)):
                origins = ",".join(str(o) for o in cfg_origins)
            else:
                origins = str(cfg_origins or "http://localhost:3000")

        environment = _choice(
            "VOLTFORGE_AI_ENVIRONMENT",
            cfg.get("environment", "development"),
            {"development", "test", "production"},
        )
        runtime_value = os.path.expanduser(
            os.environ.get(
                "VOLTFORGE_AI_RUNTIME_DIRECTORY",
                cfg.get("runtime_directory", os.path.join(BASE_DIR, "runtime")),
            ).strip()
        )
        runtime_directory = os.path.abspath(
            runtime_value
            if os.path.isabs(runtime_value)
            else os.path.join(BASE_DIR, runtime_value)
        )
        api_token = os.environ.get("VOLTFORGE_AI_API_TOKEN", cfg.get("api_token", "")).strip()
        if environment == "production" and len(api_token) < 32:
            raise ValueError(
                "VOLTFORGE_AI_API_TOKEN must contain at least 32 characters in production."
            )
        allowed_origins = tuple(origin.strip() for origin in origins.split(",") if origin.strip())
        if environment == "production" and (
            not allowed_origins or "*" in allowed_origins
        ):
            raise ValueError(
                "VOLTFORGE_AI_ALLOWED_ORIGINS must contain explicit origins in production."
            )
        memory_value = os.path.expanduser(
            os.environ.get(
                "VOLTFORGE_AI_MEMORY_DATABASE_PATH",
                cfg.get("memory_database_path", os.path.join(runtime_directory, "bounded-memory-v1.sqlite3")),
            ).strip()
        )
        memory_database_path = os.path.abspath(
            memory_value
            if os.path.isabs(memory_value)
            else os.path.join(runtime_directory, memory_value)
        )
        if environment == "production":
            try:
                common_path = os.path.commonpath([runtime_directory, memory_database_path])
            except ValueError as error:
                raise ValueError(
                    "VOLTFORGE_AI_MEMORY_DATABASE_PATH must be inside VOLTFORGE_AI_RUNTIME_DIRECTORY in production."
                ) from error
            if common_path != runtime_directory:
                raise ValueError(
                    "VOLTFORGE_AI_MEMORY_DATABASE_PATH must be inside VOLTFORGE_AI_RUNTIME_DIRECTORY in production."
                )
        return cls(
            environment=environment,
            host=os.environ.get("VOLTFORGE_AI_HOST", cfg.get("host", "127.0.0.1")).strip() or "127.0.0.1",
            port=_bounded_int("VOLTFORGE_AI_PORT", cfg.get("port", 2002), 1, 65_535),
            allowed_origins=allowed_origins,
            max_request_bytes=_bounded_int(
                "VOLTFORGE_AI_MAX_REQUEST_BYTES", cfg.get("max_request_bytes", 2_000_000), 1_024, 2_000_000
            ),
            max_context_characters=_bounded_int(
                "VOLTFORGE_AI_MAX_CONTEXT_CHARACTERS", cfg.get("max_context_characters", 48_000), 4_000, 200_000
            ),
            chat_request_timeout_seconds=_bounded_int(
                "VOLTFORGE_AI_CHAT_REQUEST_TIMEOUT_SECONDS", cfg.get("chat_request_timeout_seconds", 15), 1, 120
            ),
            internet_retrieval_enabled=_enabled(
                os.environ.get("VOLTFORGE_AI_INTERNET_RETRIEVAL_ENABLED"), default=cfg.get("internet_retrieval_enabled", False)
            ),
            internet_retrieval_timeout_seconds=_bounded_int(
                "VOLTFORGE_AI_INTERNET_RETRIEVAL_TIMEOUT_SECONDS", cfg.get("internet_retrieval_timeout_seconds", 2), 1, 10
            ),
            internet_retrieval_provider=_choice(
                "VOLTFORGE_AI_INTERNET_RETRIEVAL_PROVIDER",
                cfg.get("internet_retrieval_provider", "duckduckgo-instant-answer-v1"),
                {"duckduckgo-instant-answer-v1"},
            ),
            internet_retrieval_max_response_bytes=_bounded_int(
                "VOLTFORGE_AI_INTERNET_RETRIEVAL_MAX_RESPONSE_BYTES",
                cfg.get("internet_retrieval_max_response_bytes", 262_144),
                4_096,
                262_144,
            ),
            internet_retrieval_cache_ttl_seconds=_bounded_int(
                "VOLTFORGE_AI_INTERNET_RETRIEVAL_CACHE_TTL_SECONDS",
                cfg.get("internet_retrieval_cache_ttl_seconds", 900),
                60,
                86_400,
            ),
            # Legacy raw chat persistence is permanently disabled. VFAI-025
            # stores only governed, redacted, user-controlled memory records.
            store_conversations=False,
            bounded_memory_enabled=_enabled(
                os.environ.get("VOLTFORGE_AI_BOUNDED_MEMORY_ENABLED"), default=cfg.get("bounded_memory_enabled", True)
            ),
            memory_database_path=memory_database_path,
            api_token=api_token,
            local_model_device=_choice(
                "VOLTFORGE_AI_MODEL_DEVICE",
                cfg.get("local_model_device", "auto"),
                {"auto", "cpu", "cuda", "mps"},
            ),
            runtime_directory=runtime_directory,
        )


def get_settings() -> AiSettings:
    """Build settings on demand so environment-based tests remain isolated."""
    return AiSettings.from_environment()


_settings = get_settings()

HOST = _settings.host
PORT = _settings.port
CORS_ALLOWED_ORIGINS = list(_settings.allowed_origins)
MAX_REQUEST_BYTES = _settings.max_request_bytes

# Database configuration loaded from property file
_cfg = _load_config_file()
_db_cfg = _cfg.get("database", {})
DB_HOST = os.environ.get("VOLTFORGE_AI_DB_HOST", _db_cfg.get("host", "127.0.0.1"))
DB_PORT = int(os.environ.get("VOLTFORGE_AI_DB_PORT", str(_db_cfg.get("port", 3306))))
DB_USER = os.environ.get("VOLTFORGE_AI_DB_USERNAME", _db_cfg.get("username", "root"))
DB_PASSWORD = os.environ.get("VOLTFORGE_AI_DB_PASSWORD", _db_cfg.get("password", ""))
DB_NAME = os.environ.get("VOLTFORGE_AI_DB_NAME", _db_cfg.get("name", "voltforge_ai"))
DB_ENABLED = _enabled(os.environ.get("VOLTFORGE_AI_DB_ENABLED"), default=_db_cfg.get("enabled", False))

_log_cfg = _cfg.get("logging", {})
LOG_LEVEL = os.environ.get("LOG_LEVEL", _log_cfg.get("level", "INFO"))
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("voltforge-ai")
logger.info(
    "VoltForge AI configuration loaded (host=%s, port=%s, databaseEnabled=%s, localGeneration=true, modelDevice=%s, internetRetrievalEnabled=%s)",
    HOST,
    PORT,
    DB_ENABLED,
    _settings.local_model_device,
    _settings.internet_retrieval_enabled,
)
