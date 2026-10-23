"""Validated local runtime configuration; malformed settings fail explicitly."""
import json
import os
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

ROOT = Path(__file__).resolve().parent
CONFIG_FILE = ROOT / 'config.json'


class Settings(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)
    environment: str = 'development'
    host: str = '127.0.0.1'
    port: int = Field(default=2002, ge=1, le=65535)
    allowed_origins: list[str] = Field(default_factory=lambda: ['http://localhost:5173', 'http://localhost:3000'])
    api_token: str = Field(default='', repr=False)
    max_request_bytes: int = Field(default=2_000_000, ge=1024, le=10_000_000)
    model_path: str = 'model/artifacts/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf'
    model_name: str = 'Qwen2.5-Coder-1.5B-Instruct'
    model_sha256: str = ''
    model_provider: Literal['local_gguf', 'local'] = 'local_gguf'
    auto_download_model: bool = True
    n_ctx: int = Field(default=4096, ge=2048, le=32768)
    n_threads: int = Field(default=4, ge=1, le=64)
    n_gpu_layers: int = Field(default=0, ge=-1)
    max_tokens: int = Field(default=768, ge=1, le=8192)
    temperature: float = Field(default=0.3, ge=0, le=2)
    generation_timeout_seconds: float = Field(default=120, ge=1, le=600)
    research_source_tokens: int = Field(default=384, ge=128, le=1200)
    research_source_count: int = Field(default=3, ge=1, le=5)
    embeddings_enabled: bool = True
    queue_timeout_seconds: float = Field(default=15, ge=0.1, le=60)
    max_pending_requests: int = Field(default=2, ge=0, le=16)
    web_search_enabled: bool = False
    brave_api_key: SecretStr = Field(default=SecretStr(''), repr=False)
    web_timeout_seconds: float = Field(default=20, ge=1, le=60)
    web_page_timeout_seconds: float = Field(default=5, ge=0.5, le=15)
    web_max_results: int = Field(default=5, ge=1, le=10)
    web_max_pages: int = Field(default=3, ge=0, le=5)
    web_max_response_bytes: int = Field(default=524288, ge=1024, le=2000000)
    web_max_concurrent: int = Field(default=2, ge=1, le=8)
    web_min_search_interval_seconds: float = Field(default=1.1, ge=1, le=60)

    @model_validator(mode='after')
    def validate_budget(self):
        if self.max_tokens > self.n_ctx // 2:
            raise ValueError('max_tokens must leave at least half of n_ctx for the prompt')
        return self

    def resolve_model_path(self):
        path = Path(self.model_path)
        return path if path.is_absolute() else ROOT / path


def load_settings():
    data = {}
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding='utf-8'))
        except (OSError, ValueError) as error:
            raise RuntimeError('Invalid config.json; correct it before starting the service') from error
    token_file = ROOT / '.service-token'
    if token_file.exists():
        data['api_token'] = token_file.read_text(encoding='utf-8').strip()
    aliases = {'environment': ['VOLTFORGE_AI_ENV', 'ENVIRONMENT'], 'host': ['VOLTFORGE_AI_HOST', 'HOST'],
               'port': ['VOLTFORGE_AI_PORT', 'PORT'], 'api_token': ['VOLTFORGE_AI_API_TOKEN', 'API_TOKEN'],
               'model_provider': ['VOLTFORGE_AI_MODEL_PROVIDER', 'MODEL_PROVIDER'],
               'brave_api_key': ['VOLTFORGE_AI_BRAVE_API_KEY', 'BRAVE_API_KEY']}
    for field in Settings.model_fields:
        for name in aliases.get(field, ['VOLTFORGE_AI_' + field.upper()]):
            if name in os.environ:
                data[field] = os.environ[name]; break
    if isinstance(data.get('allowed_origins'), str):
        data['allowed_origins'] = [item.strip() for item in data['allowed_origins'].split(',') if item.strip()]
    if 'ALLOWED_ORIGINS' in os.environ:
        data['allowed_origins'] = [item.strip() for item in os.environ['ALLOWED_ORIGINS'].split(',') if item.strip()]
    return Settings(**data)


settings = load_settings()


def get_settings():
    return settings
