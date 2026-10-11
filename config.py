import os
import json
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field

CONFIG_FILE = Path(__file__).resolve().parent / "config.json"

class Settings(BaseModel):
    environment: str = "development"
    host: str = "127.0.0.1"
    port: int = 2002
    allowed_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173"
    ]
    api_token: str = "vfa_sec_9f7a8b3c1d2e4f5a6b7c8d9e0f1a2b3c"
    max_request_bytes: int = 2_000_000
    
    # Model configuration
    model_path: str = "model/artifacts/Qwen2.5-Coder-1.5B-Instruct.Q4_K_M.gguf"
    model_name: str = "Qwen2.5-Coder-1.5B-Instruct"
    model_provider: str = "local_gguf" # "local_gguf", "local", or "openai_compatible"
    n_ctx: int = 4096
    n_threads: int = 4
    n_gpu_layers: int = 0
    openai_compatible_base_url: str = "" # e.g. "http://localhost:11434/v1" if user runs an external server
    openai_compatible_api_key: str = "EMPTY"
    device: str = "cpu"
    max_tokens: int = 2048
    temperature: float = 0.7

def load_settings() -> Settings:
    defaults = Settings()
    data = defaults.model_dump()
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                file_data = json.load(f)
                data.update({k: v for k, v in file_data.items() if v is not None})
        except Exception as e:
            print(f"Warning: Failed to parse config.json ({e}), using default settings.")

    # Environment variables take precedence over config.json
    env_overrides = {}
    if host := os.environ.get("VOLTFORGE_AI_HOST") or os.environ.get("HOST"):
        env_overrides["host"] = host
    if port := os.environ.get("VOLTFORGE_AI_PORT") or os.environ.get("PORT"):
        try:
            env_overrides["port"] = int(port)
        except ValueError:
            pass
    if token := os.environ.get("VOLTFORGE_AI_API_TOKEN") or os.environ.get("API_TOKEN"):
        env_overrides["api_token"] = token
    if env := os.environ.get("VOLTFORGE_AI_ENV") or os.environ.get("ENVIRONMENT"):
        env_overrides["environment"] = env
    if origins := os.environ.get("ALLOWED_ORIGINS"):
        env_overrides["allowed_origins"] = [o.strip() for o in origins.split(",") if o.strip()]
    if model_provider := os.environ.get("MODEL_PROVIDER"):
        env_overrides["model_provider"] = model_provider
    if base_url := os.environ.get("OPENAI_COMPATIBLE_BASE_URL"):
        env_overrides["openai_compatible_base_url"] = base_url
    if api_key := os.environ.get("OPENAI_COMPATIBLE_API_KEY"):
        env_overrides["openai_compatible_api_key"] = api_key

    data.update(env_overrides)
    return Settings(**data)

settings = load_settings()

def get_settings() -> Settings:
    return settings
