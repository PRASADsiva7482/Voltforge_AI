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
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return Settings(**{**defaults.model_dump(), **data})
        except Exception as e:
            print(f"Warning: Failed to parse config.json ({e}), using default settings.")
    return defaults

settings = load_settings()

def get_settings() -> Settings:
    return settings
