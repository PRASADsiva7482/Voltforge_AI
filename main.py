"""
VoltForge AI Microservice
Domain-specialized AI Engine for Electronics EDA, Circuit Safety, and MCU Firmware Synthesis.
Runs on 127.0.0.1:2002
"""

import sys
import subprocess
from pathlib import Path

# Auto-delegate to local virtual environment if llama_cpp is missing in current Python
try:
    import llama_cpp
except ImportError:
    venv_python = Path(__file__).resolve().parent / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists() and sys.executable.lower() != str(venv_python).lower():
        print(f"[VoltForge AI] Redirecting execution to project virtual environment (.venv)...")
        sys.exit(subprocess.call([str(venv_python)] + sys.argv))

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import config
from api.routes import router
from engine.llm import get_llm_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("voltforge-ai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = config.get_settings()
    logger.info("Starting VoltForge AI Microservice on %s:%d...", settings.host, settings.port)
    llm = get_llm_engine()
    llm.initialize()
    logger.info("VoltForge AI ready to serve requests.")
    yield
    logger.info("VoltForge AI shutting down.")


settings = config.get_settings()
is_dev = settings.environment.lower() == "development"

app = FastAPI(
    title="VoltForge AI Microservice",
    description="Domain-specialized AI Engine for Electronics EDA, Circuit Safety, and MCU Firmware Synthesis.",
    version="2.1.0",
    lifespan=lifespan,
    docs_url="/docs" if is_dev else None,
    redoc_url="/redoc" if is_dev else None,
    openapi_url="/openapi.json" if is_dev else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    info = {
        "service": "VoltForge AI Microservice",
        "status": "ONLINE",
        "version": "2.1.0",
        "healthUrl": "/voltForge-ai/api/v1/model/health"
    }
    if is_dev:
        info["docsUrl"] = "/docs"
    return info


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=False
    )
