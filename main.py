"""
VoltForge AI Microservice
Domain-specialized AI Engine for Electronics EDA, Circuit Safety, and MCU Firmware Synthesis.
Runs on 127.0.0.1:2002
"""

# IDEs commonly launch this file with the global Python. Dispatch before importing
# native/runtime dependencies; importing main:app must remain side-effect free.
if __name__ == '__main__':
    from bootstrap import ensure_launch_runtime
    ensure_launch_runtime()

import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import config
from api.routes import router
from engine.llm import get_llm_engine
from engine.runtime import RuntimeBusy, GenerationTimedOut
from api.limits import RequestSizeLimit
from api.embeddings import router as embedding_router
from fastapi.responses import JSONResponse

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
    if not settings.api_token:
        raise RuntimeError("Service authentication is not configured. Run bootstrap.py and configure the backend with the matching service token.")
    await asyncio.to_thread(llm.initialize)
    if llm.llm is None:
        await asyncio.to_thread(llm.close)
        raise RuntimeError('The AI model failed to load. Run python bootstrap.py --verify; the service will not accept chat requests without a working model.')
    logger.info("AI service started; model ready=%s", llm.llm is not None)
    from engine.embeddings import embedding_engine
    if settings.embeddings_enabled:
        try:
            await asyncio.to_thread(embedding_engine.initialize)
        except Exception as error:
            logger.error('Reference embeddings unavailable (%s); ordinary chat remains available', type(error).__name__)
    try:
        yield
    finally:
        await asyncio.to_thread(llm.close)
        if embedding_engine.session is not None:
            await asyncio.to_thread(embedding_engine.session.close)
            embedding_engine.session = None


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
app.include_router(embedding_router)
app.add_middleware(RequestSizeLimit, max_bytes=settings.max_request_bytes)


@app.exception_handler(RuntimeBusy)
async def busy_response(request, error):
    return JSONResponse(status_code=429, content={"code": "AI_BUSY", "message": "AI is busy. Retry shortly."}, headers={"Retry-After": "5"})


@app.exception_handler(GenerationTimedOut)
async def timeout_response(request, error):
    return JSONResponse(status_code=504, content={"code": "AI_GENERATION_TIMEOUT", "message": "Generation timed out. Try a shorter question."})


@app.exception_handler(ValueError)
async def context_response(request, error):
    return JSONResponse(status_code=422, content={"code": "AI_CONTEXT_LIMIT", "message": "Input exceeds the local model context. Send a shorter question or less project context."})


@app.exception_handler(RuntimeError)
async def inference_error(request, error):
    logger.error("Inference failed (%s)", type(error).__name__)
    return JSONResponse(status_code=503, content={"code": "AI_GENERATION_FAILED", "message": "Local AI generation failed. Your saved conversation is unchanged."})


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
