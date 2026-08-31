import uvicorn
import logging
import asyncio
import uuid
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import config
from api.routes import router
from api_contract import error_response
from api.security import require_service_token
from model.runtime_service import get_model_runtime_service
from observability import observability, safe_request_id


logger = logging.getLogger("voltforge-ai.startup")
model_runtime_service = get_model_runtime_service()


def log_model_registry_identity() -> None:
    """Log immutable model identity only; request and prompt content is excluded."""
    health = model_runtime_service.health()
    catalog_ids = ",".join(
        str(item["artifactId"]) for item in health.get("catalogArtifacts", [])
    ) or "none"
    logger.info(
        "Model registry state=%s activeArtifactId=%s catalogArtifactIds=%s "
        "registryRevision=%s registrySha256=%s",
        health["state"],
        health["artifactId"],
        catalog_ids,
        health.get("registryRevision"),
        health.get("registrySha256"),
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await asyncio.to_thread(model_runtime_service.start)
    log_model_registry_identity()
    try:
        yield
    finally:
        await asyncio.to_thread(model_runtime_service.stop)

app = FastAPI(
    title="Voltforge AI Microservice",
    description="Domain-specialized AI Engine for Electronics EDA, Circuit Safety, and MCU Firmware Synthesis.",
    version="2.0.0",
    docs_url="/docs" if config._settings.environment != "production" else None,
    redoc_url="/redoc" if config._settings.environment != "production" else None,
    openapi_url="/openapi.json" if config._settings.environment != "production" else None,
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def contract_validation_error(
    _request: Request, error: RequestValidationError
) -> JSONResponse:
    fields = [
        ".".join(str(part) for part in item.get("loc", ()) if part != "body")[:200]
        for item in error.errors()[:32]
    ]
    return JSONResponse(
        status_code=422,
        content=error_response(
            "API_SCHEMA_VALIDATION_FAILED",
            "The request is incompatible with the VoltForge AI v1 contract.",
            retryable=False,
            field_errors=fields,
        ),
    )


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    """Record bounded request facts and propagate one safe request ID."""
    supplied_request_id = request.headers.get("x-voltforge-request-id")
    request_id = safe_request_id(supplied_request_id)
    if request_id == "request-unknown":
        request_id = f"request-{uuid.uuid4()}"
    request.state.voltforge_request_id = request_id
    observation = observability.begin_request(request_id, request.url.path)
    response = None
    status_code = None
    outcome = None
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except asyncio.CancelledError:
        outcome = "cancelled"
        raise
    except Exception:
        outcome = "failure"
        raise
    finally:
        route = getattr(request.scope.get("route"), "path", None) or request.url.path
        observability.finish_request(
            observation,
            status_code=status_code,
            outcome=outcome,
            route=route,
        )
        if response is not None:
            response.headers["X-Voltforge-Request-Id"] = request_id


@app.middleware("http")
async def request_size_guard(request: Request, call_next):
    """Reject oversized declared and chunked request bodies before validation."""
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            declared_length = int(content_length)
        except ValueError:
            return JSONResponse(
                status_code=400,
                content=error_response(
                    "API_CONTENT_LENGTH_INVALID",
                    "The Content-Length header is invalid.",
                    retryable=False,
                ),
            )
        if declared_length < 0:
            return JSONResponse(
                status_code=400,
                content=error_response(
                    "API_CONTENT_LENGTH_INVALID",
                    "The Content-Length header is invalid.",
                    retryable=False,
                ),
            )
        if declared_length > config.MAX_REQUEST_BYTES:
            return JSONResponse(
                status_code=413,
                content=error_response(
                    "API_REQUEST_TOO_LARGE",
                    "The AI request body is too large.",
                    retryable=False,
                ),
            )

    received = 0
    original_receive = request._receive

    async def limited_receive():
        nonlocal received
        message = await original_receive()
        if message.get("type") == "http.request":
            received += len(message.get("body", b""))
            if received > config.MAX_REQUEST_BYTES:
                raise ValueError("AI request body is too large.")
        return message

    request._receive = limited_receive
    try:
        return await call_next(request)
    except ValueError as error:
        if str(error) == "AI request body is too large.":
            return JSONResponse(
                status_code=413,
                content=error_response(
                    "API_REQUEST_TOO_LARGE",
                    "The AI request body is too large.",
                    retryable=False,
                ),
            )
        raise

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "X-Voltforge-AI-Token",
        "X-Voltforge-Request-Id",
        "X-Voltforge-User-Id",
        "X-Voltforge-Project-Id",
        "X-Voltforge-Session-Id",
        "X-Requested-With",
    ],
)

app.include_router(router)


@app.get("/", dependencies=[Depends(require_service_token)])
def root():
    return {
        "service": "Voltforge AI Microservice",
        "status": "ONLINE",
        "docsUrl": "/docs",
        "healthUrl": "/voltForge-ai/api/v1/model/health"
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=False)
