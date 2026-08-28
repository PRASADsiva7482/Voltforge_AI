import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import config
from api.routes import router

app = FastAPI(
    title="Voltforge AI Microservice",
    description="Domain-specialized AI Engine for Electronics EDA, Circuit Safety, and MCU Firmware Synthesis.",
    version="2.0.0",
)


@app.middleware("http")
async def request_size_guard(request: Request, call_next):
    """Reject oversized declared and chunked request bodies before validation."""
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            declared_length = int(content_length)
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header."})
        if declared_length < 0:
            return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header."})
        if declared_length > config.MAX_REQUEST_BYTES:
            return JSONResponse(status_code=413, content={"detail": "AI request body is too large."})

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
            return JSONResponse(status_code=413, content={"detail": str(error)})
        raise

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-Voltforge-AI-Token", "X-Requested-With"],
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "service": "Voltforge AI Microservice",
        "status": "ONLINE",
        "docsUrl": "/docs",
        "healthUrl": "/voltForge-ai/api/v1/model/health"
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=False)
