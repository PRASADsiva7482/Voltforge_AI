import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import config
from api.routes import router

app = FastAPI(
    title="Voltforge AI Microservice",
    description="Domain-specialized AI Engine for Electronics EDA, Circuit Safety, and MCU Firmware Synthesis.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
