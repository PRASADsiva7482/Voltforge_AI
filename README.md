# VoltForge AI Microservice (v2.1.0)

Domain-specialized AI Engine for Electronics EDA, Circuit Safety, and MCU Firmware Synthesis.

## Architecture
- **FastAPI** web service running on `http://127.0.0.1:2002`
- **Target Microcontroller & Electronics Support**: Arduino (Uno, Nano, Mega), ESP32, ESP8266, Raspberry Pi Pico, STM32.
- **Model Engine**: Designed for Qwen 2.5 Coder 1.5B (GGUF / Transformers / Local) with real-time SSE streaming.
- **Circuit Verification**: Real-time electrical rule checks (short circuit detection, LED current limiting, floating ground checks).
- **Firmware Generator**: Generates compilable C++ firmware matching placed circuit canvas components.

## Running the Service
```powershell
# Using the local Python 3.12 virtual environment:
.\.venv\Scripts\python.exe main.py
```

The service will be live on `http://127.0.0.1:2002` and ready to serve `Voltforge_BL` (Spring Boot) and `Voltforge_UI` (React/Vite).

## API Endpoints
- `GET /` — Microservice status
- `GET /voltForge-ai/api/v1/model/health` — Health and model state
- `GET /voltForge-ai/api/v1/model/hardware-coverage` — 30 supported microcontroller platforms
- `GET /voltForge-ai/api/v1/model/component-coverage` — Supported sensors & electronics
- `POST /voltForge-ai/api/v1/model/chat` — Full chat response
- `POST /voltForge-ai/api/v1/model/chat/stream` — Real-time Server-Sent Events (SSE) token stream
- `POST /voltForge-ai/api/v1/model/validate-circuit` — Circuit safety and DRC inspection
- `POST /voltForge-ai/api/v1/model/suggest-wiring` — Pin routing recommendations
- `POST /voltForge-ai/api/v1/model/generate-code` — Microcontroller C++ code synthesis
