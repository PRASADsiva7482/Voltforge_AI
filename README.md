# VoltForge AI Microservice (v2.1.0)

Domain-specialized AI Engine for Electronics EDA, Circuit Safety, and MCU Firmware Synthesis powered by **Qwen 2.5 Coder 1.5B Instruct** (quantized GGUF Q4_K_M running via `llama-cpp-python`).

---

## Architecture Overview
- **Service Core**: FastAPI web service running on `http://127.0.0.1:2002`.
- **Backend Consumer**: `Voltforge_BL` (Spring Boot API Gateway on port 2001) forwards canvas requests with HMAC service token authentication (`X-Voltforge-AI-Token: vfa_sec_9f7a8b3c1d2e4f5a6b7c8d9e0f1a2b3c`).
- **Frontend Consumer**: `Voltforge_UI` (React/Vite) receives token streams and interactive workspace proposals (wiring suggestions, DRC corrections, and Monaco C++ firmware updates).
- **Inference Engine**:
  - **Neural (Primary)**: In-process GGUF inference via `llama-cpp-python` loading `model/artifacts/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` into RAM in < 0.5s.
  - **Deterministic (Fallback)**: Embedded electronics rule engine guaranteeing 100% uptime even if model weights are offline.
- **Hardware Coverage**: 30 verified microcontroller platforms (Arduino Uno/Nano/Mega/Giga, ESP32, ESP8266, Raspberry Pi Pico, STM32, Teensy, etc.).
- **Electrical DRC Engine**: Real-time rule checks (VCC-GND short circuits, LED current-limiting resistors, 5V-to-3.3V logic level incompatibilities, inductive kickback flyback diodes, floating inputs, and common ground).

---

## Model Weight Management & Git Policy
Model binary weights (`*.gguf`, ~1.04 GiB) are **intentionally excluded from Git** in `.gitignore`:
```gitignore
*.gguf
model/artifacts/*.gguf
```
Git repositories should never store multi-gigabyte binary model files because GitHub rejects pushes over 100MB and Git history would bloat permanently.

---

## Quick Start: Download Fresh Model & Run

### 1. Download Clean Base Model (Qwen 2.5 Coder 1.5B Instruct)
Run the automated downloader (downloads directly from official Hugging Face repository and validates integrity):
```powershell
# Option A: Double-click or run the batch script
.\download_model.bat

# Option B: Run via Python in .venv
.\.venv\Scripts\python.exe download_model.py

# To force a fresh re-download from scratch:
.\.venv\Scripts\python.exe download_model.py --force
```

### 2. Launch the AI Microservice
> [!IMPORTANT]
> Always run using the local Python 3.12 virtual environment (`.venv`) where `llama-cpp-python` is installed. Running with system Python 3.14 will cause `MODEL_NOT_LOADED` because `llama_cpp` is not installed globally.

```powershell
# Option A: Using the convenient launcher script
.\run.bat

# Option B: Directly with virtual environment python
.\.venv\Scripts\python.exe main.py
```

The service will start on `http://127.0.0.1:2002`.

### 3. Verify Health & Model Readiness
Check the health status endpoint with the service token:
```powershell
curl -H "X-Voltforge-AI-Token: vfa_sec_9f7a8b3c1d2e4f5a6b7c8d9e0f1a2b3c" http://127.0.0.1:2002/voltForge-ai/api/v1/model/health
```
Expected healthy output:
```json
{
  "status": "ONLINE",
  "state": "ready",
  "ready": true,
  "code": "MODEL_RUNTIME_READY",
  "serviceReady": true,
  "responseMode": "neural",
  "model": "Qwen2.5-Coder-1.5B-Instruct",
  "provider": "local_gguf",
  "engine": "llama-cpp-python (in-process)",
  "neuralLoaded": true,
  "version": "2.1.0"
}
```

---

## API Endpoints
All `/voltForge-ai/api/v1/model/*` endpoints require the header `X-Voltforge-AI-Token: vfa_sec_9f7a8b3c1d2e4f5a6b7c8d9e0f1a2b3c`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Microservice status (Public) |
| `GET` | `/docs` | Interactive Swagger UI (Development mode) |
| `GET` | `/voltForge-ai/api/v1/model/health` | Model loading state & engine readiness |
| `GET` | `/voltForge-ai/api/v1/model/hardware-coverage` | 30 supported microcontroller platforms & pinouts |
| `GET` | `/voltForge-ai/api/v1/model/component-coverage` | Supported electronics, sensors, displays, and ICs |
| `POST` | `/voltForge-ai/api/v1/model/chat` | Full synchronous copilot response with proposals |
| `POST` | `/voltForge-ai/api/v1/model/chat/stream` | Server-Sent Events (SSE) real-time token stream |
| `POST` | `/voltForge-ai/api/v1/model/validate-circuit` | Circuit safety DRC verification |
| `POST` | `/voltForge-ai/api/v1/model/suggest-wiring` | Board-specific pin routing suggestions |
| `POST` | `/voltForge-ai/api/v1/model/generate-code` | Microcontroller C++ code generation matching canvas |

---

## Training & Fine-Tuning Pipeline
If you wish to fine-tune a specialized model on the VoltForge Electronics Dataset:
- Dataset Generator: `training/prepare_training_data.py` (530+ verified electronics samples)
- Dataset File: `training/voltforge_electronics_dataset.jsonl`
- Colab Runner: `training/voltforge_colab_training.ipynb` or `training/colab_train_runner.py`
- Specification & Rules: `training/VOLTFORGE_AI_USECASES.md`

> [!TIP]
> When exporting from Google Colab, mount Google Drive or push directly to Hugging Face Hub to prevent browser download timeouts on 1GB GGUF files.
