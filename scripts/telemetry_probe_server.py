"""Opt-in loopback integration-test host; never imported by the production app.

Exposes an authenticated aggregate snapshot only in this test process. The model,
routes, request authentication, admission and streaming are production code.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi import Depends
from api.security import require_service_token
from engine.llm import get_llm_engine
from main import app
import uvicorn


@app.get('/__test/telemetry', dependencies=[Depends(require_service_token)])
def snapshot():
    return get_llm_engine().runtime.telemetry.snapshot()


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=int(sys.argv[1]), access_log=False)
