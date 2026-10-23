"""Opt-in real-model HTTP smoke test in an isolated temporary service process.

Uses synthetic prompts only. Does not alter configured credentials or the database.
"""
import asyncio
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import httpx

ROOT = Path(__file__).resolve().parents[1]


async def verify():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    token = secrets.token_urlsafe(32)
    environment = dict(os.environ, VOLTFORGE_AI_PORT=str(port), VOLTFORGE_AI_HOST='127.0.0.1',
                       VOLTFORGE_AI_API_TOKEN=token, VOLTFORGE_AI_MAX_TOKENS='48',
                       VOLTFORGE_AI_MAX_PENDING_REQUESTS='0')
    base = f'http://127.0.0.1:{port}/voltForge-ai/api/v1/model'
    with tempfile.TemporaryFile() as logs:
        process = subprocess.Popen([sys.executable, 'main.py'], cwd=ROOT, env=environment,
                                   stdout=logs, stderr=logs,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        try:
            async with httpx.AsyncClient(headers={'X-Voltforge-AI-Token':token}, timeout=90) as client:
                deadline = time.monotonic() + 90
                while True:
                    try:
                        health = await client.get(base + '/health')
                        if health.status_code == 200 and health.json()['ready']:
                            break
                    except httpx.ConnectError:
                        pass
                    if process.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError('Isolated service did not become ready; inspect local runtime setup')
                    await asyncio.sleep(.25)
                unauthorized = await client.get(base + '/health', headers={'X-Voltforge-AI-Token':'invalid'})
                assert unauthorized.status_code == 401
                oversized = await client.post(base + '/chat', content=b'x' * 2_000_001)
                assert oversized.status_code == 413
                async with client.stream('POST', base + '/chat/stream', json={'message':'Explain what a resistor does in one sentence.'}) as response:
                    assert response.status_code == 200
                    lines = response.aiter_lines()
                    assert await anext(lines) == 'event: start'
                    await asyncio.sleep(.5)
                    latencies = []
                    for _ in range(5):
                        began = time.monotonic()
                        check = await client.get(base + '/health')
                        assert check.json()['ready']
                        latencies.append(time.monotonic() - began)
                        await asyncio.sleep(.05)
                    busy = await client.post(base + '/chat/stream', json={'message':'A second request'})
                    assert 'AI_BUSY' in busy.text and 'event: complete' not in busy.text
                    async for line in lines:
                        if line == 'event: token':
                            break
                    else:
                        raise AssertionError('Real model did not emit a token')
                    # Closing this HTTP connection must stop native work and release its lock.
                await asyncio.sleep(1)
                began = time.monotonic()
                resumed = await client.post(base + '/chat/stream', json={'message':'In one short sentence, what is a resistor?'})
                assert 'event: token' in resumed.text and 'event: complete' in resumed.text
                assert 'event: error' not in resumed.text
                print(json.dumps({'realModel':True, 'authentication':True, 'bodyLimit':True,
                                  'busyGuard':True, 'disconnectThenGenerate':True,
                                  'maxHealthLatencySeconds':round(max(latencies),3),
                                  'completedResponseSeconds':round(time.monotonic()-began,2)}))
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=10)


if __name__ == '__main__':
    asyncio.run(verify())
