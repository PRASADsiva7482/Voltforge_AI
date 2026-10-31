"""Opt-in real-model HTTP smoke test in an isolated temporary service process.

Uses synthetic prompts only. Does not alter configured credentials or the database.
"""
import asyncio
import argparse
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
import hashlib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
REQUIRED_CHECKS = ('nativeModelReady', 'authentication', 'bodyLimit', 'busyGuard',
                   'incrementalNativeToken', 'disconnectThenGenerate')
CAPACITY_CHECKS = ('configuredResponseBudget', 'sequentialWorkloads', 'boundedOverload',
                   'responsiveHealth', 'workingSetBudget')


def process_peak_working_set(pid):
    # The managed Windows venv executable can launch the actual interpreter as
    # a child. Measuring only Popen.pid would incorrectly report about 5 MiB.
    command = (
        '$all = @(Get-CimInstance Win32_Process); '
        '$ids = [System.Collections.Generic.List[int]]::new(); '
        f'$ids.Add({int(pid)}); '
        'for ($i = 0; $i -lt $ids.Count; $i++) { '
        'foreach ($p in $all) { if ($p.ProcessId -ne $PID -and $p.ParentProcessId -eq $ids[$i] -and -not $ids.Contains([int]$p.ProcessId)) '
        '{ $ids.Add([int]$p.ProcessId) } } }; '
        'ConvertTo-Json -Compress -InputObject @($ids.ToArray())'
    )
    result = subprocess.run(['powershell', '-NoProfile', '-Command', command],
                            capture_output=True, text=True, check=True, timeout=15,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    identifiers = json.loads(result.stdout)
    return round(sum(_single_process_peak(identifier) for identifier in identifiers), 2)


def _single_process_peak(pid):
    """Windows child-process high-water RSS; do not report harness memory as AI memory."""
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ('peak', 'working', 'peakPaged', 'paged',
                                               'peakNonpaged', 'nonpaged', 'pagefile', 'peakPagefile')]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    query = ctypes.WinDLL('psapi', use_last_error=True).GetProcessMemoryInfo
    query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    handle = kernel.OpenProcess(0x410, False, pid)
    if not handle:
        raise OSError('Cannot measure AI process memory')
    try:
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not query(handle, ctypes.byref(counters), counters.cb):
            raise OSError('Cannot read AI process memory')
        return round(counters.peak / 1024 ** 2, 2)
    finally:
        kernel.CloseHandle(handle)


async def capacity_workloads(client, base, process, report):
    samples = report['measurements'].setdefault('workloads', [])
    excerpt = 'Arduino Uno R3 supports millis(), LED_BUILTIN, pinMode and digitalWrite.'
    document = {'sourceId': 'capacity-document', 'kind': 'document_chunk',
                'documentId': 'capacity-document', 'versionId': 'v1', 'page': 1,
                'title': 'Synthetic capacity reference', 'excerpt': excerpt,
                'contentHash': hashlib.sha256(excerpt.encode()).hexdigest(),
                'retrievedAt': '2026-10-26T00:00:00Z'}
    payloads = [
        ('general', {'message': 'Explain what a resistor does in one sentence.'}),
        ('document_firmware', {'message': 'Write a minimal Arduino sketch to blink LED_BUILTIN every 425 ms using millis(), no delay(). State it is uncompiled.',
             'boardType': 'ARDUINO_UNO', 'attemptId': 'capacity-attempt',
             'research': {'documentIds': ['capacity-document']},
             'authorizedDocumentEvidence': [document]}),
    ]
    for repetition in range(3):
        for name, payload in payloads:
            started, first, count, terminal = time.monotonic(), None, 0, False
            chunks = []
            async with client.stream('POST', base + '/chat/stream', json=payload) as response:
                assert response.status_code == 200
                event = None
                async for line in response.aiter_lines():
                    if line.startswith('event: '):
                        event = line[7:]
                        assert event != 'error', 'Capacity workload returned an error'
                        terminal = terminal or event == 'complete'
                    elif event == 'token' and line.startswith('data: ') and json.loads(line[6:]).get('delta'):
                        first = first if first is not None else time.monotonic() - started
                        count += 1
                        chunks.append(json.loads(line[6:])['delta'])
            elapsed = time.monotonic() - started
            assert terminal and first is not None and count > 0
            if name == 'document_firmware':
                answer = ''.join(chunks)
                assert all(value in answer for value in ('void loop', 'millis()', '425'))
            assert elapsed <= report['profile']['maxResponseSeconds']
            samples.append({'workload': name, 'repetition': repetition + 1,
                            'firstOutputSeconds': round(first, 3), 'totalSeconds': round(elapsed, 3),
                            'chunks': count, 'peakWorkingSetMiB': process_peak_working_set(process.pid)})
    report['checks']['sequentialWorkloads'] = len(samples) == 6
    report['checks']['workingSetBudget'] = max(s['peakWorkingSetMiB'] for s in samples) <= 4096


async def verify(report=None):
    report = report if report is not None else {}
    checks = report.setdefault('checks', {})
    measurements = report.setdefault('measurements', {})
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    token = secrets.token_urlsafe(32)
    environment = dict(os.environ, VOLTFORGE_AI_PORT=str(port), VOLTFORGE_AI_HOST='127.0.0.1',
                       VOLTFORGE_AI_API_TOKEN=token, VOLTFORGE_AI_MAX_TOKENS='48',
                       VOLTFORGE_AI_MAX_PENDING_REQUESTS='0')
    capacity = report.get('scope') == 'local_single_request_capacity'
    if capacity:
        from config import get_settings
        settings = get_settings()
        with settings.resolve_model_path().open('rb') as artifact:
            artifact_hash = hashlib.file_digest(artifact, 'sha256').hexdigest()
        environment['VOLTFORGE_AI_MAX_TOKENS'] = str(settings.max_tokens)
        report['overrides']['maxTokens'] = settings.max_tokens
        report['profile'].update(contextTokens=settings.n_ctx, threads=settings.n_threads,
                                 generationTimeoutSeconds=settings.generation_timeout_seconds,
                                 maxResponseSeconds=settings.generation_timeout_seconds + 5,
                                 modelSha256=artifact_hash,
                                 memoryScope='sum of peak working sets of managed launcher and descendants')
        checks['configuredResponseBudget'] = True
    base = f'http://127.0.0.1:{port}/voltForge-ai/api/v1/model'
    with tempfile.TemporaryFile() as logs:
        process = subprocess.Popen([sys.executable, 'main.py'], cwd=ROOT, env=environment,
                                   stdout=logs, stderr=logs,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        try:
            startup_started = time.monotonic()
            async with httpx.AsyncClient(headers={'X-Voltforge-AI-Token':token}, timeout=report.get('profile', {}).get('maxResponseSeconds', 90)) as client:
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
                measurements['readinessSeconds'] = round(time.monotonic() - startup_started, 3)
                checks['nativeModelReady'] = True
                unauthorized = await client.get(base + '/health', headers={'X-Voltforge-AI-Token':'invalid'})
                assert unauthorized.status_code == 401
                checks['authentication'] = True
                oversized = await client.post(base + '/chat', content=b'x' * 2_000_001)
                assert oversized.status_code == 413
                checks['bodyLimit'] = True
                stream_started = time.monotonic()
                async with client.stream('POST', base + '/chat/stream', json={'message':'Explain what a resistor does in one sentence.'}) as response:
                    assert response.status_code == 200
                    lines = response.aiter_lines()
                    assert await anext(lines) == 'event: start'
                    measurements['firstStartEventSeconds'] = round(time.monotonic() - stream_started, 3)
                    await asyncio.sleep(.5)
                    latencies = []
                    for _ in range(5):
                        began = time.monotonic()
                        check = await client.get(base + '/health')
                        assert check.json()['ready']
                        latencies.append(time.monotonic() - began)
                        await asyncio.sleep(.05)
                    busy_started = time.monotonic()
                    busy = await client.post(base + '/chat/stream', json={'message':'A second request'})
                    measurements['busyRejectionSeconds'] = round(time.monotonic() - busy_started, 3)
                    assert 'AI_BUSY' in busy.text and 'event: complete' not in busy.text
                    checks['busyGuard'] = True
                    event = None
                    async for line in lines:
                        if line.startswith('event: '):
                            event = line[7:]
                            assert event not in ('complete', 'error'), 'Generation ended before the first token'
                        elif event == 'token' and line.startswith('data: '):
                            if json.loads(line[6:]).get('delta'):
                                measurements['firstTokenSeconds'] = round(time.monotonic() - stream_started, 3)
                                checks['incrementalNativeToken'] = True
                                break
                    else:
                        raise AssertionError('Real model did not emit a token')
                    # Closing this HTTP connection must stop native work and release its lock.
                await asyncio.sleep(1)
                began = time.monotonic()
                resumed = await client.post(base + '/chat/stream', json={'message':'In one short sentence, what is a resistor?'})
                assert 'event: token' in resumed.text and 'event: complete' in resumed.text
                assert 'event: error' not in resumed.text
                checks['disconnectThenGenerate'] = True
                measurements['maxHealthLatencySeconds'] = round(max(latencies), 3)
                measurements['completedResponseSeconds'] = round(time.monotonic()-began, 2)
                if capacity:
                    checks['boundedOverload'] = measurements['busyRejectionSeconds'] <= 2
                    checks['responsiveHealth'] = max(latencies) <= 1
                    await capacity_workloads(client, base, process, report)
        finally:
            if os.name == 'nt' and process.poll() is None:
                # Kill only this owned service tree, including a venv launcher
                # child. Terminating the wrapper alone can leak a native runtime.
                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               timeout=15, creationflags=subprocess.CREATE_NO_WINDOW)
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=10)
    return report


def write_report(output, report):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, output)


async def run_report(output, capacity=False):
    report = {'schemaVersion': 1, 'status': 'running', 'scope': 'isolated_ai_http',
              'syntheticOnly': True, 'liveProviderVerified': False, 'databasePersistenceVerified': False,
              'browserVerified': False, 'releaseAccepted': False,
              'overrides': {'maxTokens': 48, 'maxPendingRequests': 0}, 'checks': {}, 'measurements': {}}
    if capacity:
        report['scope'] = 'local_single_request_capacity'
        report['profile'] = {'activeRequests': 1, 'queuedRequests': 0, 'workloadSamples': 6,
                             'maxPeakWorkingSetMiB': 4096, 'maxBusyRejectionSeconds': 2,
                             'maxHealthSeconds': 1, 'sustainedSoakVerified': False,
                             'distributedCapacityVerified': False}
    if output:
        write_report(output, report)
    try:
        await verify(report)
        required = REQUIRED_CHECKS + (CAPACITY_CHECKS if capacity else ())
        if not all(report['checks'].get(name) is True for name in required):
            raise RuntimeError('Isolated HTTP verification did not complete every required check')
        report['status'] = 'passed'
    except BaseException as error:
        report['status'] = 'interrupted' if isinstance(error, (KeyboardInterrupt, asyncio.CancelledError)) else 'failed'
        # No exception text, subprocess logs, prompts, URLs or credentials.
        report['errorType'] = type(error).__name__
        raise
    finally:
        if output:
            write_report(output, report)
    print(json.dumps(report))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, help='Atomic, credential-free acceptance report; never a full release certificate')
    parser.add_argument('--single-request-capacity', action='store_true', help='Windows local one-active/no-queue profile with six configured-budget workloads')
    args = parser.parse_args()
    asyncio.run(run_report(args.output, args.single_request_capacity))
