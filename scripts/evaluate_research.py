"""Repeatable native quality probe with public synthetic inputs only.

Writes answers for rubric review. Pattern checks are not an LLM judge and do
not prove factual correctness; reviewed acceptance is recorded separately.
"""
import argparse
import asyncio
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
import os
import platform
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.llm import LLMEngine
from download_model import ensure_model, digest


def evaluate_answer(case, answer, error=None):
    return (error is None and bool(answer.strip())
            and ('maxWords' not in case or len(answer.split()) <= case['maxWords'])
            and all(re.search(pattern, answer) for pattern in case['expectedPatterns'])
            and not any(re.search(pattern, answer) for pattern in case['forbiddenPatterns']))


def load_fixture(fixture_bytes):
    fixture = json.loads(fixture_bytes)
    cases = fixture.get('cases')
    if not isinstance(cases, list) or not cases:
        raise ValueError('Evaluation needs at least one case')
    identifiers = set()
    for case in cases:
        for field in ('id', 'question', 'rubric'):
            if not isinstance(case.get(field), str) or not case[field].strip():
                raise ValueError('Case needs a nonempty ' + field)
        if case['id'] in identifiers:
            raise ValueError('Duplicate evaluation case ID')
        identifiers.add(case['id'])
        for field in ('excerpts', 'expectedPatterns', 'forbiddenPatterns'):
            if not isinstance(case.get(field), list) or not all(isinstance(x, str) for x in case[field]):
                raise ValueError('Case needs a string list: ' + field)
        if not case['expectedPatterns']:
            raise ValueError('Case needs an acceptance pattern')
        if 'maxWords' in case and (type(case['maxWords']) is not int or case['maxWords'] < 1):
            raise ValueError('maxWords must be a positive integer')
        for pattern in case['expectedPatterns'] + case['forbiddenPatterns']:
            re.compile(pattern)
        if 'requestContext' in case:
            if fixture.get('syntheticOnly') is not True:
                raise ValueError('Context evaluations must declare syntheticOnly')
            validate_request_context(case['requestContext'])
    return fixture


def validate_request_context(context):
    """Allow existing engine inputs only; never fixture-controlled runtime settings."""
    fields = {'board_type': str, 'code': str, 'context': str, 'history': list,
              'components': list, 'wires': list, 'simulation_state': dict, 'preferences': dict}
    if not isinstance(context, dict) or set(context) - fields.keys():
        raise ValueError('Unknown evaluation context field')
    for key, value in context.items():
        if not isinstance(value, fields[key]):
            raise ValueError('Invalid evaluation context type: ' + key)
        if key in ('components', 'wires') and not all(isinstance(item, dict) for item in value):
            raise ValueError('Context components/wires must be object lists')
        if key == 'history':
            for message in value:
                if (not isinstance(message, dict) or set(message) != {'role', 'content'}
                        or message['role'] not in ('user', 'assistant')
                        or not isinstance(message['content'], str)):
                    raise ValueError('Invalid evaluation history message')
    if len(json.dumps(context, ensure_ascii=False)) > 24000:
        raise ValueError('Evaluation context exceeds synthetic fixture limit')


def write_report(output, report):
    # A terminated native probe must leave the completed cases reviewable, not
    # an old report that appears to describe this run or a partially written JSON.
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, output)


def answer_origin(metadata):
    if metadata.get('mode') == 'clarification' and 'capabilityDecision' in metadata:
        return 'capability_clarification'
    return {'evidence-guidance': 'evidence_guard', 'clarification': 'clarification',
            'calculation': 'deterministic_calculation'}.get(
        metadata.get('mode', 'neural'), 'native_model')


def native_runtime_version():
    try:
        return version('llama-cpp-python')
    except PackageNotFoundError:
        # Initialization will report model unavailability. Missing package
        # metadata must not prevent the failed-run checkpoint or mocked tests.
        return None


def peak_process_memory_mib():
    if sys.platform != 'win32':
        return None
    import ctypes
    from ctypes import wintypes
    class Memory(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
                'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
    memory = Memory()
    memory.cb = ctypes.sizeof(memory)
    if ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(-1), ctypes.byref(memory), memory.cb):
        return round(memory.PeakWorkingSetSize / 1024**2, 2)
    return None


async def evaluate(output, fixture_path, model_manifest=None, temperature=0, max_tokens=192, seed=42, batch_threads=None, answer_policy=None):
    fixture_bytes = fixture_path.read_bytes()
    fixture = load_fixture(fixture_bytes)
    engine = LLMEngine()
    # Validate CLI overrides using the same bounds as the service; model_copy
    # alone bypasses validators and previously allowed impossible token budgets.
    overrides = {'temperature': temperature, 'max_tokens': max_tokens}
    if batch_threads is not None:
        overrides['n_threads_batch'] = batch_threads
    if answer_policy is not None:
        overrides['answer_policy'] = answer_policy
    engine.settings = type(engine.settings).model_validate({**engine.settings.model_dump(), **overrides})
    engine.runtime.settings = engine.settings
    results = []
    report = {'status': 'initializing', 'model': engine.settings.model_name,
              'temperature': temperature, 'maxTokens': max_tokens, 'seed': seed,
              'modelSha256': None, 'fixtureSha256': hashlib.sha256(fixture_bytes).hexdigest(),
              'caseCount': len(fixture['cases']), 'completedCases': 0,
              'platform': platform.platform(), 'processor': platform.processor(),
              'pythonVersion': platform.python_version(), 'llamaCppVersion': native_runtime_version(),
              'contextTokens': engine.settings.n_ctx, 'threads': engine.settings.n_threads,
              'gpuLayers': engine.settings.n_gpu_layers,
              'generationTimeoutSeconds': engine.settings.generation_timeout_seconds,
              'answerPolicy': engine.settings.answer_policy,
              'evidenceTokens': engine.settings.research_source_tokens,
              'runtimeSourceSha256': {name: digest(ROOT / name) for name in (
                  'engine/llm.py', 'engine/runtime.py', 'engine/research_workflow.py', 'engine/evidence_guard.py')},
              'syntheticOnly': True, 'liveSearchVerified': False,
              'reviewRequired': True, 'results': results}
    write_report(output, report)
    try:
        if model_manifest:
            manifest = json.loads(model_manifest.read_text(encoding='utf-8'))
            report['model'] = manifest['name']
            write_report(output, report)
            model_path = ensure_model(ROOT / 'model/artifacts' / manifest['filename'], manifest=manifest)
            engine.settings = engine.settings.model_copy(update={'model_path': str(model_path),
                'model_sha256': manifest['sha256'], 'model_name': manifest['name']})
            engine.runtime.settings = engine.settings
        report['model'] = engine.settings.model_name
        loaded = time.monotonic()
        engine.initialize()
        report['loadSeconds'] = round(time.monotonic() - loaded, 2)
        if engine.llm is None:
            raise RuntimeError('Configured native model unavailable')
        # llama-cpp can use more threads for prompt batches than for decoding.
        # Record its actual values rather than implying n_threads covers both.
        for field, attribute in (('promptBatchThreads', 'n_threads_batch'), ('promptBatchSize', 'n_batch')):
            value = getattr(engine.llm, attribute, None)
            report[field] = value if isinstance(value, int) else None
        # Hash what was actually loaded, not the default manifest when a custom
        # model is configured. Initialization already verifies artifact trust.
        report['modelSha256'] = digest(engine.settings.resolve_model_path())
        report['status'] = 'running'
        write_report(output, report)
        for case in fixture['cases']:
            engine.llm.set_seed(seed)
            sources = [{'sourceId': 'synthetic-' + str(index), 'kind': 'document_chunk',
                        'title': 'Synthetic reference ' + str(index + 1),
                        'documentId': 'synthetic', 'versionId': 'v1', 'page': index + 1,
                        'retrievedAt': '2026-10-23T00:00:00Z',
                        'excerpt': text, 'contentHash': hashlib.sha256(text.encode()).hexdigest()}
                       for index, text in enumerate(case['excerpts'])]
            started, first, chunks, used = time.monotonic(), None, [], []
            cpu_started, metadata = time.process_time(), {}
            error = None
            try:
                async for chunk in engine.stream_chat(case['question'], evidence=sources, used_sources=used,
                                                     answer_metadata=metadata, **case.get('requestContext', {})):
                    first = first or time.monotonic()
                    chunks.append(chunk)
            except Exception as failure:
                error = type(failure).__name__
            answer = ''.join(chunks)
            origin = answer_origin(metadata)
            result = {'id': case['id'], 'answer': answer, 'rubric': case['rubric'],
                      'answerOrigin': origin,
                      'capabilityDecision': metadata.get('capabilityDecision'),
                      'assessmentTokens': metadata.get('assessmentTokens', 0),
                      'assessmentTokenLimit': metadata.get('assessmentTokenLimit', 0),
                      'assessmentSeconds': metadata.get('assessmentSeconds', 0),
                      'contextFields': sorted(case.get('requestContext', {})),
                      'requestContextSha256': hashlib.sha256(json.dumps(case['requestContext'], sort_keys=True).encode()).hexdigest() if 'requestContext' in case else None,
                      'patternChecksPassed': bool(evaluate_answer(case, answer, error)), 'error': error,
                      'wordCount': len(answer.split()),
                      'streamChunks': len(chunks), 'nativeChunks': len(chunks) if origin == 'native_model' else 0,
                      'firstOutputSeconds': round(first - started, 2) if first else None,
                      'firstTokenSeconds': round(first - started, 2) if first and origin == 'native_model' else None,
                      'totalSeconds': round(time.monotonic() - started, 2),
                      'processCpuSeconds': round(time.process_time() - cpu_started, 2),
                      'packedSources': [{'citationId': s['citationId'], 'snippet': s['snippet']} for s in used]}
            results.append(result)
            report['completedCases'] = len(results)
            report['peakProcessMemoryMiB'] = peak_process_memory_mib()
            write_report(output, report)
            print(json.dumps({key: result[key] for key in ('id', 'patternChecksPassed', 'totalSeconds', 'error')}), flush=True)
        report['status'] = 'complete'
    except BaseException as failure:
        report['status'] = 'interrupted' if isinstance(failure, (KeyboardInterrupt, asyncio.CancelledError)) else 'failed'
        report['error'] = type(failure).__name__
        raise
    finally:
        engine.close()
        report['peakProcessMemoryMiB'] = peak_process_memory_mib()
        write_report(output, report)
    return len(results) == len(fixture['cases']) and all(result['patternChecksPassed'] for result in results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'test-results/research/latest.json')
    parser.add_argument('--fixture', type=Path, default=ROOT / 'tests/fixtures/research_quality_eval.json')
    parser.add_argument('--model-manifest', type=Path)
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--max-tokens', type=int, default=192)
    parser.add_argument('--seed', type=int, default=42, help='Reset native sampling seed before each case; cross-hardware bitwise identity is not guaranteed')
    parser.add_argument('--batch-threads', type=int, help='Evaluation override for prompt-processing threads; does not edit service configuration')
    parser.add_argument('--answer-policy', choices=['baseline', 'focused'], help='Operator-only prompt policy for this probe; default preserves configured policy')
    args = parser.parse_args()
    raise SystemExit(0 if asyncio.run(evaluate(args.output, args.fixture, args.model_manifest, args.temperature, args.max_tokens, args.seed, args.batch_threads, args.answer_policy)) else 1)
