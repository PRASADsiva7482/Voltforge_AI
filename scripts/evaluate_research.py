"""Repeatable native quality probe with public synthetic inputs only.

Writes answers for rubric review. Pattern checks are not an LLM judge and do
not prove factual correctness; reviewed acceptance is recorded separately.
"""
import argparse
import asyncio
import hashlib
import json
import platform
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.llm import LLMEngine
from download_model import ensure_model, MANIFEST
from engine.evidence_guard import specification_guidance, clarification_without_context


def evaluate_answer(case, answer, error=None):
    return (error is None and bool(answer.strip())
            and all(re.search(pattern, answer) for pattern in case['expectedPatterns'])
            and not any(re.search(pattern, answer) for pattern in case['forbiddenPatterns']))


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


async def evaluate(output, fixture_path, model_manifest=None, temperature=0, max_tokens=192):
    fixture_bytes = fixture_path.read_bytes()
    fixture = json.loads(fixture_bytes)
    engine = LLMEngine()
    manifest = json.loads(model_manifest.read_text()) if model_manifest else MANIFEST
    if model_manifest:
        model_path = ensure_model(ROOT / 'model/artifacts' / manifest['filename'], manifest=manifest)
        engine.settings = engine.settings.model_copy(update={'model_path': str(model_path),
            'model_sha256': manifest['sha256'], 'model_name': manifest['name']})
    loaded = time.monotonic()
    engine.initialize()
    load_seconds = time.monotonic() - loaded
    if engine.llm is None:
        raise RuntimeError('Configured native model unavailable')
    engine.settings = engine.settings.model_copy(update={'temperature': temperature, 'max_tokens': max_tokens})
    results = []
    try:
        for case in fixture['cases']:
            sources = [{'sourceId': 'synthetic-' + str(index), 'kind': 'document_chunk',
                        'title': 'Synthetic reference ' + str(index + 1),
                        'documentId': 'synthetic', 'versionId': 'v1', 'page': index + 1,
                        'retrievedAt': '2026-10-23T00:00:00Z',
                        'excerpt': text, 'contentHash': hashlib.sha256(text.encode()).hexdigest()}
                       for index, text in enumerate(case['excerpts'])]
            started, first, chunks, used = time.monotonic(), None, [], []
            error = None
            try:
                async for chunk in engine.stream_chat(case['question'], evidence=sources, used_sources=used):
                    first = first or time.monotonic()
                    chunks.append(chunk)
            except Exception as failure:
                error = type(failure).__name__
            answer = ''.join(chunks)
            result = {'id': case['id'], 'answer': answer, 'rubric': case['rubric'],
                      'answerOrigin': ('evidence_guard' if answer and answer == specification_guidance(case['question'], used)
                                       else 'clarification' if not sources and answer == clarification_without_context(case['question'])
                                       else 'native_model'),
                      'patternChecksPassed': bool(evaluate_answer(case, answer, error)), 'error': error,
                      'nativeChunks': len(chunks), 'firstTokenSeconds': round(first - started, 2) if first else None,
                      'totalSeconds': round(time.monotonic() - started, 2),
                      'packedSources': [{'citationId': s['citationId'], 'snippet': s['snippet']} for s in used]}
            results.append(result)
            print(json.dumps({key: result[key] for key in ('id', 'patternChecksPassed', 'totalSeconds', 'error')}), flush=True)
    finally:
        engine.close()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({'model': engine.settings.model_name, 'temperature': temperature, 'maxTokens': max_tokens,
                                     'modelSha256': manifest['sha256'], 'fixtureSha256': hashlib.sha256(fixture_bytes).hexdigest(),
                                     'loadSeconds': round(load_seconds, 2),
                                     'platform': platform.platform(), 'peakProcessMemoryMiB': peak_process_memory_mib(),
                                     'contextTokens': engine.settings.n_ctx, 'threads': engine.settings.n_threads,
                                     'evidenceTokens': engine.settings.research_source_tokens,
                                     'syntheticOnly': True, 'liveSearchVerified': False,
                                     'reviewRequired': True, 'results': results}, indent=2) + '\n', encoding='utf-8')
    return all(result['patternChecksPassed'] for result in results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/AIR-06_QUALITY_EVAL.json')
    parser.add_argument('--fixture', type=Path, default=ROOT / 'tests/fixtures/research_quality_eval.json')
    parser.add_argument('--model-manifest', type=Path)
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--max-tokens', type=int, default=192)
    args = parser.parse_args()
    raise SystemExit(0 if asyncio.run(evaluate(args.output, args.fixture, args.model_manifest, args.temperature, args.max_tokens)) else 1)
