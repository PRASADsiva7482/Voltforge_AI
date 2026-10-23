"""Probe real GGUF streaming with synthetic evidence, without a search credential.

This tests native prompt packing/generation only, not live retrieval or factual
quality. No private project data or full generated output is printed.
"""
import asyncio
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.llm import LLMEngine


async def main():
    engine = LLMEngine()
    started = time.monotonic()
    engine.initialize()
    if engine.llm is None:
        raise RuntimeError('Configured model could not load')
    try:
        evidence = [{'sourceId': 'synthetic-probe', 'kind': 'search_snippet',
                     'title': 'Synthetic resistor reference', 'url': 'https://example.com/',
                     'retrievedAt': '2026-10-22T00:00:00Z', 'contentHash': None,
                     'excerpt': 'A resistor limits current in a circuit.'}]
        question = 'Using the provided reference, say what a resistor does in one sentence and cite its ID.'
        used = []
        fitted = engine._fit_research(engine._build_messages(question), evidence, used)
        prompt_tokens = len(engine._prompt_tokens(fitted))
        assert prompt_tokens + engine.settings.max_tokens + 32 <= engine.settings.n_ctx
        engine.settings = engine.settings.model_copy(update={'max_tokens': 64})
        used = []
        chunks = []
        first_token = None
        async for chunk in engine.stream_chat(question, evidence=evidence, used_sources=used):
            first_token = first_token or time.monotonic()
            chunks.append(chunk)
        assert used and len(chunks) > 1 and ''.join(chunks).strip()
        print(json.dumps({'syntheticEvidenceOnly': True, 'liveSearchVerified': False,
                          'modelLoaded': True, 'promptTokens': prompt_tokens,
                          'contextTokens': engine.settings.n_ctx, 'sourceCount': len(used),
                          'nativeChunks': len(chunks), 'responseCharacters': sum(map(len, chunks)),
                          'containsExpectedReference': '[S1]' in ''.join(chunks),
                          'firstTokenSecondsIncludingLoad': round(first_token - started, 2),
                          'totalSeconds': round(time.monotonic() - started, 2)}))
    finally:
        engine.close()


if __name__ == '__main__':
    asyncio.run(main())
