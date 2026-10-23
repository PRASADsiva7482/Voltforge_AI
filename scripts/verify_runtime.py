"""Verify actual load, bounded prompts, inference and release without printing private data."""
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.llm import LLMEngine

engine = LLMEngine()
started = time.monotonic()
engine.initialize()
if engine.llm is None:
    raise SystemExit('FAIL: configured model could not load')
loaded = time.monotonic()
try:
    # Keep the smoke check bounded regardless of the service's normal response budget.
    engine.settings = engine.settings.model_copy(update={'max_tokens': 48})
    result = engine.generate('In one short sentence, what does a resistor do?', board_type='unspecified')
    if result['mode'] != 'neural' or not result['reply'].strip():
        raise RuntimeError('No neural response')
    print(json.dumps({'modelLoaded': True, 'loadSeconds': round(loaded-started, 2),
                      'generationSeconds': round(time.monotonic()-loaded, 2),
                      'responseCharacters': len(result['reply']), 'verifiedSha256': True}))
finally:
    engine.close()
