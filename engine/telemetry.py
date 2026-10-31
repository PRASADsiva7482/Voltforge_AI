"""Bounded, content-free internal timing aggregates; no HTTP/export surface."""
from collections import Counter, deque
from enum import Enum
import math
import threading


class Stage(Enum):
    QUEUE_WAIT = 'queue_wait'
    NATIVE_FIRST_TEXT = 'native_first_text'
    NATIVE_COMPLETION = 'native_completion'
    NATIVE_PREFILL = 'native_prefill'


class Outcome(Enum):
    SUCCESS = 'success'
    REJECTED = 'rejected'
    CANCELLED = 'cancelled'
    TIMEOUT = 'timeout'
    TRUNCATED = 'truncated'
    FAILURE = 'failure'


class RuntimeTelemetry:
    def __init__(self):
        self._lock = threading.Lock()
        self._samples = {stage: deque(maxlen=256) for stage in Stage}
        self._outcomes = Counter()

    def record(self, stage, outcome, seconds):
        # Never retain arbitrary labels, content, exceptions, or request IDs.
        if not isinstance(stage, Stage) or not isinstance(outcome, Outcome):
            return
        if not math.isfinite(seconds):
            return
        milliseconds = max(0.0, min(600_000.0, seconds * 1000))
        with self._lock:
            self._samples[stage].append(milliseconds)
            self._outcomes[(stage, outcome)] += 1

    def snapshot(self):
        with self._lock:
            timings = {}
            for stage, samples in self._samples.items():
                ordered = sorted(samples)
                timings[stage.value] = {
                    'count': len(ordered),
                    'p50Ms': ordered[int((len(ordered) - 1) * .50)] if ordered else None,
                    'p95Ms': ordered[int((len(ordered) - 1) * .95)] if ordered else None,
                    'maxMs': ordered[-1] if ordered else None,
                }
            return {
                'latencyMs': timings,
                'outcomes': {f'{stage.value}.{outcome.value}': count
                             for (stage, outcome), count in self._outcomes.items()},
            }
