import concurrent.futures
import threading
import time
import unittest
from unittest.mock import patch

from config import Settings
from engine.llm import LLMEngine
from engine.runtime import LocalRuntime, RuntimeBusy, GenerationTimedOut, GenerationTruncated
from engine.telemetry import RuntimeTelemetry, Stage, Outcome


class TelemetryTests(unittest.TestCase):
    def test_prefill_is_native_counter_delta_not_elapsed_first_text(self):
        engine = self.engine([{'choices': [{'text': 'answer'}]}])
        with patch.object(engine, '_native_prefill_ms', side_effect=[100.0, 125.0]):
            list(self.tokens(engine))
        self.assertEqual(engine.runtime.telemetry.snapshot()['latencyMs']['native_prefill']['maxMs'], 25.0)
        # A reset/decreasing counter cannot fabricate a negative or total-time sample.
        with patch.object(engine, '_native_prefill_ms', side_effect=[125.0, 0.0]):
            list(self.tokens(engine))
        self.assertEqual(engine.runtime.telemetry.snapshot()['latencyMs']['native_prefill']['count'], 1)
    def test_bounded_thread_safe_aggregates_and_label_redaction(self):
        telemetry = RuntimeTelemetry()
        def record(_):
            telemetry.record(Stage.QUEUE_WAIT, Outcome.SUCCESS, .01)
            telemetry.record('private user', 'private exception', 1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(record, range(1000)))
        telemetry.record(Stage.QUEUE_WAIT, Outcome.SUCCESS, float('nan'))
        snapshot = telemetry.snapshot()
        self.assertEqual(snapshot['latencyMs']['queue_wait']['count'], 256)
        self.assertEqual(snapshot['outcomes'], {'queue_wait.success': 1000})
        self.assertNotIn('private', str(snapshot))
        snapshot['outcomes'].clear()
        self.assertEqual(telemetry.snapshot()['outcomes']['queue_wait.success'], 1000)

    def test_queue_rejection_timeout_cancellation_and_slot_recovery(self):
        runtime = LocalRuntime(Settings(max_pending_requests=0, queue_timeout_seconds=.1))
        cancelled = threading.Event()
        with runtime.admission(cancelled):
            with self.assertRaises(RuntimeBusy):
                with runtime.admission(cancelled):
                    self.fail('Overload admitted')
        runtime.gate.acquire()
        try:
            with self.assertRaises(RuntimeBusy):
                with runtime.admission(cancelled):
                    self.fail('Locked runtime admitted')
            cancelled.set()
            with self.assertRaises(RuntimeBusy):
                with runtime.admission(cancelled):
                    self.fail('Cancelled work admitted')
        finally:
            runtime.gate.release()
        cancelled.clear()
        with runtime.admission(cancelled):
            pass
        counts = runtime.telemetry.snapshot()['outcomes']
        self.assertEqual(counts, {'queue_wait.success': 2, 'queue_wait.rejected': 1,
                                'queue_wait.timeout': 1, 'queue_wait.cancelled': 1})

    def engine(self, chunks):
        class Model:
            def tokenize(self, text, **kwargs): return list(text)
            def create_completion(self, **kwargs): yield from chunks
        engine = LLMEngine()
        engine.llm = Model()
        return engine

    def tokens(self, engine, cancelled=None):
        return engine._completion_tokens([{'role': 'user', 'content': 'private prompt'}],
            cancelled or threading.Event(), time.monotonic() + 10, 100, .2)

    def test_native_first_text_only_once_and_empty_completion_has_no_first_text(self):
        for texts, expected in [(['', 'private output', ' more'], 1), (['', ''], 0)]:
            engine = self.engine([{'choices': [{'text': text}]} for text in texts])
            self.assertEqual(''.join(self.tokens(engine)), ''.join(texts))
            snapshot = engine.runtime.telemetry.snapshot()
            self.assertEqual(snapshot['latencyMs']['native_first_text']['count'], expected)
            self.assertEqual(snapshot['outcomes']['native_completion.success'], 1)
            self.assertNotIn('private', str(snapshot))

    def test_native_close_truncation_failure_and_timeout_keep_original_behavior(self):
        engine = self.engine([{'choices': [{'text': 'partial'}]}])
        tokens = self.tokens(engine)
        self.assertEqual(next(tokens), 'partial')
        tokens.close()
        self.assertEqual(engine.runtime.telemetry.snapshot()['outcomes']['native_completion.cancelled'], 1)
        engine = self.engine([{'choices': [{'text': 'partial', 'finish_reason': 'length'}]}])
        with self.assertRaises(GenerationTruncated):
            list(self.tokens(engine))
        self.assertEqual(engine.runtime.telemetry.snapshot()['outcomes']['native_completion.truncated'], 1)
        for error, outcome in [(RuntimeError('private error'), 'failure'),
                               (GenerationTimedOut('private timeout'), 'timeout')]:
            def broken():
                raise error
                yield
            engine = self.engine(broken())
            with self.assertRaises(type(error)) as caught:
                list(self.tokens(engine))
            self.assertIs(caught.exception, error)
            snapshot = engine.runtime.telemetry.snapshot()
            self.assertEqual(snapshot['outcomes'][f'native_completion.{outcome}'], 1)
            self.assertNotIn('private', str(snapshot))
