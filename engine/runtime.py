"""One native model owner, bounded waiting, and cancellation-aware token delivery."""
import asyncio
from contextlib import contextmanager
import queue
import threading
import time


class RuntimeBusy(RuntimeError):
    pass


class GenerationTimedOut(RuntimeError):
    pass


class LocalRuntime:
    def __init__(self, settings):
        self.settings = settings
        self.gate = threading.Lock()
        self.slots = threading.BoundedSemaphore(settings.max_pending_requests + 1)
        self.shutdown = threading.Event()

    @contextmanager
    def admission(self, cancelled):
        if self.shutdown.is_set() or not self.slots.acquire(blocking=False):
            raise RuntimeBusy('AI is busy. Retry shortly.')
        acquired = False
        try:
            deadline = time.monotonic() + self.settings.queue_timeout_seconds
            while not acquired:
                if cancelled.is_set() or self.shutdown.is_set():
                    raise RuntimeBusy('Request cancelled')
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeBusy('AI queue wait expired. Retry shortly.')
                acquired = self.gate.acquire(timeout=min(remaining, 0.1))
            yield
        finally:
            if acquired:
                self.gate.release()
            self.slots.release()

    async def stream(self, producer):
        cancelled = threading.Event()
        output = queue.Queue(maxsize=32)
        done = object()

        def put(value):
            while not cancelled.is_set():
                try:
                    output.put(value, timeout=0.1)
                    return
                except queue.Full:
                    continue

        def work():
            try:
                with self.admission(cancelled):
                    iterator = producer(cancelled)
                    try:
                        for item in iterator:
                            if cancelled.is_set():
                                break
                            put(item)
                    finally:
                        close = getattr(iterator, 'close', None)
                        if close:
                            close()
            except Exception as error:
                put(error)
            finally:
                put(done)

        worker = threading.Thread(target=work, name='voltforge-inference', daemon=True)
        worker.start()
        try:
            while True:
                try:
                    item = await asyncio.to_thread(output.get, True, 0.2)
                except queue.Empty:
                    continue
                if item is done:
                    return
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            # Native calls cannot be killed safely. The producer stops at the next
            # token boundary and retains its lock until the iterator is closed.
            cancelled.set()
