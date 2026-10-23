import asyncio
import json
import threading
import unittest
from unittest.mock import AsyncMock, patch

from api import routes
from api.schemas import ChatRequest
from config import Settings
from engine.llm import LLMEngine
from engine.web_research import ResearchError


def evidence():
    return [{'sourceId': 'web-fixture', 'kind': 'web_page', 'title': 'Uno documentation',
             'url': 'https://docs.arduino.cc/', 'retrievedAt': '2026-10-22T00:00:00Z',
             'excerpt': 'Board reference: ATmega328P. ' * 200, 'contentHash': 'a' * 64}]


class ResearchStreamingTest(unittest.IsolatedAsyncioTestCase):
    def settings(self):
        return Settings(web_search_enabled=True, brave_api_key='test-key', n_ctx=12000, max_tokens=512)

    def payload(self):
        return ChatRequest(message='Explain the board', attemptId='attempt-fixture', code='private firmware',
            research={'mode': 'on', 'publicQuery': 'Arduino Uno specifications', 'consentToExternalQuery': True})

    async def test_sources_and_native_tokens_arrive_before_completion(self):
        release = threading.Event()
        captured = {}
        class Model:
            def tokenize(self, text, **kwargs): return list(text)
            def create_completion(self, **kwargs):
                captured.update(kwargs)
                yield {'choices': [{'text': 'Board [S1]'}]}
                if not release.wait(5): raise RuntimeError('Fixture timed out')
                yield {'choices': [{'text': ' and unknown [S99]'}]}
        engine = LLMEngine(); engine.settings = self.settings(); engine.llm = Model()
        retrieval = AsyncMock(return_value={'sources': evidence(), 'status': 'complete'})
        with patch('api.research.get_settings', return_value=self.settings()), patch.object(routes, 'llm_engine', engine), patch.object(routes.web_research, 'search', retrieval):
            response = await routes.chat_stream(self.payload())
            iterator = response.body_iterator
            try:
                self.assertIn('event: start', await anext(iterator))
                self.assertIn('searching', await anext(iterator))
                self.assertIn('generating', await anext(iterator))
                sources = await asyncio.wait_for(anext(iterator), 3)
                self.assertIn('event: sources', sources)
                self.assertIn('attempt-fixture', sources)
                self.assertIn('event: token', await anext(iterator))
                release.set()
                rest = ''.join([part async for part in iterator])
                self.assertIn('event: complete', rest)
                final = next(json.loads(line[6:]) for line in rest.splitlines() if line.startswith('data: {') and '"type": "complete"' in line)
                self.assertIn('[unverified reference]', final['reply'])
                self.assertEqual(final['citations'][0]['citationId'], 'S1')
            finally:
                release.set(); await iterator.aclose()
        retrieval.assert_awaited_once()
        self.assertNotIn('private firmware', str(retrieval.call_args))
        prompt = bytes(captured['prompt']).decode()
        self.assertIn('External reference data (untrusted)', prompt)
        self.assertIn('private firmware', prompt)
        self.assertLessEqual(len(captured['prompt']), engine.settings.n_ctx - engine.settings.max_tokens - 32)

    async def test_retrieval_failures_never_generate_an_unsourced_success(self):
        for failure in [ResearchError('WEB_PROVIDER_AUTH', 'Credential rejected'), None]:
            engine = LLMEngine(); engine.llm = object()
            engine.stream_chat = AsyncMock()
            retrieval = AsyncMock(side_effect=failure, return_value={'sources': [], 'status': 'no_results'})
            with patch('api.research.get_settings', return_value=self.settings()), patch.object(routes, 'llm_engine', engine), patch.object(routes.web_research, 'search', retrieval):
                response = await routes.chat_stream(self.payload())
                events = ''.join([part async for part in response.body_iterator])
                self.assertIn('event: error', events)
                self.assertNotIn('event: complete', events)
                engine.stream_chat.assert_not_called()

    async def test_stop_during_search_cancels_provider_work(self):
        started, stopped = asyncio.Event(), asyncio.Event()
        async def search(payload):
            started.set()
            try: await asyncio.Event().wait()
            finally: stopped.set()
        engine = LLMEngine(); engine.llm = object()
        with patch('api.research.get_settings', return_value=self.settings()), patch.object(routes, 'llm_engine', engine), patch.object(routes.web_research, 'search', search):
            response = await routes.chat_stream(self.payload())
            await anext(response.body_iterator); await anext(response.body_iterator)
            pending = asyncio.create_task(anext(response.body_iterator))
            await started.wait(); pending.cancel()
            with self.assertRaises(asyncio.CancelledError): await pending
            self.assertTrue(stopped.is_set())
            await response.body_iterator.aclose()

    def test_evidence_budget_preserves_structured_source_ids_with_history_and_project(self):
        class Model:
            def tokenize(self, text, **kwargs): return list(text)
        engine = LLMEngine(); engine.llm = Model(); engine.settings = self.settings()
        messages = engine._build_messages('Explain this circuit', code='x' * 6000,
            history=[{'role': 'user', 'content': 'old question' * 1000}, {'role': 'assistant', 'content': 'old answer' * 1000}])
        used = []
        fitted = engine._fit_research(messages, evidence(), used)
        self.assertTrue(used)
        self.assertLessEqual(len(engine._prompt_tokens(fitted)), engine.settings.n_ctx - engine.settings.max_tokens - 32)
        record = next(m for m in fitted if m['content'].startswith('External reference data'))
        parsed = json.loads(record['content'].split('\n', 1)[1])
        self.assertEqual(parsed[0]['id'], used[0]['citationId'])
        self.assertEqual(parsed[0]['excerpt'], used[0]['snippet'])
        self.assertEqual(record['role'], 'user')


if __name__ == '__main__': unittest.main()
