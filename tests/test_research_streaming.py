import asyncio
import json
import threading
import unittest
from unittest.mock import AsyncMock, Mock, patch

from api import routes
from api.schemas import ChatRequest
from config import Settings
from engine.llm import LLMEngine
from engine.web_research import ResearchError
from engine.runtime import GenerationTimedOut, GenerationTruncated
import time
from pathlib import Path


def evidence():
    return [{'sourceId': 'web-fixture', 'kind': 'web_page', 'title': 'Uno documentation',
             'url': 'https://docs.arduino.cc/', 'retrievedAt': '2026-10-22T00:00:00Z',
             'excerpt': 'Board reference: ATmega328P. ' * 200, 'contentHash': 'a' * 64}]


class ResearchStreamingTest(unittest.IsolatedAsyncioTestCase):
    async def test_shared_stream_contract_preserves_unicode_and_terminal_reply(self):
        fixture = json.loads((Path(__file__).resolve().parents[1] / 'docs/contracts/chat-stream-v1.json').read_text(encoding='utf-8'))
        engine = LLMEngine(); engine.llm = object()
        async def generate(**kwargs):
            yield fixture['answer']
        engine.stream_chat = generate
        with patch.object(routes, 'llm_engine', engine):
            response = await routes.chat_stream(ChatRequest(message='Explain the resistor', attemptId=fixture['attemptId']))
            wire = ''.join([part async for part in response.body_iterator])
        events = [json.loads(line[6:]) for line in wire.splitlines() if line.startswith('data: ') and line != 'data: [DONE]']
        self.assertEqual([e['type'] for e in events], [e['event'] for e in fixture['upstreamFrames']])
        self.assertEqual(events[1]['delta'], fixture['upstreamFrames'][1]['data']['delta'])
        self.assertEqual(events[-1]['reply'], fixture['answer'])
        self.assertEqual(events[-1]['attemptId'], fixture['attemptId'])

    firmware_question = 'Write Arduino code to blink LED_BUILTIN using millis(), no delay().'

    def firmware_engine(self, decision='yes'):
        engine = LLMEngine()
        engine.settings = Settings(n_ctx=12000, max_tokens=768, research_source_tokens=1200)
        engine.llm = Mock()
        engine.llm.tokenize.side_effect = lambda value, **kwargs: list(value)
        engine.llm.create_completion.side_effect = [
            iter([{'choices': [{'text': json.dumps({'claims': [{'sourceId': 'S1', 'support': decision}]})}]}]),
            iter([{'choices': [{'text': 'First delta'}]}, {'choices': [{'text': ' last delta'}]}])]
        return engine

    def firmware_source(self):
        return {**evidence()[0], 'kind': 'document_chunk', 'url': None, 'documentId': 'doc-fixture',
                'versionId': 'v1', 'page': 1, 'excerpt': 'Arduino Uno R3 supports millis(), pinMode and digitalWrite.'}

    async def test_capability_assessment_is_hidden_and_preserves_synthesis_context_and_budget(self):
        engine = self.firmware_engine()
        metadata, used = {}, []
        code = 'const byte ledPin = 6;'
        result = [token async for token in engine.stream_chat(self.firmware_question, evidence=[self.firmware_source()],
            used_sources=used, answer_metadata=metadata, board_type='ARDUINO_UNO', code=code,
            history=[{'role': 'user', 'content': 'Use 650 milliseconds.'}])]
        self.assertEqual(result, ['First delta', ' last delta'])
        assessment, synthesis = [call.kwargs for call in engine.llm.create_completion.call_args_list]
        self.assertIn('grammar', assessment)
        self.assertEqual(assessment['temperature'], 0)
        self.assertNotIn('grammar', synthesis)
        self.assertIn('ARDUINO_UNO', bytes(assessment['prompt']).decode())
        for value in (code, 'Use 650 milliseconds.'):
            self.assertIn(value, bytes(synthesis['prompt']).decode())
        self.assertIn('unsigned long previous', bytes(synthesis['prompt']).decode())
        self.assertEqual(metadata['capabilityDecision'], 'supported')
        self.assertEqual(synthesis['max_tokens'] + assessment['max_tokens'], engine.settings.max_tokens)
        self.assertEqual(metadata['assessmentTokenLimit'], assessment['max_tokens'])
        for call in (assessment, synthesis):
            self.assertLessEqual(len(call['prompt']) + call['max_tokens'] + 32, engine.settings.n_ctx)

    async def test_capability_clarification_keeps_document_sse_without_code_proposals(self):
        engine = self.firmware_engine('unspecified')
        payload = ChatRequest(message=self.firmware_question, attemptId='attempt-capability',
            research={'documentIds': ['doc-fixture']}, authorizedDocumentEvidence=[self.firmware_source()])
        with patch('api.research.get_settings', return_value=Settings()), patch.object(routes, 'llm_engine', engine), \
             patch.object(routes, 'workspace_wire_suggestions') as wires, \
             patch.object(routes, 'validate_circuit_state') as validate, \
             patch.object(routes.web_research, 'search', AsyncMock()) as search:
            response = await routes.chat_stream(payload)
            events = ''.join([part async for part in response.body_iterator])
        self.assertLess(events.index('event: sources'), events.index('event: token'))
        self.assertIn('event: complete', events)
        self.assertIn('attempt-capability', events)
        self.assertIn('[S1]', events)
        self.assertIn('"mode": "clarification"', events)
        self.assertNotIn('event: proposal', events)
        self.assertNotIn('"claims"', events)
        self.assertNotIn('capabilityDecision', events)
        engine.llm.create_completion.assert_called_once()
        wires.assert_not_called(); validate.assert_not_called(); search.assert_not_awaited()

    async def test_malformed_or_partial_evidence_never_falls_through_to_code(self):
        for source, answer in [(self.firmware_source(), '{invalid'),
                               ({**self.firmware_source(), 'excerpt': 'millis supported. ' * 400}, None)]:
            engine = self.firmware_engine()
            engine.llm.create_completion.side_effect = [iter([{'choices': [{'text': answer}]}])]
            result = ''.join([token async for token in engine.stream_chat(self.firmware_question,
                evidence=[source], used_sources=[])])
            self.assertIn('cannot establish support', result)
            self.assertNotIn('First delta', result)
            self.assertEqual(engine.llm.create_completion.call_count, 0 if answer is None else 1)

    async def test_missing_api_and_unmatched_board_skip_native_assessment_and_code(self):
        for excerpt in ('Arduino Uno R3 supports digitalWrite.',
                        'Arduino Uno R4 supports millis().',
                        'Arduino Uno R3 supports millis(). Errata: millis() is unavailable.',
                        'Arduino Uno R3 supports millis() only if its timer is enabled.',
                        'The unspecified target supports millis().'):
            engine = self.firmware_engine()
            metadata = {}
            reply = ''.join([token async for token in engine.stream_chat(self.firmware_question,
                board_type='ARDUINO_UNO', evidence=[{**self.firmware_source(), 'excerpt': excerpt}],
                used_sources=[], answer_metadata=metadata)])
            self.assertIn('cannot establish support', reply)
            self.assertEqual(metadata['capabilityDecision'], 'unknown')
            self.assertEqual(metadata['assessmentTokenLimit'], 0)
            engine.llm.create_completion.assert_not_called()

    async def test_assessment_cancel_closes_native_iterator_and_does_not_synthesize(self):
        engine = self.firmware_engine()
        entered, closed = threading.Event(), threading.Event()
        def assessment(**kwargs):
            entered.set()
            try:
                while not kwargs['stopping_criteria']([], []):
                    time.sleep(.01)
                yield {'choices': [{'text': 'never-visible'}]}
            finally:
                closed.set()
        engine.llm.create_completion.side_effect = assessment
        stream = engine.stream_chat(self.firmware_question, evidence=[self.firmware_source()], used_sources=[])
        pending = asyncio.create_task(anext(stream))
        self.assertTrue(await asyncio.to_thread(entered.wait, 3))
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError): await pending
        await stream.aclose()
        self.assertTrue(await asyncio.to_thread(closed.wait, 3))
        engine.llm.create_completion.assert_called_once()
        with engine.runtime.admission(threading.Event()): pass

    def test_assessment_and_synthesis_share_one_deadline_and_errors_never_become_success(self):
        for reason, expected in [('length', GenerationTruncated), ('timeout', GenerationTimedOut)]:
            engine = self.firmware_engine()
            def assessment(**kwargs):
                if reason == 'timeout':
                    clock[0] = 1000
                yield {'choices': [{'text': '', 'finish_reason': reason}]}
            clock = [0]
            engine.llm.create_completion.side_effect = assessment
            with patch('engine.llm.time.monotonic', side_effect=lambda: clock[0]), self.assertRaises(expected):
                list(engine._native_tokens(threading.Event(), self.firmware_question,
                    evidence=[self.firmware_source()], used_sources=[]))
            engine.llm.create_completion.assert_called_once()
        engine = self.firmware_engine()
        observed = []
        def stage(messages, cancelled, deadline, max_tokens, temperature, grammar=None):
            observed.append(deadline)
            yield json.dumps({'claims': [{'sourceId': 'S1', 'support': 'yes'}]}) if grammar else 'answer'
        with patch.object(engine, '_completion_tokens', side_effect=stage):
            self.assertEqual(''.join(engine._native_tokens(threading.Event(), self.firmware_question,
                evidence=[self.firmware_source()], used_sources=[])), 'answer')
        self.assertEqual(len(observed), 2)
        self.assertEqual(observed[0], observed[1])

    async def test_serial_configuration_guidance_preserves_offline_document_sse(self):
        engine = LLMEngine(); engine.settings = self.settings(); engine.llm = Mock()
        engine.llm.tokenize.side_effect = lambda value, **kwargs: list(value)
        source = {**evidence()[0], 'kind': 'document_chunk', 'url': None,
            'documentId': 'doc-fixture', 'versionId': 'v1', 'page': 1,
            'excerpt': 'User-shared firmware calls Serial.begin(19200). The serial monitor is configured to 9600 baud. No runtime trace or compiler output is available.'}
        payload = ChatRequest(message='Why might my serial text look garbled? Use only the provided configuration and suggest one check. Do not claim you ran the firmware.',
            attemptId='attempt-serial', research={'documentIds': ['doc-fixture']}, authorizedDocumentEvidence=[source])
        with patch('api.research.get_settings', return_value=Settings()), patch.object(routes, 'llm_engine', engine), \
             patch.object(routes.web_research, 'search', AsyncMock()) as search:
            response = await routes.chat_stream(payload)
            events = ''.join([part async for part in response.body_iterator])
        self.assertLess(events.index('event: sources'), events.index('event: token'))
        self.assertLess(events.index('event: token'), events.index('event: complete'))
        self.assertIn('possible cause', events)
        self.assertIn('19200 baud', events)
        self.assertIn('9600 baud', events)
        self.assertIn('"mode": "evidence-guidance"', events)
        self.assertIn('[S1]', events)
        self.assertIn('attempt-serial', events)
        self.assertNotIn('event: error', events)
        engine.llm.create_completion.assert_not_called()
        search.assert_not_awaited()

    async def test_serial_checks_never_discard_live_context_history_or_source_tails(self):
        text = 'Firmware baud rate: 19200; Serial monitor baud rate: 9600.'
        contexts = [{'history': [{'role': 'user', 'content': 'I changed it to 19200.'}]},
                    {'components': [{'id': 'board'}]}, {'wires': [{'from': 'rx'}]},
                    {'code': 'Serial.begin(9600);'}, {'board_type': 'ARDUINO_LEONARDO'},
                    {'simulation_state': {'baudRate': 19200}}, {'context': 'USB CDC port'},
                    {'preferences': {'response_detail': 'detailed'}}]
        for context, suffix in [(c, '') for c in contexts] + [({}, ' This uses USB CDC.')]:
            with self.subTest(context=context, suffix=suffix):
                engine = LLMEngine(); engine.settings = self.settings(); engine.llm = Mock()
                engine.llm.tokenize.side_effect = lambda value, **kwargs: list(value)
                engine.llm.create_completion.return_value = iter([{'choices': [{'text': 'Native context answer'}]}])
                used = [{'citationId': 'S1', 'sourceId': 'doc', 'snippet': text}]
                with patch.object(engine, '_fit_research', side_effect=lambda messages, *args: messages):
                    answer = ''.join([x async for x in engine.stream_chat('Check the serial baud settings.',
                        evidence=[{**evidence()[0], 'excerpt': text + suffix}], used_sources=used, **context)])
                self.assertEqual(answer, 'Native context answer')
                engine.llm.create_completion.assert_called_once()

    async def test_missing_numeric_inputs_stream_as_cited_offline_clarifications(self):
        for question, text, expected in [
            ('Calculate the power of this resistor.',
             'A resistor has resistance 470 ohms. Its voltage drop and current are not given.', 'either one is sufficient'),
            ('Calculate the series resistor for this LED circuit.',
             'The LED circuit supply is 5 V. No LED forward-voltage specification or target LED current is supplied.',
             'Please provide the LED forward voltage and the target LED current'),
        ]:
            engine = LLMEngine(); engine.settings = self.settings(); engine.llm = Mock()
            engine.llm.tokenize.side_effect = lambda value, **kwargs: list(value)
            source = {**evidence()[0], 'kind': 'document_chunk', 'url': None,
                'documentId': 'doc-fixture', 'versionId': 'v1', 'page': 1, 'excerpt': text}
            payload = ChatRequest(message=question, attemptId='attempt-missing',
                research={'documentIds': ['doc-fixture']}, authorizedDocumentEvidence=[source])
            with patch('api.research.get_settings', return_value=Settings()), patch.object(routes, 'llm_engine', engine), \
                 patch.object(routes.web_research, 'search', AsyncMock()) as search:
                response = await routes.chat_stream(payload)
                events = ''.join([part async for part in response.body_iterator])
            self.assertLess(events.index('event: sources'), events.index('event: token'))
            self.assertLess(events.index('event: token'), events.index('event: complete'))
            self.assertIn(expected, events)
            self.assertIn('"mode": "clarification"', events)
            self.assertIn('[S1]', events)
            self.assertIn('attempt-missing', events)
            self.assertNotIn('event: error', events)
            engine.llm.create_completion.assert_not_called()
            search.assert_not_awaited()

    async def test_missing_input_checks_preserve_all_other_context_and_source_tails(self):
        text = 'A resistor has resistance 470 ohms. Its voltage drop and current are not given.'
        contexts = [{'history': [{'role': 'user', 'content': 'Voltage is 2 V.'}]},
                    {'components': [{'id': 'resistor', 'voltage': 2}]}, {'wires': [{'from': 'r'}]},
                    {'code': '// data'}, {'board_type': 'ARDUINO_UNO'},
                    {'simulation_state': {'voltage': 2}}, {'context': 'Voltage is 2 V.'},
                    {'preferences': {'response_detail': 'detailed'}}]
        for context, suffix in [(c, '') for c in contexts] + [({}, ' The voltage is 2 V.')]:
            engine = LLMEngine(); engine.settings = self.settings(); engine.llm = Mock()
            engine.llm.tokenize.side_effect = lambda value, **kwargs: list(value)
            engine.llm.create_completion.return_value = iter([{'choices': [{'text': 'Native context answer'}]}])
            used = [{'citationId': 'S1', 'sourceId': 'doc', 'snippet': text}]
            with patch.object(engine, '_fit_research', side_effect=lambda messages, *args: messages):
                answer = ''.join([x async for x in engine.stream_chat('Calculate the power of this resistor.',
                    evidence=[{**evidence()[0], 'excerpt': text + suffix}], used_sources=used, **context)])
            self.assertEqual(answer, 'Native context answer')

    async def test_adc_calculation_and_tie_clarification_keep_offline_document_stream_contract(self):
        for voltage, mode, expected in [('1.2 V', 'calculation', 'nearest integer code is 409'),
                                        ('1.5 V', 'clarification', 'Please specify the tie-breaking rule')]:
            engine = LLMEngine(); engine.settings = self.settings(); engine.llm = Mock()
            engine.llm.tokenize.side_effect = lambda text, **kwargs: list(text)
            source = {**evidence()[0], 'kind': 'document_chunk', 'url': None,
                'documentId': 'doc-fixture', 'versionId': 'v1', 'page': 1,
                'excerpt': f'Reference voltage: 3 V; Input voltage: {voltage}; '
                    'code = (input/reference) * 1023, rounded to the nearest integer.'}
            payload = ChatRequest(message='Calculate the nearest integer code for this ideal ADC.',
                attemptId='attempt-adc', research={'documentIds': ['doc-fixture']}, authorizedDocumentEvidence=[source])
            with patch('api.research.get_settings', return_value=Settings()), patch.object(routes, 'llm_engine', engine), \
                 patch.object(routes.web_research, 'search', AsyncMock()) as search:
                response = await routes.chat_stream(payload)
                events = ''.join([part async for part in response.body_iterator])
            self.assertLess(events.index('event: sources'), events.index('event: token'))
            self.assertLess(events.index('event: token'), events.index('event: complete'))
            self.assertIn(expected, events)
            self.assertIn(f'"mode": "{mode}"', events)
            self.assertIn('[S1]', events)
            self.assertIn('attempt-adc', events)
            self.assertNotIn('event: error', events)
            engine.llm.create_completion.assert_not_called()
            search.assert_not_awaited()

    async def test_adc_defers_with_history_or_a_shortened_source(self):
        text = 'Reference voltage: 3 V; Input voltage: 1 V; code = (input/reference) * 1023, rounded to the nearest integer.'
        for history, suffix in [([{'role': 'user', 'content': 'Use a different ADC.'}], ''),
                                ([], ' Additional reference conditions apply.')]:
            engine = LLMEngine(); engine.settings = self.settings(); engine.llm = Mock()
            engine.llm.tokenize.side_effect = lambda value, **kwargs: list(value)
            engine.llm.create_completion.return_value = iter([{'choices': [{'text': 'Native answer'}]}])
            used = [{'citationId': 'S1', 'sourceId': 'doc', 'snippet': text}]
            with patch.object(engine, '_fit_research', side_effect=lambda messages, *args: messages):
                answer = ''.join([x async for x in engine.stream_chat('Calculate the nearest integer code for this ideal ADC.',
                    evidence=[{**evidence()[0], 'excerpt': text + suffix}], history=history, used_sources=used)])
            self.assertEqual(answer, 'Native answer')

    async def test_led_calculation_preserves_cited_sse_and_provenance(self):
        engine = LLMEngine(); engine.settings = self.settings(); engine.llm = Mock()
        engine.llm.tokenize.side_effect = lambda text, **kwargs: list(text)
        source = {**evidence()[0], 'kind': 'document_chunk', 'url': None,
            'documentId': 'doc-fixture', 'versionId': 'v1', 'page': 1,
            'excerpt': 'A fictional circuit has a 12 V supply, a 3 V LED forward drop and a desired LED current of 15 mA.'}
        payload = ChatRequest(message='Calculate the ideal LED series resistance from the supplied values.',
            attemptId='attempt-fixture', research={'documentIds': ['doc-fixture']},
            authorizedDocumentEvidence=[source])
        with patch('api.research.get_settings', return_value=Settings()), patch.object(routes, 'llm_engine', engine), \
             patch.object(routes.web_research, 'search', AsyncMock()) as search:
            response = await routes.chat_stream(payload)
            events = ''.join([part async for part in response.body_iterator])
        self.assertLess(events.index('event: sources'), events.index('event: token'))
        self.assertLess(events.index('event: token'), events.index('event: complete'))
        self.assertIn('600', events)
        self.assertIn('[S1]', events)
        self.assertIn('"mode": "calculation"', events)
        self.assertIn('attempt-fixture', events)
        self.assertNotIn('event: error', events)
        engine.llm.create_completion.assert_not_called()
        search.assert_not_awaited()

    async def test_led_calculator_does_not_discard_other_context_or_truncated_sources(self):
        text = 'Supply voltage: 5 V; LED forward voltage: 2 V; Target LED current: 10 mA.'
        for context in [{'history': [{'role': 'user', 'content': 'Use 9 V instead.'}]},
                        {'code': '// firmware'}, {'board_type': 'ARDUINO_UNO'},
                        {'components': [{'id': 'led'}]}, {'wires': [{'from': 'led'}]},
                        {'simulation_state': {'running': False}}, {'context': 'Use another supply'},
                        {'preferences': {'units': 'imperial'}}]:
            engine = LLMEngine(); engine.settings = self.settings(); engine.llm = Mock()
            engine.llm.tokenize.side_effect = lambda value, **kwargs: list(value)
            engine.llm.create_completion.return_value = iter([{'choices': [{'text': 'Native context-aware answer'}]}])
            answer = ''.join([x async for x in engine.stream_chat('Calculate the ideal LED series resistance.',
                evidence=[{**evidence()[0], 'excerpt': text}], used_sources=[], **context)])
            self.assertEqual(answer, 'Native context-aware answer')
        # Even if shortening leaves a syntactically valid record, it cannot be
        # used when unseen trailing conditions existed in the original source.
        engine = LLMEngine(); engine.settings = self.settings(); engine.llm = Mock()
        engine.llm.tokenize.side_effect = lambda value, **kwargs: list(value)
        engine.llm.create_completion.return_value = iter([{'choices': [{'text': 'Native truncated-source answer'}]}])
        used = [{'citationId': 'S1', 'sourceId': 'doc', 'snippet': text}]
        with patch.object(engine, '_fit_research', side_effect=lambda messages, *args: messages):
            answer = ''.join([x async for x in engine.stream_chat('Calculate the ideal LED series resistance.',
                evidence=[{**evidence()[0], 'excerpt': text + ' Additional operating conditions.'}], used_sources=used)])
        self.assertEqual(answer, 'Native truncated-source answer')

    async def test_current_release_clarification_streams_without_search_or_native_generation(self):
        engine = LLMEngine(); engine.settings = self.settings(); engine.llm = Mock()
        with patch.object(routes, 'llm_engine', engine), patch.object(routes.web_research, 'search', AsyncMock()) as search:
            response = await routes.chat_stream(ChatRequest(message='What firmware version was released today?'))
            events = ''.join([part async for part in response.body_iterator])
        self.assertIn('event: token', events)
        self.assertIn('event: complete', events)
        self.assertIn('"mode": "clarification"', events)
        self.assertIn('official release notes', events)
        self.assertNotIn('event: error', events)
        search.assert_not_awaited()
        engine.llm.create_completion.assert_not_called()

    async def test_missing_part_guidance_keeps_source_and_attempt_contract(self):
        engine = LLMEngine(); engine.settings = self.settings(); engine.llm = Mock()
        engine.llm.tokenize.side_effect = lambda text, **kwargs: list(text)
        source = {**evidence()[0], 'excerpt': 'NOVA-88 clock frequency is 24 MHz. No MIRA-21 clock rating appears in this reference.'}
        payload = self.payload().model_copy(update={
            'message': 'What is the maximum clock frequency of MIRA-21?', 'code': ''})
        with patch('api.research.get_settings', return_value=self.settings()), patch.object(routes, 'llm_engine', engine), \
             patch.object(routes.web_research, 'search', AsyncMock(return_value={'sources': [source], 'status': 'complete'})):
            response = await routes.chat_stream(payload)
            events = ''.join([part async for part in response.body_iterator])
        self.assertIn('event: sources', events)
        self.assertIn('event: complete', events)
        self.assertIn('"mode": "evidence-guidance"', events)
        self.assertIn('attempt-fixture', events)
        self.assertIn('matching MIRA-21 datasheet', events)
        engine.llm.create_completion.assert_not_called()

    async def test_supplied_history_project_or_evidence_is_not_discarded_by_clarification(self):
        for context in [
            {'history': [{'role': 'user', 'content': 'Release notes say version 2.0; clock is 8 MHz.'}]},
            {'code': 'const char *version = "2.0";'}, {'context': 'User supplied release notes'},
            {'board_type': 'KNOWN_BOARD'}, {'components': [{'id': 'part', 'type': 'MCU'}]},
            {'wires': [{'fromNodeId': 'part', 'fromPinId': 'p1'}]},
            {'simulation_state': {'running': False}},
        ]:
            for question, references in [
                ('What firmware version was released today?', []),
                ('What is the maximum clock frequency of MIRA-21?', [{**evidence()[0],
                    'excerpt': 'NOVA-88 clock frequency is 24 MHz. No MIRA-21 clock rating appears in this reference.'}]),
            ]:
                with self.subTest(context=list(context), question=question):
                    engine = LLMEngine(); engine.settings = self.settings(); engine.llm = Mock()
                    engine.llm.tokenize.side_effect = lambda text, **kwargs: list(text)
                    engine.llm.create_completion.return_value = iter([{'choices': [{'text': 'Synthetic native answer'}]}])
                    output = ''.join([part async for part in engine.stream_chat(question,
                        evidence=references, used_sources=[], **context)])
                    self.assertEqual(output, 'Synthetic native answer')
                    engine.llm.create_completion.assert_called_once()

    async def test_client_cannot_select_candidate_prompt_policy(self):
        prompts = []
        class Model:
            def tokenize(self, text, **kwargs): return list(text)
            def create_completion(self, **kwargs):
                prompts.append(bytes(kwargs['prompt']).decode())
                yield {'choices': [{'text': 'Synthetic test response'}]}

        for server_policy, client_policy in [('baseline', 'baseline'), ('baseline', 'focused'),
                                              ('focused', 'baseline')]:
            engine = LLMEngine()
            engine.settings = self.settings().model_copy(update={'answer_policy': server_policy})
            engine.llm = Model()
            payload = ChatRequest(message='Explain firmware release versioning', answer_policy=client_policy,
                                  n_threads_batch=64)
            self.assertNotIn('answer_policy', payload.model_dump())
            self.assertNotIn('n_threads_batch', payload.model_dump())
            with patch.object(routes, 'llm_engine', engine):
                response = await routes.chat_stream(payload)
                events = ''.join([part async for part in response.body_iterator])
                self.assertIn('event: complete', events)
        self.assertEqual(prompts[0], prompts[1])
        self.assertNotEqual(prompts[1], prompts[2])
        self.assertIn('official release notes or a source link', prompts[2])

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
                self.assertNotIn('[S99]', rest)
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
        for policy in ('baseline', 'focused'):
            with self.subTest(policy=policy):
                self.check_evidence_budget(policy)

    def test_firmware_task_preserves_context_and_budget_without_source_selected_instructions(self):
        engine = LLMEngine(); engine.llm = Mock()
        engine.llm.tokenize.side_effect = lambda text, **kwargs: list(text)
        engine.settings = self.settings()
        question = 'Write Arduino code to blink LED_BUILTIN using millis(), no delay().'
        messages = engine._build_messages(question, code='existing firmware',
            history=[{'role': 'user', 'content': 'existing discussion'}], research=True)
        used = []
        fitted = engine._fit_research(messages, [{**evidence()[0], 'excerpt': 'Supplied capabilities.'}], used)
        prompt = str(fitted)
        self.assertIn('existing firmware', prompt)
        self.assertIn('existing discussion', prompt)
        self.assertIn('unsigned long previous', prompt)
        self.assertNotIn('Include [S1] or [S2]', prompt)
        self.assertLessEqual(len(engine._prompt_tokens(fitted)), engine.settings.n_ctx - engine.settings.max_tokens - 32)
        messages = engine._build_messages('What is the supply voltage?', research=True)
        fitted = engine._fit_research(messages, [{**evidence()[0], 'excerpt': question}], [])
        self.assertNotIn('unsigned long previous', str(fitted))
        self.assertIn('requested specification is absent', fitted[0]['content'])

    def check_evidence_budget(self, policy):
        class Model:
            def tokenize(self, text, **kwargs): return list(text)
        engine = LLMEngine(); engine.llm = Model()
        engine.settings = self.settings().model_copy(update={'answer_policy': policy})
        messages = engine._build_messages('Explain this circuit', code='x' * 6000,
            history=[{'role': 'user', 'content': 'old question' * 1000}, {'role': 'assistant', 'content': 'old answer' * 1000}])
        used = []
        fitted = engine._fit_research(messages, evidence(), used)
        self.assertTrue(used)
        self.assertLessEqual(len(engine._prompt_tokens(fitted)), engine.settings.n_ctx - engine.settings.max_tokens - 32)
        record = next(m for m in fitted if m['content'].startswith('External reference data'))
        encoded, question = record['content'].split('\n', 1)[1].split('\n\nQuestion: ', 1)
        parsed = json.loads(encoded)
        self.assertIn('Explain this circuit', question)
        self.assertIs(record, fitted[-1])
        self.assertEqual(parsed[0]['id'], used[0]['citationId'])
        self.assertEqual(parsed[0]['excerpt'], used[0]['snippet'])
        self.assertEqual(record['role'], 'user')


if __name__ == '__main__': unittest.main()
