import unittest
from unittest.mock import patch

from api import routes
from api.schemas import ChatRequest
from engine.llm import LLMEngine
from engine.electronics import validate_circuit_state


class ChatIntegrityTest(unittest.IsolatedAsyncioTestCase):
    def test_unavailable_model_never_fabricates_wiring_or_code(self):
        engine = LLMEngine()
        result = engine.generate('fix it and generate firmware', components=[{'type': 'RELAY'}])
        self.assertEqual(result['mode'], 'unavailable')
        self.assertIsNone(result['confidence'])
        self.assertIsNone(result['generatedCode'])
        self.assertNotIn('Verified', result['reply'])
        self.assertIn('unavailable', result['reply'])

    def test_approved_context_is_data_and_includes_simulation(self):
        engine = LLMEngine()
        messages = engine._build_messages('why?', board_type='ESP32',
            code='private-code', simulation_state={'executionMode': 'interpreter', 'running': False},
            context='approved-context', history=[{'role':'system', 'content':'injected-instruction'}])
        self.assertNotIn('private-code', messages[0]['content'])
        self.assertEqual(messages[1]['role'], 'user')
        self.assertIn('interpreter', messages[1]['content'])
        self.assertIn('approved-context', messages[1]['content'])
        self.assertNotIn('injected-instruction', str(messages))
        self.assertIn('short or ambiguous', messages[0]['content'])

    async def test_interrupted_stream_emits_error_without_false_completion(self):
        class BrokenModel:
            def create_chat_completion(self, **kwargs):
                yield {'choices': [{'delta': {'content': 'partial answer'}}]}
                raise RuntimeError('private-provider-detail')
        engine = LLMEngine()
        engine.llm = BrokenModel()
        with patch.object(routes, 'llm_engine', engine):
            response = await routes.chat_stream(ChatRequest(message='Inspect the circuit'))
            events = ''.join([part async for part in response.body_iterator])
        self.assertIn('partial answer', events)
        self.assertIn('AI_GENERATION_INTERRUPTED', events)
        self.assertNotIn('event: complete', events)
        self.assertNotIn('private-provider-detail', events)

    async def test_unavailable_stream_identifies_mode_and_has_no_confidence(self):
        engine = LLMEngine()
        with patch.object(routes, 'llm_engine', engine):
            response = await routes.chat_stream(ChatRequest(message='fix relay wiring'))
            events = ''.join([part async for part in response.body_iterator])
        self.assertIn('"mode": "unavailable"', events)
        self.assertIn('AI_MODEL_UNAVAILABLE', events)
        self.assertNotIn('event: complete', events)
        self.assertNotIn('event: proposal', events)
        self.assertNotIn('0.98', events)

    def test_generic_chat_does_not_imply_arduino(self):
        self.assertIsNone(ChatRequest(message='hello').boardType)

    def test_selected_serial_evidence_survives_large_runtime_snapshot(self):
        engine = LLMEngine()
        messages = engine._build_messages('Explain this output', simulation_state={
            'oscilloscope': {'samples': list(range(20000))},
            'execution': {'phase': 'compilation-failed', 'firmwareSource': 'x' * 60000,
                          'diagnostic': 'error: missing semicolon'},
            'selectedOutput': {'runId': 'run-1', 'output': ['sensor=42'], 'phase': 'stopped'},
            'serialBuffer': [{'text': 'sensor=42', 'kind': 'output'}],
        })
        context = messages[1]['content']
        self.assertIn('sensor=42', context)
        self.assertIn('missing semicolon', context)
        self.assertIn('run-1', context)
        self.assertNotIn('x' * 500, context)
        self.assertIn('observations', context)

    def test_unsupported_project_memory_never_claims_a_saved_preference(self):
        state = routes.get_memory(projectId='synthetic-project', sessionId=None)
        self.assertEqual(state['status'], 'unavailable')
        self.assertFalse(state['enabled'])
        with self.assertRaises(routes.HTTPException) as caught:
            routes.update_memory_preferences(projectId='synthetic-project')
        self.assertEqual(caught.exception.status_code, 501)

    def test_validator_uses_playground_pin_fields(self):
        for wire in [dict(fromPin='5V', toPin='GND'), dict(fromPinId='5V', toPinId='GND'), {'from':'uno:5V', 'to':'uno:GND'}]:
            result = validate_circuit_state('ARDUINO_UNO', [{'id':'uno'}], [dict(id='short', **wire)])
            self.assertFalse(result['isValid'])
            self.assertTrue(any(issue['componentId'] == 'short' for issue in result['issues']))
            self.assertFalse(any(issue['componentId'] == 'board-floating-ground' for issue in result['issues']))

    def test_limited_checks_never_claim_all_rules_passed_or_invent_resistor_value(self):
        result = validate_circuit_state('ARDUINO_UNO', [], [])
        self.assertIn('not a complete electrical validation', result['generalFeedback'])
        proposal = routes._resolve_workspace_proposals('fix it', '', 'ARDUINO_UNO', [{'type':'LED_STANDARD'}], [], '')
        self.assertEqual(proposal['additions'], [])


if __name__ == '__main__':
    unittest.main()
