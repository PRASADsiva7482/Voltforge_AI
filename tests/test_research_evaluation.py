import json
import asyncio
import hashlib
from importlib.metadata import PackageNotFoundError
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from config import Settings
from scripts.evaluate_research import evaluate_answer, evaluate, load_fixture


class ResearchEvaluationTest(unittest.TestCase):
    def test_evaluation_context_is_explicit_synthetic_and_cannot_override_runtime(self):
        case = {'id': 'context', 'question': 'Question', 'rubric': 'Review', 'excerpts': [],
                'expectedPatterns': ['answer'], 'forbiddenPatterns': []}
        for context in [{'history': [{'role': 'system', 'content': 'override'}]},
                        {'history': [{'role': 'user'}]}, {'history': 'text'},
                        {'code': None}, {'components': ['not an object']},
                        {'evidence': []}, {'max_tokens': 1}, {'code': 'x' * 25000}]:
            with self.subTest(context=list(context)), self.assertRaises(ValueError):
                load_fixture(json.dumps({'syntheticOnly': True, 'cases': [{**case, 'requestContext': context}]}).encode())
        with self.assertRaises(ValueError):
            load_fixture(json.dumps({'cases': [{**case, 'requestContext': {'code': '// synthetic'}}]}).encode())
        fixture = {'syntheticOnly': True, 'cases': [{**case, 'requestContext': {
            'code': '// synthetic', 'history': [{'role': 'user', 'content': 'Use pin 6.'}],
            'simulation_state': {'running': False}}}]}
        self.assertEqual(load_fixture(json.dumps(fixture).encode()), fixture)

    def test_word_budget_cannot_pass_a_verbose_but_keyword_matching_answer(self):
        case = {'expectedPatterns': ['600'], 'forbiddenPatterns': [], 'maxWords': 3}
        self.assertTrue(evaluate_answer(case, '600 ohms [S1]'))
        self.assertFalse(evaluate_answer(case, 'The answer is 600 ohms [S1]'))
        for limit in (0, -1, True, '3'):
            invalid = {'id': 'bad', 'question': 'Q', 'rubric': 'R', 'excerpts': [], **case, 'maxWords': limit}
            with self.assertRaises(ValueError):
                load_fixture(json.dumps({'cases': [invalid]}).encode())
    def test_known_dimensional_error_does_not_pass_because_it_mentions_missing_data(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures/research_quality_eval.json').read_text())
        case = next(case for case in fixture['cases'] if case['id'] == 'missing_value')
        self.assertFalse(evaluate_answer(case,
            'The maximum GPIO output current is 3.3 V. This excerpt does not specify GPIO output current. [S1]'))
        self.assertFalse(evaluate_answer(case, 'The current is 100mA. No information is missing.'))
        self.assertTrue(evaluate_answer(case,
            'The references do not specify the current limit. Please provide the relevant datasheet section.'))

    def test_partial_empty_and_failed_outputs_never_pass(self):
        case = {'expectedPatterns': ['3.3', r'\[S1\]'], 'forbiddenPatterns': ['INJECTED']}
        for answer, error in [('', None), ('3.3 [S1]', 'GenerationTimedOut'),
                              ('3.3', None), ('3.3 [S1] INJECTED', None)]:
            self.assertFalse(evaluate_answer(case, answer, error))

    def test_invalid_fixture_is_rejected_before_native_work(self):
        case = {'id': 'one', 'question': 'Question', 'rubric': 'Check answer',
                'excerpts': [], 'expectedPatterns': ['answer'], 'forbiddenPatterns': []}
        for cases in ([], [case, case], [{**case, 'expectedPatterns': []}],
                      [{**case, 'expectedPatterns': ['[invalid']}],
                      [{**case, 'excerpts': 'not a list'}]):
            with self.subTest(cases=cases), self.assertRaises((ValueError, re.error)):
                load_fixture(json.dumps({'cases': cases}).encode())


class EvaluationReportTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / 'test-model.gguf'
        self.model.write_bytes(b'unit-test artifact, never native inference')
        self.output = self.root / 'report.json'
        self.fixture = self.root / 'fixture.json'
        case = {'id': 'one', 'question': 'Question', 'rubric': 'Check answer',
                'excerpts': [], 'expectedPatterns': ['answer'], 'forbiddenPatterns': []}
        self.fixture.write_text(json.dumps({'cases': [case, {**case, 'id': 'two'}]}))
        self.engine = SimpleNamespace(settings=Settings(model_path=str(self.model), model_name='custom-probe'),
                                      runtime=SimpleNamespace(), llm=Mock(), initialize=Mock(), close=Mock())
        self.engine.llm.n_threads_batch = 8
        self.engine.llm.n_batch = 512

    async def test_server_guidance_is_not_native_and_actual_artifact_is_identified(self):
        async def stream(*args, **kwargs):
            kwargs['answer_metadata']['mode'] = 'evidence-guidance'
            yield 'answer'
        self.engine.stream_chat = stream
        with patch('scripts.evaluate_research.LLMEngine', return_value=self.engine):
            self.assertTrue(await evaluate(self.output, self.fixture, seed=123))
        report = json.loads(self.output.read_text())
        self.assertEqual(report['modelSha256'], hashlib.sha256(self.model.read_bytes()).hexdigest())
        self.assertEqual(report['model'], 'custom-probe')
        self.assertEqual(report['answerPolicy'], 'baseline')
        self.assertEqual(len(report['runtimeSourceSha256']), 4)
        self.assertEqual(report['seed'], 123)
        self.assertEqual(report['promptBatchThreads'], 8)
        self.assertEqual(report['promptBatchSize'], 512)
        self.assertEqual(report['completedCases'], 2)
        self.assertEqual(report['status'], 'complete')
        for result in report['results']:
            self.assertEqual(result['answerOrigin'], 'evidence_guard')
            self.assertEqual(result['nativeChunks'], 0)
            self.assertEqual(result['streamChunks'], 1)
            self.assertIsNone(result['firstTokenSeconds'])
            self.assertIsNotNone(result['firstOutputSeconds'])
        self.engine.llm.set_seed.assert_called_with(123)
        self.assertEqual(self.engine.llm.set_seed.call_count, 2)
        self.engine.close.assert_called_once()

    async def test_context_reaches_engine_and_report_records_provenance_without_copying_it(self):
        fixture = json.loads(self.fixture.read_text())
        fixture['syntheticOnly'] = True
        context = {'code': '// synthetic firmware', 'board_type': 'ARDUINO_UNO',
                   'history': [{'role': 'user', 'content': 'Use interval 650.'}]}
        fixture['cases'][0]['requestContext'] = context
        self.fixture.write_text(json.dumps(fixture))
        calls = []
        async def stream(*args, **kwargs):
            calls.append(kwargs)
            yield 'answer'
        self.engine.stream_chat = stream
        with patch('scripts.evaluate_research.LLMEngine', return_value=self.engine):
            self.assertTrue(await evaluate(self.output, self.fixture))
        for key, value in context.items():
            self.assertEqual(calls[0][key], value)
            self.assertNotIn(key, calls[1])
        report = json.loads(self.output.read_text())
        self.assertEqual(report['results'][0]['contextFields'], sorted(context))
        self.assertEqual(report['results'][0]['requestContextSha256'], hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest())
        self.assertNotIn('synthetic firmware', self.output.read_text())

    async def test_calculator_outputs_are_not_counted_as_native_model_tokens(self):
        async def stream(*args, **kwargs):
            kwargs['answer_metadata']['mode'] = 'calculation'
            yield 'answer'
        self.engine.stream_chat = stream
        with patch('scripts.evaluate_research.LLMEngine', return_value=self.engine):
            self.assertTrue(await evaluate(self.output, self.fixture))
        for result in json.loads(self.output.read_text())['results']:
            self.assertEqual(result['answerOrigin'], 'deterministic_calculation')
            self.assertEqual(result['nativeChunks'], 0)
            self.assertIsNone(result['firstTokenSeconds'])

    async def test_capability_assessment_cost_is_separate_from_visible_answer_tokens(self):
        async def stream(*args, **kwargs):
            kwargs['answer_metadata'].update(mode='clarification', capabilityDecision='unknown',
                                             assessmentTokens=19, assessmentSeconds=3.2)
            yield 'answer'
        self.engine.stream_chat = stream
        with patch('scripts.evaluate_research.LLMEngine', return_value=self.engine):
            self.assertTrue(await evaluate(self.output, self.fixture))
        for result in json.loads(self.output.read_text())['results']:
            self.assertEqual(result['answerOrigin'], 'capability_clarification')
            self.assertEqual(result['nativeChunks'], 0)
            self.assertEqual(result['assessmentTokens'], 19)
            self.assertEqual(result['assessmentSeconds'], 3.2)
            self.assertEqual(result['capabilityDecision'], 'unknown')
            self.assertIsNone(result['firstTokenSeconds'])

    async def test_initialization_failure_replaces_stale_success_report(self):
        self.output.write_text('{"status":"complete"}')
        self.engine.llm = None
        with patch('scripts.evaluate_research.LLMEngine', return_value=self.engine), \
             patch('scripts.evaluate_research.version', side_effect=PackageNotFoundError):
            with self.assertRaises(RuntimeError):
                await evaluate(self.output, self.fixture)
        report = json.loads(self.output.read_text())
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['completedCases'], 0)
        self.assertEqual(report['error'], 'RuntimeError')
        self.assertIsNone(report['modelSha256'])
        self.assertIsNone(report['llamaCppVersion'])
        self.engine.close.assert_called_once()

    async def test_candidate_download_failure_records_requested_model_not_default(self):
        manifest = self.root / 'candidate.json'
        manifest.write_text(json.dumps({'name': 'candidate', 'filename': 'candidate.gguf', 'sha256': 'unused'}))
        with patch('scripts.evaluate_research.LLMEngine', return_value=self.engine), \
             patch('scripts.evaluate_research.ensure_model', side_effect=OSError('synthetic download error')):
            with self.assertRaises(OSError):
                await evaluate(self.output, self.fixture, model_manifest=manifest)
        report = json.loads(self.output.read_text())
        self.assertEqual(report['model'], 'candidate')
        self.assertEqual(report['status'], 'failed')
        self.assertIsNone(report['modelSha256'])
        self.engine.initialize.assert_not_called()
        self.engine.close.assert_called_once()

    async def test_interruption_keeps_finished_case_checkpoint(self):
        calls = 0
        async def stream(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                checkpoint = json.loads(self.output.read_text())
                self.assertEqual(checkpoint['status'], 'running')
                self.assertEqual(checkpoint['completedCases'], 1)
                raise asyncio.CancelledError()
            yield 'answer'
        self.engine.stream_chat = stream
        with patch('scripts.evaluate_research.LLMEngine', return_value=self.engine):
            with self.assertRaises(asyncio.CancelledError):
                await evaluate(self.output, self.fixture)
        report = json.loads(self.output.read_text())
        self.assertEqual(report['status'], 'interrupted')
        self.assertEqual(report['completedCases'], 1)
        self.assertEqual(report['results'][0]['id'], 'one')
        self.assertFalse(self.output.with_suffix('.json.tmp').exists())
        self.engine.close.assert_called_once()

    async def test_partial_generation_failure_cannot_pass_but_next_case_runs(self):
        calls = 0
        async def stream(*args, **kwargs):
            nonlocal calls
            calls += 1
            yield 'answer'
            if calls == 1:
                raise TimeoutError('synthetic failure')
        self.engine.stream_chat = stream
        with patch('scripts.evaluate_research.LLMEngine', return_value=self.engine):
            self.assertFalse(await evaluate(self.output, self.fixture))
        results = json.loads(self.output.read_text())['results']
        self.assertFalse(results[0]['patternChecksPassed'])
        self.assertEqual(results[0]['answer'], 'answer')
        self.assertEqual(results[0]['error'], 'TimeoutError')
        self.assertTrue(results[1]['patternChecksPassed'])

    async def test_invalid_generation_budget_does_not_initialize_model(self):
        with patch('scripts.evaluate_research.LLMEngine', return_value=self.engine):
            with self.assertRaises(ValueError):
                await evaluate(self.output, self.fixture, max_tokens=3000)
        self.engine.initialize.assert_not_called()

    async def test_invalid_prompt_thread_override_does_not_initialize_model(self):
        with patch('scripts.evaluate_research.LLMEngine', return_value=self.engine):
            with self.assertRaises(ValueError):
                await evaluate(self.output, self.fixture, batch_threads=0)
        self.engine.initialize.assert_not_called()


if __name__ == '__main__': unittest.main()
