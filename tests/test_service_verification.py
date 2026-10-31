import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

from scripts.verify_service import (run_report, REQUIRED_CHECKS, CAPACITY_CHECKS,
                                    process_peak_working_set, capacity_workloads)


class ServiceVerificationReportTest(unittest.IsolatedAsyncioTestCase):
    def test_memory_includes_native_child_not_only_launcher(self):
        with patch('scripts.verify_service.subprocess.run', return_value=Mock(stdout='[100, 200]')) as command, \
             patch('scripts.verify_service._single_process_peak', side_effect=[5, 1400]) as memory:
            self.assertEqual(process_peak_working_set(100), 1405)
        self.assertEqual([call.args[0] for call in memory.call_args_list], [100, 200])
        self.assertIn('$ids.Add(100)', command.call_args.args[0][-1])

    async def test_capacity_workloads_use_valid_document_attempt_and_reject_clarification(self):
        from api.schemas import ChatRequest
        class Response:
            status_code = 200
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def aiter_lines(self):
                for line in ['event: token', 'data: {"delta":"Please clarify"}', 'event: complete']:
                    yield line
        client = Mock()
        def stream(*args, **kwargs):
            ChatRequest.model_validate(kwargs['json'])
            return Response()
        client.stream.side_effect = stream
        report = {'measurements': {}, 'checks': {}, 'profile': {'maxResponseSeconds': 125}}
        with patch('scripts.verify_service.process_peak_working_set', return_value=1405), self.assertRaises(AssertionError):
            await capacity_workloads(client, '', Mock(pid=100), report)
        self.assertNotIn('sequentialWorkloads', report['checks'])
        self.assertEqual(len(report['measurements']['workloads']), 1)

    async def test_smoke_checks_cannot_pass_capacity_profile(self):
        async def probe(report):
            report['checks'].update({name: True for name in REQUIRED_CHECKS})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'report.json'
            with patch('scripts.verify_service.verify', side_effect=probe), self.assertRaises(RuntimeError):
                await run_report(output, capacity=True)
            report = json.loads(output.read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(report['scope'], 'local_single_request_capacity')
            self.assertFalse(report['profile']['sustainedSoakVerified'])

    async def test_capacity_failure_cannot_be_promoted_to_pass(self):
        for missing in CAPACITY_CHECKS:
            async def probe(report):
                report['checks'].update({name: True for name in REQUIRED_CHECKS + CAPACITY_CHECKS})
                report['checks'][missing] = False
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / 'report.json'
                with patch('scripts.verify_service.verify', side_effect=probe), self.assertRaises(RuntimeError):
                    await run_report(output, capacity=True)
                self.assertEqual(json.loads(output.read_text())['status'], 'failed')

    async def test_success_cannot_claim_web_database_browser_or_release_acceptance(self):
        async def probe(report):
            report['checks'].update({name: True for name in REQUIRED_CHECKS})
            report['measurements']['firstTokenSeconds'] = 1.25
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'report.json'
            with patch('scripts.verify_service.verify', side_effect=probe):
                await run_report(output)
            report = json.loads(output.read_text())
            self.assertEqual(report['status'], 'passed')
            for field in ('liveProviderVerified', 'databasePersistenceVerified', 'browserVerified', 'releaseAccepted'):
                self.assertFalse(report[field])
            self.assertEqual(report['measurements']['firstTokenSeconds'], 1.25)
            self.assertFalse(output.with_suffix('.json.tmp').exists())

    async def test_incomplete_verification_cannot_pass(self):
        async def probe(report):
            report['checks']['nativeModelReady'] = True
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'report.json'
            with patch('scripts.verify_service.verify', side_effect=probe), self.assertRaises(RuntimeError):
                await run_report(output)
            self.assertEqual(json.loads(output.read_text())['status'], 'failed')

    async def test_failed_or_interrupted_probe_replaces_stale_success_without_private_errors(self):
        for error in (RuntimeError('synthetic-private-token'), asyncio.CancelledError()):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / 'report.json'
                output.write_text('{"status":"passed"}')
                async def probe(report):
                    report['checks']['authentication'] = True
                    raise error
                with patch('scripts.verify_service.verify', side_effect=probe), self.assertRaises(type(error)):
                    await run_report(output)
                text = output.read_text()
                report = json.loads(text)
                self.assertIn(report['status'], ('failed', 'interrupted'))
                self.assertTrue(report['checks']['authentication'])
                self.assertNotIn('synthetic-private-token', text)
                self.assertFalse(report['releaseAccepted'])
