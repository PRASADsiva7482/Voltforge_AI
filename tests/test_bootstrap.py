import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import bootstrap
from config import Settings
from pydantic import ValidationError


class BootstrapTests(unittest.TestCase):
    def test_answer_policy_requires_explicit_valid_operator_setting(self):
        from config import load_settings
        self.assertEqual(Settings().answer_policy, 'baseline')
        with self.assertRaises(ValidationError):
            Settings(answer_policy='client-selected')
        with patch.dict(os.environ, {'VOLTFORGE_AI_ANSWER_POLICY': 'focused'}):
            self.assertEqual(load_settings().answer_policy, 'focused')

    def test_prompt_thread_setting_is_bounded_and_loads_through_existing_environment_path(self):
        from config import load_settings
        for value in (0, -1, 65):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                Settings(n_threads_batch=value)
        self.assertIsNone(Settings().n_threads_batch)
        with patch.dict(os.environ, {'VOLTFORGE_AI_N_THREADS_BATCH': '4'}):
            self.assertEqual(load_settings().n_threads_batch, 4)

    def test_global_python_launch_bootstraps_managed_environment(self):
        with patch('bootstrap.sys.version_info', (3,14)), patch('bootstrap.subprocess.call', return_value=0) as launch:
            with self.assertRaises(SystemExit) as result: bootstrap.ensure_launch_runtime()
        self.assertEqual(result.exception.code, 0)
        self.assertEqual(launch.call_args.args[0][-1], '--run')

    def test_ready_runtime_does_not_recursively_bootstrap(self):
        with patch('bootstrap.sys.version_info', (3,12)), patch('bootstrap.subprocess.call') as launch:
            bootstrap.ensure_launch_runtime()
        launch.assert_not_called()

    def test_failed_managed_runtime_cannot_enter_restart_loop(self):
        with patch('bootstrap.sys.version_info',(3,14)), patch.dict(os.environ,{'VOLTFORGE_AI_MANAGED_LAUNCH':'1'}), patch('bootstrap.subprocess.call') as launch:
            with self.assertRaisesRegex(RuntimeError,'restart loop'): bootstrap.ensure_launch_runtime()
        launch.assert_not_called()

    def test_token_migration_preserves_gateway_secret_and_removes_tracked_copy(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(bootstrap, 'ROOT', Path(directory)), patch.dict(os.environ, {}, clear=True):
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps({'api_token':'synthetic-token','n_threads':4}))
            bootstrap.prepare_token()
            self.assertEqual((Path(directory)/'.service-token').read_text(), 'synthetic-token')
            self.assertEqual(json.loads(path.read_text()), {'n_threads':4})
            bootstrap.prepare_token()
            self.assertEqual((Path(directory)/'.service-token').read_text(), 'synthetic-token')

    def test_conflicting_tokens_stop_migration_without_destroying_configuration(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(bootstrap, 'ROOT', Path(directory)), patch.dict(os.environ, {}, clear=True):
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps({'api_token':'synthetic-original'}))
            (Path(directory)/'.service-token').write_text('synthetic-conflict')
            with self.assertRaises(RuntimeError): bootstrap.prepare_token()
            self.assertEqual(json.loads(path.read_text())['api_token'], 'synthetic-original')

    def test_fresh_setup_creates_private_token(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(bootstrap, 'ROOT', Path(directory)), patch.dict(os.environ, {}, clear=True):
            bootstrap.prepare_token()
            self.assertGreaterEqual(len((Path(directory)/'.service-token').read_text()), 48)

    def test_invalid_settings_errors_never_include_service_token(self):
        with self.assertRaises(ValidationError) as caught:
            Settings(n_ctx=2048,max_tokens=2048,api_token='synthetic-private-secret')
        self.assertNotIn('synthetic-private-secret',str(caught.exception))

    def test_python_selector_requires_tested_runtime_version(self):
        with patch('bootstrap.subprocess.check_output',return_value='True') as command:
            self.assertTrue(bootstrap.supported('python'))
        self.assertIn('== (3,12)',command.call_args.args[0][-1])


if __name__ == '__main__':
    unittest.main()
