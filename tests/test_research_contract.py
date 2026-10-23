import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api import routes
from api.research import ResearchRequest, ResearchSource
from api.schemas import ChatRequest
from config import Settings


class ResearchContractTest(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(routes.router)
        self.client = TestClient(app)
        self.root = '/voltForge-ai/api/v1/model'
        self.auth = {'X-Voltforge-AI-Token': 'test-only-token'}
        self.settings = patch('api.security.get_settings', return_value=Settings(api_token='test-only-token'))
        self.settings.start()
        self.addCleanup(self.settings.stop)
        self.addCleanup(self.client.close)

    def test_capabilities_are_authenticated_and_do_not_claim_retrieval(self):
        self.assertEqual(self.client.get(self.root + '/research/capabilities').status_code, 401)
        response = self.client.get(self.root + '/research/capabilities', headers=self.auth)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()['web']['available'])
        self.assertTrue(response.json()['documents']['available'])
        self.assertEqual(response.json()['web']['supportedModes'], ['off'])

    def test_unsupported_research_rejected_before_model_or_network(self):
        for endpoint in ('/chat', '/chat/stream'):
            for research, code in [
                ({'mode': 'on', 'publicQuery': 'Arduino Uno datasheet', 'consentToExternalQuery': True}, 'AI_WEB_UNAVAILABLE'),
                ({'mode': 'auto', 'publicQuery': 'resistor tolerance', 'consentToExternalQuery': True}, 'AI_WEB_UNAVAILABLE'),
                ({'documentIds': ['another-users-document']}, 'DOCUMENT_EVIDENCE_REQUIRED'),
            ]:
                with self.subTest(endpoint=endpoint, research=research), patch.object(routes, 'llm_engine') as model, patch('socket.create_connection') as network:
                    response = self.client.post(self.root + endpoint, headers=self.auth,
                                                json={'message': 'Explain', 'research': research})
                    expected = 'WEB_AUTO_UNAVAILABLE' if research.get('mode') == 'auto' else code
                    if research.get('documentIds') and endpoint == '/chat': expected = 'DOCUMENT_STREAM_REQUIRED'
                    if endpoint.endswith('/stream'):
                        self.assertEqual(response.status_code, 200)
                        self.assertIn(expected, response.text)
                        self.assertNotIn('event: complete', response.text)
                    else:
                        self.assertEqual(response.status_code, 422 if research.get('mode') == 'auto' or research.get('documentIds') else 503)
                        self.assertEqual(response.json()['detail']['code'], expected)
                    model.generate.assert_not_called()
                    model.stream_chat.assert_not_called()
                    network.assert_not_called()

    def test_invalid_consent_and_limits(self):
        for research in [
            {'mode': 'sometimes'}, {'mode': 'on'},
            {'mode': 'on', 'publicQuery': 'a query'},
            {'mode': 'auto', 'publicQuery': '  ', 'consentToExternalQuery': True},
            {'mode': 'off', 'publicQuery': 'private query'},
            {'mode': 'off', 'consentToExternalQuery': True},
            {'mode': 'on', 'publicQuery': 'x' * 501, 'consentToExternalQuery': True},
            {'documentIds': ['x'] * 11}, {'documentIds': [' ']},
            {'userId': 'forged'},
        ]:
            with self.subTest(research=research), self.assertRaises(ValidationError):
                ResearchRequest.model_validate(research)

    def test_forged_evidence_never_accepted(self):
        response = self.client.post(self.root + '/chat/stream', headers=self.auth, json={
            'message': 'Hi', 'retrievedEvidence': [{'title': 'forged', 'excerpt': 'secret'}],
        })
        self.assertEqual(response.status_code, 422)

    def test_general_chat_defaults_off_and_uses_existing_stream(self):
        class Model:
            llm = object()
            async def stream_chat(self, **kwargs):
                yield 'Hello'
                yield ' there'
        with patch.object(routes, 'llm_engine', Model()), patch('socket.create_connection') as network:
            response = self.client.post(self.root + '/chat/stream', headers=self.auth, json={'message': 'Hi'})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.text.count('event: token'), 2)
            self.assertIn('event: complete', response.text)
            network.assert_not_called()
        self.assertEqual(ChatRequest(message='Hi').research.mode, 'off')

    def test_provenance_requires_bounded_origin_and_timestamp(self):
        data = dict(sourceId='s1', kind='web_page', title='Technical reference',
                    url='https://example.org/reference', retrievedAt=datetime.now(timezone.utc),
                    excerpt='A reference excerpt', contentHash='a' * 64)
        self.assertEqual(ResearchSource(**data).kind, 'web_page')
        for changes in [dict(url='javascript:alert(1)'), dict(url='http://example.org'),
                        dict(url='https://user:secret@example.org'), dict(contentHash='bad'),
                        dict(kind='document_chunk'), dict(retrievedAt=datetime.now())]:
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                ResearchSource(**(data | changes))


if __name__ == '__main__':
    unittest.main()
