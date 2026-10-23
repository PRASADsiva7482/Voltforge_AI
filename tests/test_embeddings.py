import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from embedding_model import prepare
from engine.embeddings import EmbeddingEngine
from engine.runtime import RuntimeBusy
from fastapi.testclient import TestClient
from main import app
from api.security import require_service_token
from embedding_model import MODEL_ID


class EmbeddingArtifactTest(unittest.TestCase):
    def test_offline_repeat_corruption_and_model_revision_rebuild(self):
        payload = b'synthetic pinned artifact'
        manifest = {'repository':'fixture/model','revision':'fixed', 'pooling':'mean', 'files':[
            {'filename':'model.onnx','size':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}]}
        with tempfile.TemporaryDirectory() as root:
            def download(target, **kwargs):
                Path(target).parent.mkdir(parents=True,exist_ok=True); Path(target).write_bytes(payload)
            with patch('embedding_model.ensure_model',side_effect=download) as fetch:
                folder = prepare(root, manifest=manifest)
                self.assertTrue(fetch.call_args.kwargs['allow_download'])
            self.assertEqual(prepare(root,allow_download=False,manifest=manifest), folder)
            (folder/'model.onnx').write_bytes(b'x'*len(payload))
            with self.assertRaises(RuntimeError): prepare(root,allow_download=False,manifest=manifest)
            with patch('embedding_model.ensure_model',side_effect=download):
                changed = prepare(root,manifest={**manifest,'revision':'next'})
                self.assertNotEqual(changed,folder)
                self.assertEqual((changed/'model.onnx').read_bytes(),payload)

    def test_artifact_path_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as root, self.assertRaises(ValueError):
            prepare(root,manifest={'files':[{'filename':'../escape'}]})

    def test_unavailable_and_busy_are_explicit(self):
        engine = EmbeddingEngine()
        with self.assertRaises(RuntimeError): engine.encode(['hello'])
        engine.gate.acquire()
        try:
            with self.assertRaises(RuntimeBusy): engine.encode(['hello'])
        finally: engine.gate.release()

    def test_private_encoder_auth_limits_model_binding_and_availability(self):
        client=TestClient(app)
        self.assertEqual(client.get('/voltForge-ai/api/v1/model/embeddings').status_code,401)
        app.dependency_overrides[require_service_token]=lambda:None
        try:
            path='/voltForge-ai/api/v1/model/embeddings'
            self.assertEqual(client.post(path,json={'modelId':MODEL_ID,'texts':['a']*33}).status_code,422)
            self.assertEqual(client.post(path,json={'modelId':MODEL_ID,'texts':['x'*2001]}).status_code,422)
            self.assertEqual(client.post(path,json={'modelId':MODEL_ID,'texts':['ok']}).status_code,503)
            with patch('api.embeddings.embedding_engine.session',object()), patch('api.embeddings.embedding_engine.encode',return_value=[[1.0]+[0.0]*383]):
                self.assertEqual(client.post(path,json={'modelId':'a'*64,'texts':['ok']}).status_code,409)
                response=client.post(path,json={'modelId':MODEL_ID,'texts':['ok']})
                self.assertEqual(response.status_code,200)
                self.assertEqual(response.json()['modelId'],MODEL_ID)
        finally: app.dependency_overrides.pop(require_service_token,None)


if __name__=='__main__': unittest.main()
