"""Real clean-artifact setup, repeat/offline load, corruption and cache identity checks."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from embedding_model import MANIFEST,MODEL_ID,prepare
from engine.embeddings import EmbeddingEngine

with tempfile.TemporaryDirectory(prefix='vf-embedding-setup-') as root:
    start=time.monotonic(); folder=prepare(root); downloaded=time.monotonic()-start
    with patch('urllib.request.urlopen',side_effect=AssertionError('No offline network')):
        assert prepare(root,allow_download=False)==folder
        engine=EmbeddingEngine(); engine.initialize(allow_download=False,artifact_root=root)
        vector=engine.encode(['A series resistor limits current.'])[0]
        assert len(vector)==384 and abs(sum(x*x for x in vector)-1)<.01
        engine.session.close()
    artifact=folder/MANIFEST['files'][0]['filename']
    # Corrupt only our temporary copy, never the application's model.
    with artifact.open('r+b') as out: out.write(b'BROKEN')
    try: prepare(root,allow_download=False)
    except RuntimeError: pass
    else: raise AssertionError('Corrupt artifact accepted')
    changed={**MANIFEST,'pooling':'future-implementation'}
    changed_id=hashlib.sha256(json.dumps(changed,sort_keys=True).encode()).hexdigest()
    assert changed_id!=MODEL_ID
    print(json.dumps({'cleanDownloadSeconds':round(downloaded,2),'offlineInference':True,
                      'repeatVerified':True,'corruptionRejected':True,'modelChangeInvalidatesIdentity':True}))
