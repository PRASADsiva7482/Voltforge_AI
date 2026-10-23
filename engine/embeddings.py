"""Bounded in-process CPU embeddings. No provider, text cache or database access."""
import threading
import numpy as np
from config import get_settings
from embedding_model import MANIFEST, MODEL_ID, prepare
from engine.runtime import RuntimeBusy


class EmbeddingEngine:
    def __init__(self):
        self.session = None
        self.gate = threading.Lock()

    def initialize(self, *, allow_download=None, artifact_root=None):
        if self.session is not None:
            return
        from llama_cpp import Llama, LLAMA_POOLING_TYPE_MEAN
        settings = get_settings()
        folder = prepare(artifact_root,allow_download=settings.auto_download_model if allow_download is None else allow_download)
        self.session = Llama(model_path=str(folder / MANIFEST['files'][0]['filename']),
                             embedding=True, pooling_type=LLAMA_POOLING_TYPE_MEAN,
                             n_ctx=256, n_batch=256, n_threads=2, n_threads_batch=2,
                             n_gpu_layers=0, verbose=False)

    def encode(self, texts):
        if not self.gate.acquire(blocking=False):
            raise RuntimeBusy('Reference indexing is busy. Retry shortly.')
        try:
            if self.session is None:
                raise RuntimeError('Embedding model unavailable; run bootstrap.py')
            vectors = []
            for start in range(0, len(texts), 4):
                pooled = np.array(self.session.embed(texts[start:start+4], normalize=True, truncate=True))
                if not np.isfinite(pooled).all() or pooled.shape[1] != MANIFEST['dimensions']:
                    raise RuntimeError('Invalid embedding output')
                vectors.extend(pooled.astype(np.float32).tolist())
            return vectors
        finally:
            self.gate.release()


embedding_engine = EmbeddingEngine()
