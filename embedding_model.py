"""Pinned optional embedding artifacts; reuse the verified resumable downloader."""
import argparse
import hashlib
import json
from pathlib import Path
from download_model import ensure_model

ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / 'embedding_manifest.json').read_text(encoding='utf-8'))
MODEL_ID = hashlib.sha256(json.dumps(MANIFEST, sort_keys=True).encode()).hexdigest()


def prepare(root=None, *, allow_download=True, manifest=MANIFEST):
    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    folder = Path(root or ROOT / 'model' / 'embeddings') / identity
    for artifact in manifest['files']:
        name = Path(artifact['filename'])
        if name.is_absolute() or '..' in name.parts:
            raise ValueError('Invalid artifact path')
        ensure_model(folder / name, allow_download=allow_download,
                     manifest={**artifact, 'repository': manifest['repository'], 'revision': manifest['revision']})
    return folder


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--root', type=Path)
    args = parser.parse_args()
    prepare(args.root, allow_download=not args.verify_only)
    print('Embedding artifacts verified:', MODEL_ID)
