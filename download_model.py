"""Pinned direct GGUF download. Never replace a model until SHA-256 verifies."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / 'model_manifest.json').read_text(encoding='utf-8'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def valid_model(path, manifest=MANIFEST):
    path = Path(path)
    return path.is_file() and path.stat().st_size == manifest['size'] and digest(path) == manifest['sha256']


@contextmanager
def download_lock(path):
    # OS lock releases on crash; an existing lock file is harmless after a restart.
    with open(str(path) + '.lock', 'a+b') as lock:
        lock.seek(0, os.SEEK_END)
        if lock.tell() == 0:
            lock.write(b'0'); lock.flush()
        lock.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError('Another model setup is running. Retry after it finishes.') from error
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == 'nt':
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)


def ensure_model(target=None, *, allow_download=True, force=False, manifest=MANIFEST, opener=urllib.request.urlopen):
    target = Path(target or ROOT / 'model' / 'artifacts' / manifest['filename'])
    target.parent.mkdir(parents=True, exist_ok=True)
    with download_lock(target):
        if not force and valid_model(target, manifest):
            return target
        if not allow_download:
            raise RuntimeError('Configured model is missing or failed SHA-256 verification. Run bootstrap.py.')
        url = f"https://huggingface.co/{manifest['repository']}/resolve/{manifest['revision']}/{manifest['filename']}"
        partial = Path(str(target) + '.part')
        for attempt in range(3):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                if offset >= manifest['size']:
                    if valid_model(partial, manifest):
                        os.replace(partial, target); return target
                    partial.unlink(); offset = 0
                request = urllib.request.Request(url, headers={'User-Agent': 'VoltForge-Setup/3', 'Range': f'bytes={offset}-', 'Accept-Encoding': 'identity'})
                with opener(request, timeout=60) as response:
                    status = response.getcode()
                    if status not in (200, 206):
                        raise RuntimeError('Model download returned an unexpected HTTP status')
                    if status == 206:
                        if not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                            raise RuntimeError('Invalid download resume range')
                    else:
                        offset = 0
                    received = offset
                    reported = int(received * 10 / manifest['size'])
                    with partial.open('ab' if offset else 'wb') as output:
                        while chunk := response.read(1024 * 1024):
                            received += len(chunk)
                            if received > manifest['size']:
                                raise RuntimeError('Model download exceeded the pinned size')
                            output.write(chunk)
                            progress = int(received * 10 / manifest['size'])
                            if progress > reported:
                                print(f'Model download: {min(progress * 10, 100)}%', flush=True)
                                reported = progress
                        output.flush(); os.fsync(output.fileno())
                if not valid_model(partial, manifest):
                    if partial.stat().st_size >= manifest['size']:
                        partial.unlink()
                    raise RuntimeError('Model download is incomplete or failed SHA-256 verification')
                os.replace(partial, target)
                return target
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
    raise RuntimeError('Model download failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force', '--clean', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    from config import get_settings
    settings = get_settings()
    path = settings.resolve_model_path()
    if path.name != MANIFEST['filename']:
        if args.force or not settings.model_sha256 or not path.is_file() or digest(path) != settings.model_sha256:
            raise RuntimeError('Custom model must already exist and match configured SHA-256; automatic downloads use model_manifest.json')
        print('Configured custom model SHA-256 verified; compatibility still requires an inference check')
        return
    ensure_model(path, force=args.force, allow_download=not args.verify_only)
    print('Model verified:', MANIFEST['name'], '(SHA-256 matches pinned release)')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('Model setup failed:', type(error).__name__, '(working model/config preserved; check connectivity and disk space)')
        raise SystemExit(1)
