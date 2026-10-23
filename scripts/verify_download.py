"""Opt-in real network check; downloads a temporary model without replacing the active one."""
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from download_model import MANIFEST, ensure_model, valid_model

cache = ROOT / '.bootstrap'
cache.mkdir(exist_ok=True)
directory = Path(tempfile.mkdtemp(prefix='download-check-', dir=cache)).resolve()
assert directory.is_relative_to(cache.resolve())
target = directory / MANIFEST['filename']
started = time.monotonic()
try:
    ensure_model(target)
    assert valid_model(target)
    print(f'Real pinned download verified: {MANIFEST["size"]} bytes, {time.monotonic() - started:.1f}s')
finally:
    for suffix in ('', '.part', '.lock'):
        Path(str(target) + suffix).unlink(missing_ok=True)
    directory.rmdir()
