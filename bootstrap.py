"""Idempotent local setup: Python environment, dependencies, pinned model, inference check."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def ensure_launch_runtime():
    """Called only by the executable entry point, never when importing the ASGI app."""
    from importlib import metadata
    ready = sys.version_info[:2] == (3, 12)
    if ready:
        try:
            for line in (ROOT / 'requirements.txt').read_text().splitlines():
                if '==' in line and not line.lstrip().startswith('#'):
                    name, expected = line.split('==')
                    if metadata.version(name) != expected:
                        ready = False
                        break
            if ready:
                __import__('llama_cpp')  # Also check that its native library loads.
        except (metadata.PackageNotFoundError, ImportError, OSError):
            ready = False
    if not ready:
        if os.environ.get('VOLTFORGE_AI_MANAGED_LAUNCH') == '1':
            raise RuntimeError('Managed AI runtime is not usable. Run python bootstrap.py --verify and inspect dependency installation; automatic launch stopped to avoid a restart loop.')
        print('Starting VoltForge AI in its managed Python 3.12 runtime...', flush=True)
        raise SystemExit(subprocess.call([sys.executable, str(ROOT / 'bootstrap.py'), '--run'], cwd=ROOT))


def run(args):
    subprocess.run([str(arg) for arg in args], cwd=ROOT, check=True)


def supported(python):
    try:
        return subprocess.check_output([str(python), '-c', 'import sys; print(sys.version_info[:2] == (3,12))'], text=True).strip() == 'True'
    except (OSError, subprocess.SubprocessError):
        return False


def create_environment(path):
    python = path / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if python.exists():
        if not supported(python):
            raise RuntimeError('Existing environment uses an unsupported Python; choose a new --venv path.')
        return python
    if path.exists() and any(path.iterdir()):
        raise RuntimeError('Refusing to overwrite a non-empty environment directory')
    base = sys.executable
    if not supported(base):
        candidates = [shutil.which('python3.12')]
        if os.name == 'nt' and shutil.which('py'):
            for version in ('3.12',):
                result = subprocess.run(['py', '-' + version, '-c', 'import sys; print(sys.executable)'], capture_output=True, text=True)
                if result.returncode == 0:
                    candidates.append(result.stdout.strip())
        base = next((p for p in candidates if p and supported(p)), None)
    if not base:
        # uv is only an installer for a managed Python, not an AI server or runtime.
        installer = ROOT / '.bootstrap'
        if subprocess.run([sys.executable, '-c', 'import pip'], capture_output=True).returncode:
            run([sys.executable, '-m', 'ensurepip', '--upgrade'])
        run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '--target', installer, 'uv==0.9.9'])
        uv = installer / 'uv' / ('uv.exe' if os.name == 'nt' else 'uv')
        if not uv.exists():
            uv = installer / ('bin/uv.exe' if os.name == 'nt' else 'bin/uv')
        run([uv, 'python', 'install', '3.12'])
        base = subprocess.check_output([str(uv), 'python', 'find', '3.12'], text=True).strip()
    run([base, '-m', 'venv', path])
    return python


def prepare_token():
    config_path, token_path = ROOT / 'config.json', ROOT / '.service-token'
    config = json.loads(config_path.read_text(encoding='utf-8')) if config_path.exists() else {}
    token = os.environ.get('VOLTFORGE_AI_API_TOKEN') or os.environ.get('API_TOKEN') or config.get('api_token')
    if not token_path.exists():
        with token_path.open('x', encoding='utf-8') as output:
            output.write(token or secrets.token_urlsafe(48))
        if os.name != 'nt':
            token_path.chmod(0o600)
    if config.get('api_token'):
        # Preserve the configured gateway token while removing it from tracked config.
        existing = token_path.read_text(encoding='utf-8').strip()
        if existing != config['api_token'] and not (os.environ.get('VOLTFORGE_AI_API_TOKEN') or os.environ.get('API_TOKEN')):
            raise RuntimeError('Token file and config disagree; resolve this before starting.')
        config.pop('api_token')
        temporary = config_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
        os.replace(temporary, config_path)
    print('Service token stored privately in .service-token (or supplied by environment). Backend must use the same token.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--venv', type=Path, default=ROOT / '.venv')
    parser.add_argument('--run', action='store_true', help='Start the service after successful setup')
    parser.add_argument('--verify', action='store_true', help='Load model and perform an inference smoke check')
    parser.add_argument('--source-build', action='store_true', help='Explicit native build for unsupported wheel platforms; requires a C++ toolchain')
    parser.add_argument('--force', '--clean', action='store_true', help='Download a replacement model; existing artifact is preserved until verification succeeds. Stop the service first.')
    args = parser.parse_args()
    python = create_environment(args.venv.resolve())
    requirements = ROOT / 'requirements.txt'
    fingerprint = hashlib.sha256(requirements.read_bytes()).hexdigest()
    receipt = args.venv / '.voltforge-dependencies'
    check_code = "import fastapi,uvicorn,pydantic,httpx,llama_cpp; import importlib.metadata as m; from pathlib import Path; assert all(m.version(n)==v for n,v in (line.split('==') for line in Path('requirements.txt').read_text().splitlines() if '==' in line))"
    check = subprocess.run([str(python), '-c', check_code], cwd=ROOT, capture_output=True)
    if not receipt.exists() or receipt.read_text() != fingerprint or check.returncode:
        if subprocess.run([str(python), '-c', 'import pip'], capture_output=True).returncode:
            run([python, '-m', 'ensurepip', '--upgrade'])
        command = [python, '-m', 'pip', 'install', '--disable-pip-version-check', '-r', requirements,
                   '--extra-index-url', 'https://abetlen.github.io/llama-cpp-python/whl/cpu']
        if not args.source_build:
            command.append('--only-binary=llama-cpp-python')
        run(command)
        run([python, '-m', 'pip', 'check'])
        receipt.write_text(fingerprint)
    prepare_token()
    run([python, 'download_model.py', *(['--force'] if args.force else [])])
    run([python, '-c', "from config import get_settings; from embedding_model import prepare; prepare() if get_settings().embeddings_enabled else None"])
    if args.verify:
        run([python, 'scripts/verify_runtime.py'])
    if args.run:
        # One process/one model. No Ollama, separate inference server, or Uvicorn workers.
        environment = dict(os.environ, VOLTFORGE_AI_MANAGED_LAUNCH='1')
        return subprocess.call([str(python), str(ROOT / 'main.py')], cwd=ROOT, env=environment)
    print('Setup complete. Start with python bootstrap.py --run')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, subprocess.SubprocessError, ValueError) as error:
        print('Setup failed:', type(error).__name__, '- check the preceding installer output; existing model/data retained.')
        raise SystemExit(1)
