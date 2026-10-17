"""
VoltForge AI - Model Downloader & Health Verification Script
Downloads the official clean base model (Qwen 2.5 Coder 1.5B Instruct Q4_K_M GGUF)
from Hugging Face into model/artifacts/.
"""

import os
import sys
import json
import time
import urllib.request
from pathlib import Path

# Model details
MODEL_FILENAME = "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
HUGGINGFACE_URL = "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
EXPECTED_SIZE = 1117320768  # Exactly ~1.04 GiB (1,117,320,768 bytes)

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "model" / "artifacts"
TARGET_FILE = ARTIFACTS_DIR / MODEL_FILENAME
CONFIG_FILE = BASE_DIR / "config.json"


def format_bytes(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"


def update_config_model_path():
    """Ensure config.json points to the fresh base model file."""
    if not CONFIG_FILE.exists():
        print(f"[!] Warning: config.json not found at {CONFIG_FILE}")
        return

    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        relative_path = f"model/artifacts/{MODEL_FILENAME}"
        if cfg.get("model_path") != relative_path:
            cfg["model_path"] = relative_path
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            print(f"[OK] Updated config.json model_path -> {relative_path}")
        else:
            print(f"[OK] config.json already points to {relative_path}")
    except Exception as e:
        print(f"[!] Error updating config.json: {e}")


def verify_model_in_ram():
    """Attempt to load the model with llama-cpp-python to verify integrity."""
    print("\n--- Verifying Model with llama-cpp-python in RAM ---")
    try:
        from llama_cpp import Llama
        print(f"Loading {TARGET_FILE.name} into context...")
        t0 = time.time()
        llm = Llama(
            model_path=str(TARGET_FILE),
            n_ctx=512,
            n_threads=4,
            verbose=False
        )
        load_time = time.time() - t0
        print(f"[OK] Model verified successfully in RAM in {load_time:.2f}s!")
        print(f"    Architecture: {llm.metadata.get('general.architecture', 'qwen2')}")
        print(f"    Name:         {llm.metadata.get('general.name', 'Qwen2.5 Coder 1.5B Instruct')}")
        
        # Test quick generation
        res = llm.create_chat_completion(
            messages=[{"role": "user", "content": "Ping test"}],
            max_tokens=10
        )
        print("[OK] Neural inference smoke test passed!")
        return True
    except ImportError:
        print("[!] Note: llama-cpp-python is not installed in the current Python environment.")
        print(f"    To run with neural acceleration, use the virtual environment:")
        print(f"    .\\.venv\\Scripts\\python.exe main.py")
        return True
    except Exception as e:
        print(f"[FAIL] Model load verification failed: {e}")
        return False


def download_model(force: bool = False):
    """Download clean base model from Hugging Face with progress indicator."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if TARGET_FILE.exists() and not force:
        actual_size = TARGET_FILE.stat().st_size
        if actual_size == EXPECTED_SIZE:
            print(f"[OK] Fresh model already exists at: {TARGET_FILE}")
            print(f"    File size: {format_bytes(actual_size)} (Exact match with Hugging Face release)")
            update_config_model_path()
            verify_model_in_ram()
            return

        print(f"[!] Existing model file size mismatch: {actual_size} bytes (Expected: {EXPECTED_SIZE} bytes).")
        print("    Re-downloading clean model from scratch...")

    temp_file = TARGET_FILE.with_suffix(".tmp")
    if temp_file.exists():
        temp_file.unlink()

    print(f"--- Downloading Clean Base Model: Qwen 2.5 Coder 1.5B Instruct ---")
    print(f"URL:    {HUGGINGFACE_URL}")
    print(f"Target: {TARGET_FILE}")
    print(f"Size:   {format_bytes(EXPECTED_SIZE)} (~1.04 GiB)")
    print("------------------------------------------------------------------")

    headers = {
        "User-Agent": "VoltForge-Model-Downloader/2.1.0"
    }
    req = urllib.request.Request(HUGGINGFACE_URL, headers=headers)

    start_time = time.time()
    last_print = start_time
    downloaded = 0

    try:
        with urllib.request.urlopen(req) as response, open(temp_file, "wb") as out:
            total_size = int(response.headers.get("Content-Length", EXPECTED_SIZE))
            block_size = 1024 * 1024  # 1 MB blocks

            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                out.write(buffer)
                downloaded += len(buffer)

                now = time.time()
                if now - last_print >= 0.5 or downloaded == total_size:
                    percent = (downloaded / total_size) * 100
                    elapsed = now - start_time
                    speed = (downloaded / elapsed) if elapsed > 0 else 0
                    eta = (total_size - downloaded) / speed if speed > 0 else 0
                    sys.stdout.write(
                        f"\rDownloading: [{downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB] "
                        f"({percent:.1f}%) | Speed: {format_bytes(int(speed))}/s | ETA: {int(eta)}s   "
                    )
                    sys.stdout.flush()
                    last_print = now

        print("\n[OK] Download complete!")

        # Atomic rename
        if TARGET_FILE.exists():
            TARGET_FILE.unlink()
        temp_file.rename(TARGET_FILE)

        print(f"[OK] Model saved cleanly to: {TARGET_FILE}")
        update_config_model_path()
        verify_model_in_ram()

    except Exception as e:
        print(f"\n[FAIL] Download error: {e}")
        if temp_file.exists():
            temp_file.unlink()
        sys.exit(1)


if __name__ == "__main__":
    force_download = "--force" in sys.argv or "--clean" in sys.argv
    download_model(force=force_download)
