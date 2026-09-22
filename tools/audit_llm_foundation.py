"""Read-only foundation inventory; optional synthetic, in-process chat probes.

Never loads checkpoints, starts training, activates artifacts, or contacts a server.
Only --output writes a file. This is diagnostic evidence, not a model quality gate.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import sys
from unittest.mock import patch
import zipfile


AI_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = (
    "api/chat.py", "api/memory.py", "api/routes.py", "api/sse.py",
    "model/local_engine.py", "model/reasoning_llm.py", "model/runtime_service.py",
    "model/generation_quality.py", "model/registry_manager.py",
    "model/gen1/config.py", "model/gen1/model.py", "gen1_training/trainer.py",
    "gen1_training/data.py", "tools/activate_domain_artifact.py",
    "engine/reasoning.py", "engine/deterministic_tools.py", "engine/fuzzy_resolver.py", "grounding/gate.py",
    "tools/audit_llm_foundation.py",
    "tests/test_local_engine.py", "tests/test_local_llm_e2e.py",
    "model/registry/active_model.json",
    "model/tokenizers/vfdlm-byte-bpe-v1.1.0/tokenizer_manifest.json",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(relative: str) -> dict:
    return json.loads((AI_ROOT / relative).read_text(encoding="utf-8-sig"))


def function_inventory(relative: str, name: str) -> list[dict]:
    tree = ast.parse((AI_ROOT / relative).read_text(encoding="utf-8-sig"))
    return [
        {"name": node.name, "line": node.lineno, "end_line": node.end_lineno,
         "calls": sorted({ast.unparse(call.func) for call in ast.walk(node)
                          if isinstance(call, ast.Call)})}
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]


def synthetic_probes() -> dict:
    if str(AI_ROOT) not in sys.path:
        sys.path.insert(0, str(AI_ROOT))
    attempts = []

    def deny_network(*args, **kwargs):
        attempts.append("socket connection denied")
        raise RuntimeError("Network disabled by foundation audit")

    # No real user messages or saved conversations are used.
    prompts = [
        "Hello!", "What is an LED?", "What board am I using?",
        "Explain metastability and how a two flip-flop synchronizer helps.",
        "How can I debounce a mechanical push button without blocking the loop?",
        "Compare UART framing errors with an I2C bus stuck low.",
        "Why does an ADC reading change when its reference voltage changes?",
        "How should I measure sleep current without waking my microcontroller?",
    ]
    # Initialize the event loop before guarding sockets: Windows creates a local
    # socket pair for its wakeup pipe. No application connections are permitted.
    with asyncio.Runner() as runner, patch.object(socket.socket, "connect", deny_network), patch.object(
        socket.socket, "connect_ex", deny_network
    ), patch.object(socket, "create_connection", deny_network):
        from model.local_engine import LocalEngine
        from engine.deterministic_tools import DeterministicElectronicsTools
        from model.runtime_service import ModelRuntimeService
        from model.registry_manager import verify_registry
        from api.chat import stream_chat_sse
        from api.schemas import ChatRequest

        registry = verify_registry()
        runtime = ModelRuntimeService()
        runtime_health = runtime.start()
        runtime.stop()
        engine = LocalEngine()
        engine_health = engine.health()
        # Exclude endpoint values; the audit never reads .env files or credentials.
        engine_health.pop("serverUrl", None)
        rules = []
        rule_tools = DeterministicElectronicsTools()
        for prompt in prompts:
            solution = rule_tools.reason_and_solve(prompt)
            answer = solution.get("answer", "")
            rules.append({"prompt": prompt, "answer": answer,
                          "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
                          "answer_without_echo_sha256": hashlib.sha256(
                              answer.replace(prompt, "<PROMPT>").encode()).hexdigest()})

        async def chat_probes():
            results = []
            for prompt in prompts:
                events = []
                async for raw in stream_chat_sse(ChatRequest(
                    message=prompt, projectId="synthetic-foundation-audit",
                    sessionId="synthetic-foundation-audit", boardType="ARDUINO_UNO",
                )):
                    name, data = None, None
                    for line in raw.splitlines():
                        if line.startswith("event:"):
                            name = line[6:].strip()
                        elif line.startswith("data:"):
                            data = json.loads(line[5:].strip())
                    if name:
                        events.append((name, data))
                terminal = next((data for name, data in reversed(events)
                                 if name in ("complete", "error")), {})
                payload = terminal.get("payload", terminal)
                reply = payload.get("reply", "")
                results.append({"prompt": prompt, "events": [name for name, _ in events],
                                "mode": terminal.get("mode"),
                                "fallback_used": payload.get("fallbackUsed"),
                                "neural_attempted": payload.get("neuralAttempted"),
                                "reply": reply,
                                "reply_sha256": hashlib.sha256(reply.encode()).hexdigest()})
            return results

        chats = runner.run(chat_probes())
        groups = defaultdict(list)
        normalized_groups = defaultdict(list)
        for row in rules:
            if not row["answer"]:
                continue
            groups[row["answer_sha256"]].append(row["prompt"])
            normalized_groups[row["answer_without_echo_sha256"]].append(row["prompt"])
        return {
            "scope": "Synthetic in-process calls only; no HTTP gateway, browser or neural weights",
            "registry_signature_verified": True,
            "verified_registry_revision": registry["revision"],
            "runtime_start": {key: runtime_health.get(key) for key in
                              ("ready", "state", "code", "activeArtifactId", "runtimeOperational")},
            "wrapper_health_without_loading_weights": engine_health,
            "blocked_socket_attempts": len(attempts),
            "embedded_rule_results": rules,
            "identical_rule_answer_groups": [values for values in groups.values() if len(values) > 1],
            "identical_rule_answers_after_removing_prompt_echo": [
                values for values in normalized_groups.values() if len(values) > 1],
            "chat_results": chats,
        }


def inventory() -> dict:
    registry = read_json("model/registry/active_model.json")
    tokenizer = read_json("model/tokenizers/vfdlm-byte-bpe-v1.1.0/tokenizer_manifest.json")
    binary = AI_ROOT / "model/server/artifacts/voltforge-vfdlm-domain.bin"
    artifact = {"exists": binary.exists(), "deserialized": False}
    if binary.exists():
        artifact.update(bytes=binary.stat().st_size, sha256=digest(binary))
        with binary.open("rb") as handle:
            artifact["magic_hex"] = handle.read(16).hex()
            handle.seek(0)
            if handle.read(8) == b"VFDLM001":
                header_bytes = int.from_bytes(handle.read(4), "little")
                if 0 < header_bytes <= 1024 * 1024:
                    metadata = json.loads(handle.read(header_bytes))
                    artifact["custom_header_bytes"] = header_bytes
                    artifact["custom_header_keys"] = sorted(metadata)
                    artifact["header_claims"] = {key: metadata[key] for key in (
                        "model_name", "architecture", "parameter_count", "parameters",
                        "model", "format", "version", "model_id", "training",
                        "config", "parameterCount", "architectureId",
                    ) if key in metadata}
        artifact["is_zip_container"] = zipfile.is_zipfile(binary)
        if artifact["is_zip_container"]:
            with zipfile.ZipFile(binary) as archive:
                artifact["archive_entries"] = archive.namelist()[:30]
        legacy = AI_ROOT / "model/artifacts/model_weights.npz"
        if legacy.exists():
            artifact["identical_to_legacy_npz"] = digest(binary) == digest(legacy)
    return {
        "schema_version": 1,
        "audit_id": "voltforge-llm-foundation-audit-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_fingerprints": [{"path": path, "sha256": digest(AI_ROOT / path)}
                                for path in SOURCE_PATHS if (AI_ROOT / path).is_file()],
        "registry": {key: registry[key] for key in
                     ("revision", "activeArtifactId", "state", "releaseStatus", "artifacts")},
        "current_tokenizer": {"version": tokenizer["version"], "vocab_size": tokenizer["vocabSize"],
                              "split": tokenizer["lineage"]["split"]},
        "unregistered_domain_binary": artifact,
        "call_inventory": {
            "chat_generation": function_inventory("api/chat.py", "resolve_stage"),
            "task_record_generation": function_inventory("model/local_engine.py", "generate_task_record"),
            "deterministic_tool_constructor": function_inventory("engine/deterministic_tools.py", "__init__"),
            "wrapper_health": function_inventory("model/local_engine.py", "health"),
        },
        "limits": ["No training or benchmark rerun", "No artifact deserialization or activation",
                   "No deployed server or browser verification", "Inventory is not neural release approval"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--probe-chat", action="store_true")
    args = parser.parse_args()
    report = inventory()
    if args.probe_chat:
        report["synthetic_probes"] = synthetic_probes()
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(json.dumps({"report": str(args.output), "sha256": digest(args.output),
                          "active_artifact": report["registry"]["activeArtifactId"],
                          "binary": report["unregistered_domain_binary"]}))
    else:
        print(text)


if __name__ == "__main__":
    main()
