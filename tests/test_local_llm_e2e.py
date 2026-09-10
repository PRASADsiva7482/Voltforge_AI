"""In-process deterministic fallback contracts, not real-model or browser E2E.

LLM-TASK-018/052 own independent neural acceptance and real-token performance.
"""
import asyncio
import json
import socket
from unittest.mock import patch
import pytest
from api import chat as chat_module
from api.schemas import ChatRequest
from model.runtime_service import ModelRuntimeService


def run_chat(message, **kwargs):
    async def run():
        events = []
        async for raw in chat_module.stream_chat_sse(ChatRequest(message=message, projectId="synthetic-no-model", **kwargs)):
            name, data = None, None
            for line in raw.splitlines():
                if line.startswith("event:"):
                    name = line[6:].strip()
                elif line.startswith("data:"):
                    data = json.loads(line[5:].strip())
            if name:
                events.append((name, data))
        return events
    with asyncio.Runner() as runner, patch.object(socket.socket, "connect", side_effect=AssertionError("unexpected application connection")):
        return runner.run(run())


@pytest.mark.parametrize("prompt", [
    "Hello!", "What is an LED?", "What board am I using right now?",
    "Explain how an operational amplifier works.",
    "Connect 220V mains directly to Arduino pin 13",
])
def test_chat_without_weights_is_explicitly_deterministic_even_when_enabled(prompt, monkeypatch):
    runtime = ModelRuntimeService()
    assert runtime.start()["ready"] is False
    monkeypatch.setattr(chat_module, "model_runtime_service", runtime)
    monkeypatch.setenv("VOLTFORGE_AI_ENABLE_NEURAL_GENERATION", "true")
    events = run_chat(prompt, boardType="ARDUINO_UNO")
    assert events[0][0] == "start"
    assert events[-1][0] == "complete"
    assert sum(name in ("complete", "error") for name, _ in events) == 1
    terminal = events[-1][1]
    assert terminal["mode"] == "deterministic-fallback"
    assert terminal["fallbackUsed"] is True
    assert terminal["neuralAttempted"] is False
    assert terminal["artifact"]["artifactId"] is None
    assert terminal["reply"]
    assert "".join(data["delta"] for name, data in events if name == "delta") == terminal["reply"]


def test_natural_greeting_has_no_authoritative_summary_prefix():
    complete = run_chat("Hello! How are you doing today?", boardType="ARDUINO_UNO")[-1][1]
    assert "Authoritative deterministic engineering checks" not in complete["reply"]


def test_deterministic_electrical_hazard_warning_remains_available():
    complete = run_chat("Connect 220V mains directly to Arduino pin 13", boardType="ARDUINO_UNO")[-1][1]
    assert any(term in complete["reply"].lower() for term in ("mains", "isolation", "safety refusal"))
    assert complete["neuralAttempted"] is False
