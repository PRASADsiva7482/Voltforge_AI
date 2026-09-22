"""Unavailable legacy adapter contracts; these are not neural acceptance tests."""
import asyncio
import socket
from unittest.mock import patch
import pytest
from model.local_engine import LocalEngine, SecurityViolationError, validate_private_server_url


def test_retired_adapter_has_no_model_identity_or_readiness():
    health = LocalEngine().health()
    assert health["ready"] is False
    assert health["runtimeOperational"] is False
    assert health["generationSource"] == "unavailable"
    for field in ("artifactId", "parameterCount", "contextLength", "device"):
        assert health[field] is None


@pytest.mark.parametrize("method", ["generate_task_record", "generate_text", "generate_stream"])
def test_no_generation_entrypoint_substitutes_rules_or_connects(method):
    engine = LocalEngine()
    with asyncio.Runner() as runner, patch.object(socket.socket, "connect", side_effect=AssertionError("connection attempted")):
        with pytest.raises(RuntimeError, match="MODEL_RUNTIME_NOT_IMPLEMENTED"):
            if method == "generate_task_record":
                engine.generate_task_record({"input": {"user": {"text": "What is an LED?"}}})
            elif method == "generate_text":
                runner.run(engine.generate_text("What is an LED?"))
            else:
                async def consume():
                    return [chunk async for chunk in engine.generate_stream("What is an LED?")]
                runner.run(consume())


def test_cancelled_generation_cannot_begin():
    class Cancelled:
        def is_cancelled(self):
            return True
    with pytest.raises(asyncio.CancelledError):
        LocalEngine().generate_task_record({}, cancellation_token=Cancelled())


@pytest.mark.parametrize("url", ["http://8.8.8.8:8000", "https://api.openai.com/v1", "http://ai-server:8000"])
def test_unverified_endpoint_is_not_implicitly_approved(url):
    # Private DNS resolution/transport implementation remains LLM-TASK-043.
    with pytest.raises(SecurityViolationError):
        validate_private_server_url(url)
