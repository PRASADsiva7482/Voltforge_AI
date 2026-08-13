import asyncio
import json
import logging
from typing import AsyncIterator, Dict, Any
from api.schemas import ChatRequest, ChatResponse
from engine.reasoning import ElectronicsReasoningOrchestrator

logger = logging.getLogger("voltforge-ai.sse")
orchestrator = ElectronicsReasoningOrchestrator()


async def stream_chat_sse(payload: ChatRequest) -> AsyncIterator[str]:
    """Emits SSE events: 'thought' steps, word tokens, then final JSON response metadata."""
    try:
        # Step 1: Thinking events
        yield f"event: thought\ndata: {json.dumps({'step': 'Analyzing canvas components and board configuration...'})}\n\n"
        await asyncio.sleep(0.05)

        yield f"event: thought\ndata: {json.dumps({'step': 'Checking electrical logic voltages and bus safety...'})}\n\n"
        await asyncio.sleep(0.05)

        # Step 2: Compute full response using orchestrator
        chat_resp: ChatResponse = orchestrator.process_chat(
            message=payload.message,
            board_type=payload.boardType or "ARDUINO_UNO",
            components=payload.components,
            wires=payload.wires,
            code=payload.code or ""
        )

        # Step 3: Emit token streaming
        words = chat_resp.reply.split(" ")
        for i, word in enumerate(words):
            token_str = word + (" " if i < len(words) - 1 else "")
            yield f"event: token\ndata: {json.dumps({'token': token_str})}\n\n"
            await asyncio.sleep(0.01)

        # Step 4: Emit final metadata event
        meta_payload = {
            "reply": chat_resp.reply,
            "hasCode": chat_resp.hasCode,
            "generatedCode": chat_resp.generatedCode,
            "confidence": chat_resp.confidence,
            "citations": chat_resp.citations,
            "wireSuggestions": chat_resp.wireSuggestions,
            "additions": chat_resp.additions,
            "removals": chat_resp.removals,
            "valueChanges": chat_resp.valueChanges,
            "codeFixes": chat_resp.codeFixes,
        }
        yield f"event: metadata\ndata: {json.dumps(meta_payload)}\n\n"
        yield "event: done\ndata: {}\n\n"

    except Exception as e:
        logger.error(f"SSE Streaming error: {e}")
        yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
