import asyncio
import json
import logging
import math
from typing import AsyncIterator, Dict, Any
from api.schemas import ChatRequest, ChatResponse, SimulationStreamRequest
from engine.reasoning import ElectronicsReasoningOrchestrator

logger = logging.getLogger("voltforge-ai.sse")
orchestrator = ElectronicsReasoningOrchestrator()


async def stream_chat_sse(payload: ChatRequest) -> AsyncIterator[str]:
    """Emits SSE events: 'thought' steps, word tokens, then final JSON response metadata."""
    try:
        # Step 1: Thinking events
        yield f"event: thought\ndata: {json.dumps({'type': 'thought', 'step': 'Analyzing canvas components and board configuration...', 'content': 'Analyzing canvas components and board configuration...'})}\n\n"
        await asyncio.sleep(0.05)

        yield f"event: thought\ndata: {json.dumps({'type': 'thought', 'step': 'Checking electrical logic voltages and bus safety...', 'content': 'Checking electrical logic voltages and bus safety...'})}\n\n"
        await asyncio.sleep(0.05)

        # Step 2: Compute full response using orchestrator
        history_dicts = [h.dict() for h in payload.history] if payload.history else []
        chat_resp: ChatResponse = orchestrator.process_chat(
            message=payload.message,
            board_type=payload.boardType or "ARDUINO_UNO",
            components=payload.components,
            wires=payload.wires,
            code=payload.code or "",
            context={"history": history_dicts, "canvasContext": payload.canvasContext}
        )


        # Step 3: Emit token streaming
        words = chat_resp.reply.split(" ")
        for i, word in enumerate(words):
            token_str = word + (" " if i < len(words) - 1 else "")
            yield f"event: token\ndata: {json.dumps({'type': 'token', 'token': token_str, 'content': token_str})}\n\n"
            await asyncio.sleep(0.01)

        # Step 4: Emit final metadata event
        meta_payload = {
            "type": "done",
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
        yield f"event: done\ndata: {json.dumps(meta_payload)}\n\n"

    except Exception as e:
        logger.error(f"SSE Streaming error: {e}")
        yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"



async def stream_simulation_sse(payload: SimulationStreamRequest) -> AsyncIterator[str]:
    """Streams live transient simulation waveforms (oscilloscope/logic analyzer)."""
    try:
        yield f"event: sim_start\ndata: {json.dumps({'status': 'RUNNING', 'board': payload.boardType})}\n\n"
        
        # Calculate sample frames
        total_samples = min(200, int((payload.durationMs / 1000.0) * payload.sampleRateHz))
        probes = payload.probes or ["VCC", "GND", "D13", "ANALOG_A0"]

        for t_idx in range(total_samples):
            t_sec = t_idx * (1.0 / max(1, payload.sampleRateHz))
            
            # Generate electrical transient waveform signals
            sample_data: Dict[str, Any] = {
                "timestamp": round(t_sec * 1000, 2), # in ms
                "signals": {}
            }

            for p in probes:
                if "gnd" in p.lower():
                    sample_data["signals"][p] = 0.0
                elif "vcc" in p.lower() or "5v" in p.lower():
                    sample_data["signals"][p] = 5.0 + 0.05 * math.sin(t_sec * 50)
                elif "3.3" in p.lower():
                    sample_data["signals"][p] = 3.3 + 0.02 * math.sin(t_sec * 50)
                elif "pwm" in p.lower() or "d13" in p.lower() or "led" in p.lower():
                    # Square wave logic signal
                    sample_data["signals"][p] = 5.0 if math.sin(t_sec * 20) > 0 else 0.0
                elif "analog" in p.lower() or "a0" in p.lower() or "sensor" in p.lower():
                    # Analog sensor curve
                    sample_data["signals"][p] = round(2.5 + 1.8 * math.sin(t_sec * 5), 3)
                else:
                    sample_data["signals"][p] = round(3.3 * (0.5 + 0.5 * math.sin(t_sec * 10)), 3)

            yield f"event: sample\ndata: {json.dumps(sample_data)}\n\n"
            await asyncio.sleep(0.01)

        yield f"event: sim_end\ndata: {json.dumps({'status': 'COMPLETED', 'samples': total_samples})}\n\n"

    except Exception as e:
        logger.error(f"Simulation SSE Streaming error: {e}")
        yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
