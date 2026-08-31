"""SSE helpers for deterministic transient waveform delivery."""

import asyncio
import json
import logging
import math
from typing import Any, AsyncIterator, Dict

from api.chat import stream_chat_sse  # Backward-compatible import for callers.
from api.schemas import SimulationStreamRequest


logger = logging.getLogger("voltforge-ai.sse")


async def stream_simulation_sse(payload: SimulationStreamRequest) -> AsyncIterator[str]:
    """Stream bounded deterministic waveform samples for the requested probes."""
    try:
        yield f"event: sim_start\ndata: {json.dumps({'status': 'RUNNING', 'board': payload.boardType})}\n\n"

        total_samples = min(200, int((payload.durationMs / 1000.0) * payload.sampleRateHz))
        probes = payload.probes or ["VCC", "GND", "D13", "ANALOG_A0"]

        for index in range(total_samples):
            seconds = index * (1.0 / max(1, payload.sampleRateHz))
            sample: Dict[str, Any] = {
                "timestamp": round(seconds * 1000, 2),
                "signals": {},
            }
            for probe in probes:
                lowered = probe.lower()
                if "gnd" in lowered:
                    sample["signals"][probe] = 0.0
                elif "vcc" in lowered or "5v" in lowered:
                    sample["signals"][probe] = 5.0 + 0.05 * math.sin(seconds * 50)
                elif "3.3" in lowered:
                    sample["signals"][probe] = 3.3 + 0.02 * math.sin(seconds * 50)
                elif "pwm" in lowered or "d13" in lowered or "led" in lowered:
                    sample["signals"][probe] = 5.0 if math.sin(seconds * 20) > 0 else 0.0
                elif "analog" in lowered or "a0" in lowered or "sensor" in lowered:
                    sample["signals"][probe] = round(2.5 + 1.8 * math.sin(seconds * 5), 3)
                else:
                    sample["signals"][probe] = round(
                        3.3 * (0.5 + 0.5 * math.sin(seconds * 10)), 3
                    )
            yield f"event: sample\ndata: {json.dumps(sample)}\n\n"
            await asyncio.sleep(0.01)

        yield f"event: sim_end\ndata: {json.dumps({'status': 'COMPLETED', 'samples': total_samples})}\n\n"
    except Exception as error:
        logger.error("Simulation SSE stream failed (errorType=%s)", type(error).__name__)
        yield (
            "event: error\ndata: "
            + json.dumps({
                "code": "SIMULATION_STREAM_FAILED",
                "message": "The local simulation stream could not complete.",
            })
            + "\n\n"
        )


__all__ = ["stream_chat_sse", "stream_simulation_sse"]
