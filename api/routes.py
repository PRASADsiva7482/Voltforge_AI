"""
API Endpoints for VoltForge AI Microservice.
"""

import re
import json
import uuid
import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from api.schemas import (
    ChatRequest,
    ChatResponse,
    GenerateCodeRequest,
    ValidateRequest,
    AiGenerateResponse,
    AiValidationResponse,
)
from api.security import require_service_token
from engine.electronics import (
    get_hardware_coverage_report,
    get_component_coverage_report,
    resolve_pin_suggestions,
    validate_circuit_state,
)
from engine.llm import get_llm_engine
from config import get_settings

logger = logging.getLogger("voltforge-ai.routes")

router = APIRouter(
    prefix="/voltForge-ai/api/v1/model",
    tags=["VoltForge AI"],
    dependencies=[Depends(require_service_token)],
)

llm_engine = get_llm_engine()


@router.get("/health")
def health() -> Dict[str, Any]:
    settings = get_settings()
    llm = get_llm_engine()
    is_model_loaded = llm.llm is not None
    return {
        "status": "ONLINE",
        "state": "ready" if is_model_loaded else "fallback",
        "ready": is_model_loaded,
        "code": "MODEL_RUNTIME_READY" if is_model_loaded else "MODEL_NOT_LOADED",
        "serviceReady": True,
        "responseMode": "neural" if is_model_loaded else "deterministic",
        "model": settings.model_name,
        "provider": settings.model_provider,
        "engine": "llama-cpp-python (in-process)" if is_model_loaded else "deterministic fallback",
        "neuralLoaded": is_model_loaded,
        "version": "2.1.0"
    }


@router.get("/hardware-coverage")
def hardware_coverage() -> Dict[str, Any]:
    return get_hardware_coverage_report()


@router.get("/component-coverage")
def component_coverage() -> Dict[str, Any]:
    return get_component_coverage_report()


def _resolve_workspace_proposals(
    user_message: str,
    reply_text: str,
    board_type: str,
    components: List[Dict[str, Any]],
    wires: List[Dict[str, Any]],
    existing_code: str
) -> Dict[str, Any]:
    """Inspects chat interaction and circuit state to synthesize actionable UI proposals."""
    user_lower = user_message.lower()
    
    # Detect if user asks for wiring, pinout, circuit modifications, fixes, or code
    wants_wiring = any(k in user_lower for k in ("wire", "connect", "pin", "wiring", "route", "suggest", "how to", "correct", "edit", "fix", "repair", "apply", "update", "workspace"))
    wants_edit = any(k in user_lower for k in ("correct", "edit", "fix", "repair", "apply", "update", "workspace", "code", "firmware", "sketch"))
    
    # Identify target microcontroller component ID from placed components
    board_id = "board"
    for c in components:
        t = str(c.get("type", "")).upper()
        if any(b in t for b in ("ARDUINO", "ESP", "PICO", "STM32", "TEENSY", "BOARD", "MCU")):
            board_id = c.get("id", "board")
            break

    wire_suggestions = []
    if wants_wiring or wants_edit:
        for c in components:
            c_type = c.get("type") or c.get("componentType") or ""
            c_id = c.get("id") or c.get("name") or c_type
            if c_type and c_id != board_id:
                wire_suggestions.extend(resolve_pin_suggestions(
                    component_type=c_type,
                    board_type=board_type,
                    component_id=c_id,
                    board_id=board_id
                ))

    # Extract firmware code if present
    code_match = re.search(r"```(?:cpp|c|arduino)?\n(.*?)```", reply_text, re.DOTALL)
    generated_code = code_match.group(1).strip() if code_match else None
    
    code_fixes = []
    if generated_code and wants_edit:
        code_fixes.append({
            "type": "code_replace",
            "description": f"Update active firmware sketch for {board_type}",
            "from": existing_code[:200] if existing_code else "",
            "to": generated_code,
            "line": 1
        })

    # Run circuit validation to propose safety additions and removals
    additions = []
    removals = []
    if wants_edit or wants_wiring:
        validation = validate_circuit_state(board_type, components, wires)
        for issue in validation.get("issues", []):
            msg = issue.get("message", "")
            if "short circuit" in msg.lower():
                removals.append({
                    "type": "wire_removal",
                    "wireId": issue.get("componentId", "short-wire"),
                    "reason": "Remove direct power-to-ground short circuit wire"
                })
            # A warning alone does not establish a resistor value, supply
            # voltage, or topology. Do not fabricate an automatic 220-ohm fix.

    return {
        "wireSuggestions": wire_suggestions,
        "codeFixes": code_fixes,
        "additions": additions,
        "removals": removals,
        "hasCode": generated_code is not None,
        "generatedCode": generated_code
    }


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    logger.info("Received authenticated chat request")
    result = llm_engine.generate(
        message=payload.message,
        board_type=payload.boardType or "unspecified",
        components=payload.components,
        wires=payload.wires,
        code=payload.code or "",
        history=payload.history,
        simulation_state=payload.simulationState,
        context=payload.context or ""
    )
    
    proposals = _resolve_workspace_proposals(
        user_message=payload.message if llm_engine.llm is not None else "",
        reply_text=result["reply"] if result["mode"] != "unavailable" else "",
        board_type=payload.boardType or "unspecified",
        components=payload.components,
        wires=payload.wires,
        existing_code=payload.code or ""
    )
    
    return ChatResponse(
        requestId=str(uuid.uuid4()),
        sessionId=payload.sessionId or str(uuid.uuid4()),
        projectRevision=payload.projectRevision,
        reply=result["reply"],
        hasCode=proposals["hasCode"],
        generatedCode=proposals["generatedCode"],
        confidence=result["confidence"],
        mode=result["mode"],
        model=result["model"],
        wireSuggestions=proposals["wireSuggestions"],
        additions=proposals["additions"],
        removals=proposals["removals"],
        codeFixes=proposals["codeFixes"]
    )


@router.post("/chat/stream")
async def chat_stream(payload: ChatRequest):
    """Server-Sent Events streaming chat endpoint."""
    logger.info("Received authenticated streaming chat request")
    session_id = payload.sessionId or str(uuid.uuid4())
    request_id = str(uuid.uuid4())

    async def sse_generator():
        # 1. Start event
        start_payload = {
            "type": "start",
            "sessionId": session_id,
            "requestId": request_id,
            "mode": "neural" if llm_engine.llm is not None else "unavailable",
            "model": get_settings().model_name
        }
        yield f"event: start\ndata: {json.dumps(start_payload)}\n\n"

        if llm_engine.llm is None:
            yield 'event: error\ndata: {"type":"error","code":"AI_MODEL_UNAVAILABLE","message":"The local AI model is unavailable. Retry when it is ready."}\n\n'
            return

        # 2. Token-by-token streaming
        full_text = []
        try:
            async for token in llm_engine.stream_chat(
                message=payload.message,
                board_type=payload.boardType or "unspecified",
                components=payload.components,
                wires=payload.wires,
                code=payload.code or "",
                history=payload.history,
                simulation_state=payload.simulationState,
                context=payload.context or ""
            ):
                full_text.append(token)
                delta_payload = {
                    "type": "token",
                    "delta": token
                }
                yield f"event: token\ndata: {json.dumps(delta_payload)}\n\n"
        except Exception:
            yield 'event: error\ndata: {"type":"error","code":"AI_GENERATION_INTERRUPTED","message":"Local AI generation interrupted"}\n\n'
            return

        # 3. Synthesize proposals based on full interaction
        reply_text = "".join(full_text)
        proposals = _resolve_workspace_proposals(
            user_message=payload.message if llm_engine.llm is not None else "",
            reply_text=reply_text,
            board_type=payload.boardType or "unspecified",
            components=payload.components,
            wires=payload.wires,
            existing_code=payload.code or ""
        )

        if proposals["wireSuggestions"] or proposals["codeFixes"] or proposals["additions"] or proposals["removals"]:
            proposal_payload = {
                "type": "proposal",
                "wireSuggestions": proposals["wireSuggestions"],
                "codeFixes": proposals["codeFixes"],
                "additions": proposals["additions"],
                "removals": proposals["removals"]
            }
            yield f"event: proposal\ndata: {json.dumps(proposal_payload)}\n\n"

        # 4. Complete event
        complete_payload = {
            "type": "complete",
            "sessionId": session_id,
            "reply": reply_text,
            "confidence": None,
            "mode": start_payload["mode"],
            "wireSuggestions": proposals["wireSuggestions"],
            "codeFixes": proposals["codeFixes"],
            "additions": proposals["additions"],
            "removals": proposals["removals"]
        }
        yield f"event: complete\ndata: {json.dumps(complete_payload)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


@router.post("/generate-code", response_model=AiGenerateResponse)
def generate_code(payload: GenerateCodeRequest) -> AiGenerateResponse:
    prompt = payload.prompt or "Generate microcontroller firmware code for placed components."
    result = llm_engine.generate(
        message=prompt,
        board_type=payload.boardType or "ARDUINO_UNO",
        components=payload.components,
        wires=payload.wires,
        code=payload.code or ""
    )
    return AiGenerateResponse(
        status="SUCCESS" if result["generatedCode"] else "UNAVAILABLE",
        message="Generated suggestion; compile and test before use." if result["generatedCode"] else result["reply"],
        generatedCode=result["generatedCode"],
        confidence=result["confidence"]
    )


@router.post("/suggest-wiring", response_model=AiGenerateResponse)
def suggest_wiring(payload: GenerateCodeRequest) -> AiGenerateResponse:
    board = payload.boardType or "ARDUINO_UNO"
    board_id = "board"
    for c in payload.components:
        t = str(c.get("type", "")).upper()
        if any(b in t for b in ("ARDUINO", "ESP", "PICO", "STM32", "TEENSY", "BOARD", "MCU")):
            board_id = c.get("id", "board")
            break

    suggestions = []
    for c in payload.components:
        c_type = c.get("type") or c.get("componentType") or ""
        c_id = c.get("id") or c.get("name") or c_type
        if c_type and c_id != board_id:
            suggestions.extend(resolve_pin_suggestions(
                component_type=c_type,
                board_type=board,
                component_id=c_id,
                board_id=board_id
            ))
            
    return AiGenerateResponse(
        status="SUCCESS",
        message=f"Generated {len(suggestions)} wire routing suggestion(s).",
        wireSuggestions=suggestions,
        confidence=None
    )


@router.post("/validate-circuit", response_model=AiValidationResponse)
def validate_circuit(payload: ValidateRequest) -> AiValidationResponse:
    result = validate_circuit_state(
        board_type=payload.boardType or "ARDUINO_UNO",
        components=payload.components,
        wires=payload.wires
    )
    return AiValidationResponse(**result)


# Compatibility endpoints for memory inspection
@router.get("/memory")
def get_memory(projectId: str = Query(...), sessionId: Optional[str] = Query(None)):
    return {
        "projectId": projectId,
        "sessionId": sessionId,
        "entries": [],
        "enabled": False,
        "entryCount": 0,
        "status": "unavailable"
    }


@router.put("/memory/preferences")
def update_memory_preferences(projectId: str = Query(...)):
    raise HTTPException(status_code=501, detail="Approved project facts are not supported by this AI service. Conversation history is stored by the application backend.")
