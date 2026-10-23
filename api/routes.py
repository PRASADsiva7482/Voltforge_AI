"""
API Endpoints for VoltForge AI Microservice.
"""

import re
import asyncio
import json
import uuid
import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request
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
from api.research import ResearchRequest, research_capabilities, require_supported_research
from engine.web_research import ResearchError, web_research
from engine.electronics import (
    get_hardware_coverage_report,
    get_component_coverage_report,
    workspace_wire_suggestions,
    validate_circuit_state,
)
from engine.llm import get_llm_engine
from engine.runtime import RuntimeBusy, GenerationTimedOut
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
        "responseMode": "neural" if is_model_loaded else "unavailable",
        "model": settings.model_name,
        "provider": settings.model_provider,
        "engine": "llama-cpp-python (in-process)" if is_model_loaded else "model unavailable",
        "neuralLoaded": is_model_loaded,
        "version": "2.1.0"
    }


@router.get("/research/capabilities")
def get_research_capabilities() -> Dict[str, Any]:
    return research_capabilities()


@router.post("/research/search")
async def research_search(payload: ResearchRequest, request: Request):
    """Internal service-authenticated retrieval; not a browser URL proxy."""
    async def disconnected():
        # Pydantic has already consumed the request body. Await ASGI disconnect
        # directly so task cancellation is not swallowed by polling cancel scopes.
        while True:
            event = await request.receive()
            if event['type'] == 'http.disconnect':
                return
    work = asyncio.create_task(web_research.search(payload))
    disconnect = asyncio.create_task(disconnected())
    try:
        done, _ = await asyncio.wait((work, disconnect), return_when=asyncio.FIRST_COMPLETED)
        if work in done:
            return await work
        raise HTTPException(499, detail={'code': 'WEB_CANCELLED', 'message': 'Research cancelled after disconnection.'})
    except ResearchError as error:
        raise HTTPException(error.status, detail={'code': error.code, 'message': error.message}) from None
    finally:
        work.cancel()
        disconnect.cancel()
        await asyncio.gather(work, disconnect, return_exceptions=True)


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
    
    wire_suggestions = workspace_wire_suggestions(board_type, components, wires) if wants_wiring or wants_edit else []

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
    known_wire_ids = {wire.get('id') for wire in wires if wire.get('id')}
    if wants_edit or wants_wiring:
        validation = validate_circuit_state(board_type, components, wires)
        for issue in validation.get("issues", []):
            msg = issue.get("message", "")
            if "short circuit" in msg.lower() and issue.get('componentId') in known_wire_ids:
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
    require_supported_research(payload.research)
    logger.info("Received authenticated chat request")
    result = llm_engine.generate(
        message=payload.message,
        board_type=payload.boardType or "unspecified",
        components=payload.components,
        wires=payload.wires,
        code=payload.code or "",
        history=payload.history,
        simulation_state=payload.simulationState,
        context=payload.context or "",
        preferences=payload.approvedUserPreferences,
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

        try:
            require_supported_research(payload.research, streaming=True)
        except HTTPException as error:
            yield f'event: error\ndata: {json.dumps({"type": "error", **error.detail})}\n\n'
            return

        if llm_engine.llm is None:
            yield 'event: error\ndata: {"type":"error","code":"AI_MODEL_UNAVAILABLE","message":"The local AI model is unavailable. Retry when it is ready."}\n\n'
            return

        evidence = [source.model_dump(mode='json') for source in payload.authorizedDocumentEvidence]
        used_sources = []
        if payload.research.documentIds and not evidence:
            yield 'event: error\ndata: {"type":"error","code":"DOCUMENT_EVIDENCE_REQUIRED","message":"Authorized document text is unavailable. Reselect the document or upload it again."}\n\n'
            return
        if evidence:
            yield 'event: status\ndata: {"type":"status","stage":"generating","summary":"Writing from selected document excerpts"}\n\n'
        if payload.research.mode != 'off':
            yield 'event: status\ndata: {"type":"status","stage":"searching","summary":"Searching public sources"}\n\n'
            try:
                result = await web_research.search(payload.research.model_copy(update={'documentIds': []}))
            except ResearchError as error:
                yield f'event: error\ndata: {json.dumps({"type": "error", "code": error.code, "message": error.message})}\n\n'
                return
            web_sources = result['sources']
            if not web_sources:
                yield 'event: error\ndata: {"type":"error","code":"WEB_NO_RESULTS","message":"No usable sources were found. Refine the public query."}\n\n'
                return
            # Keep room for both explicit document and public web selections.
            evidence = [evidence[0], web_sources[0], *evidence[1:3], *web_sources[1:2]] if evidence else web_sources
            yield 'event: status\ndata: {"type":"status","stage":"generating","summary":"Writing from retrieved sources"}\n\n'

        # 2. Token-by-token streaming
        full_text = []
        research_args = {'evidence': evidence, 'used_sources': used_sources} if evidence else {}
        stream = llm_engine.stream_chat(
            message=payload.message, board_type=payload.boardType or "unspecified",
            components=payload.components, wires=payload.wires, code=payload.code or "",
            history=payload.history, simulation_state=payload.simulationState,
            context=payload.context or "", preferences=payload.approvedUserPreferences,
            **research_args,
        )
        sources_sent = False
        try:
            async for token in stream:
                if used_sources and not sources_sent:
                    # The worker filled these only after token-budgeting the prompt.
                    yield f'event: sources\ndata: {json.dumps({"type": "sources", "citations": used_sources, "attemptId": payload.attemptId})}\n\n'
                    sources_sent = True
                full_text.append(token)
                delta_payload = {
                    "type": "token",
                    "delta": token
                }
                yield f"event: token\ndata: {json.dumps(delta_payload)}\n\n"
        except RuntimeBusy:
            yield 'event: error\ndata: {"type":"error","code":"AI_BUSY","message":"AI is busy. Retry shortly."}\n\n'
            return
        except GenerationTimedOut:
            yield 'event: error\ndata: {"type":"error","code":"AI_GENERATION_TIMEOUT","message":"Generation timed out. Try a shorter question."}\n\n'
            return
        except ValueError:
            yield 'event: error\ndata: {"type":"error","code":"AI_CONTEXT_LIMIT","message":"Input exceeds the local model context. Send a shorter question."}\n\n'
            return
        except Exception:
            yield 'event: error\ndata: {"type":"error","code":"AI_GENERATION_INTERRUPTED","message":"Local AI generation interrupted"}\n\n'
            return
        finally:
            await stream.aclose()

        # 3. Synthesize proposals based on full interaction
        reply_text = "".join(full_text)
        if evidence:
            allowed = {s['citationId'] for s in used_sources}
            reply_text = re.sub(r'\[(S\d+)\]', lambda m: m.group(0) if m[1] in allowed else '[unverified reference]', reply_text)
        if not reply_text.strip():
            yield 'event: error\ndata: {"type":"error","code":"AI_EMPTY_RESPONSE","message":"The model returned no response. Please retry."}\n\n'
            return
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
            "citations": used_sources,
            "attemptId": payload.attemptId,
            "wireSuggestions": proposals["wireSuggestions"],
            "codeFixes": proposals["codeFixes"],
            "additions": proposals["additions"],
            "removals": proposals["removals"]
        }
        yield f"event: complete\ndata: {json.dumps(complete_payload)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_with_keepalive(sse_generator()), media_type="text/event-stream", headers={"Cache-Control": "no-store, no-transform", "X-Accel-Buffering": "no"})


async def _with_keepalive(events, interval=10):
    """Keep the existing gateway connection alive during retrieval/native prefill.

    SSE comments are transport heartbeats, not generated tokens or fake progress.
    One pending read is shared across heartbeats and cancelled on disconnect.
    """
    pending = None
    try:
        while True:
            pending = asyncio.create_task(anext(events))
            while not (await asyncio.wait({pending}, timeout=interval))[0]:
                yield ': keepalive\n\n'
            try:
                event = pending.result()
            except StopAsyncIteration:
                return
            pending = None
            yield event
    finally:
        if pending is not None:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        await events.aclose()


@router.post("/generate-code", response_model=AiGenerateResponse)
def generate_code(payload: GenerateCodeRequest) -> AiGenerateResponse:
    prompt = payload.prompt or "Generate microcontroller firmware code for placed components."
    result = llm_engine.generate(
        message=prompt,
        board_type=payload.boardType or "unspecified",
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
    suggestions = workspace_wire_suggestions(payload.boardType or "unspecified", payload.components, payload.wires)
    return AiGenerateResponse(
        status="SUCCESS" if suggestions else "UNAVAILABLE",
        message=(f"{len(suggestions)} I2C signal suggestion(s) from supplied pin definitions. Verify electrical compatibility; power wiring is not included."
                 if suggestions else "No supported routing can be established from these pin definitions. Provide the exact board and component pinout; no connections were invented."),
        wireSuggestions=suggestions, confidence=None,
    )


@router.post("/validate-circuit", response_model=AiValidationResponse)
def validate_circuit(payload: ValidateRequest) -> AiValidationResponse:
    result = validate_circuit_state(
        board_type=payload.boardType or "unspecified",
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
