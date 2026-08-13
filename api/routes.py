import logging
from typing import Any, Dict
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from api.schemas import (
    ChatRequest,
    ChatResponse,
    CodeReviewRequest,
    GenerateCodeRequest,
    SchematicToCodeRequest,
    ValidateRequest,
)
from api.sse import stream_chat_sse
from circuit_verifier import ElectricalVerifier
from engine.code_generator import FirmwareCodeGenerator
from engine.reasoning import ElectronicsReasoningOrchestrator

logger = logging.getLogger("voltforge-ai.routes")

router = APIRouter(prefix="/voltForge-ai/api/v1/model", tags=["Voltforge AI"])
orchestrator = ElectronicsReasoningOrchestrator()


@router.get("/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "UP",
        "service": "Voltforge AI Microservice",
        "engine": "Voltforge Electronics Reasoning Engine",
        "ragEnabled": True,
        "version": "2.0.0",
    }


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    logger.info(f"Processing chat request: {payload.message}")
    return orchestrator.process_chat(
        message=payload.message,
        board_type=payload.boardType or "ARDUINO_UNO",
        components=payload.components,
        wires=payload.wires,
        code=payload.code or ""
    )


@router.post("/chat/stream")
async def chat_stream(payload: ChatRequest):
    logger.info(f"Processing streaming chat request: {payload.message}")
    return StreamingResponse(
        stream_chat_sse(payload),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/validate-circuit")
def validate_circuit(payload: ValidateRequest) -> Dict[str, Any]:
    logger.info("Processing circuit validation")
    v_res = ElectricalVerifier.verify_circuit(
        board_type=payload.boardType or "ARDUINO_UNO",
        components=payload.components,
        wires=payload.wires,
        code=payload.code or ""
    )
    return {
        "isValid": v_res["safetyScore"] >= 75 and not any(i["severity"] == "CRITICAL" for i in v_res["issues"]),
        "safetyScore": v_res["safetyScore"],
        "generalFeedback": v_res["generalFeedback"],
        "issues": v_res["issues"],
        "additions": v_res["additions"],
        "removals": v_res["removals"],
        "valueChanges": v_res["valueChanges"],
        "wireSuggestions": v_res["wireSuggestions"],
        "codeFixes": v_res["codeFixes"],
    }


@router.post("/suggest-wiring")
def suggest_wiring(payload: ValidateRequest) -> Dict[str, Any]:
    logger.info("Processing wiring suggestions")
    v_res = ElectricalVerifier.verify_circuit(
        board_type=payload.boardType or "ARDUINO_UNO",
        components=payload.components,
        wires=payload.wires,
        code=payload.code or ""
    )
    return {
        "suggestions": v_res["wireSuggestions"],
        "wireSuggestions": v_res["wireSuggestions"],
        "confidence": 0.9 if v_res["wireSuggestions"] else 0.7,
    }


@router.post("/review-code")
def review_code(payload: CodeReviewRequest) -> Dict[str, Any]:
    logger.info("Processing code review")
    v_res = ElectricalVerifier.verify_circuit(
        board_type=payload.boardType or "ARDUINO_UNO",
        components=payload.components,
        wires=payload.wires,
        code=payload.code or ""
    )
    return {
        "summary": f"Code review complete for {payload.boardType}. Evaluated against active hardware components.",
        "score": v_res["safetyScore"],
        "confidence": 0.88,
        "issues": v_res["issues"],
    }


@router.post("/schematic-to-code")
def schematic_to_code(payload: SchematicToCodeRequest) -> Dict[str, Any]:
    logger.info("Processing schematic-to-code")
    code = FirmwareCodeGenerator.generate(
        components=payload.components,
        wires=payload.wires,
        board_type=payload.boardType or "ARDUINO_UNO",
        additional_instructions=payload.additionalInstructions or ""
    )
    return {
        "status": "SUCCESS",
        "message": "Firmware generated from active canvas schematic.",
        "generatedCode": code,
        "confidence": 0.90,
    }


@router.post("/generate-code")
def generate_code(payload: GenerateCodeRequest) -> Dict[str, Any]:
    logger.info("Processing generate-code")
    code = FirmwareCodeGenerator.generate(
        components=payload.components,
        wires=payload.wires,
        board_type=payload.boardType or "ARDUINO_UNO",
        additional_instructions=payload.prompt or ""
    )
    return {
        "status": "SUCCESS",
        "message": "Firmware generated from Voltforge prompt.",
        "generatedCode": code,
        "confidence": 0.90,
    }
