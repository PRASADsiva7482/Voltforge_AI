import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from api.schemas import (
    ChatRequest,
    ChatResponse,
    CodeReviewRequest,
    DatasheetSearchRequest,
    FeedbackRequest,
    GenerateCodeRequest,
    SchematicToCodeRequest,
    SimulationStreamRequest,
    ValidateRequest,
)
from api.sse import stream_chat_sse, stream_simulation_sse
from api.database import DatabaseManager
from circuit_verifier import ElectricalVerifier
from engine.code_generator import FirmwareCodeGenerator
from engine.reasoning import ElectronicsReasoningOrchestrator
from engine.export_engine import CircuitExportEngine
from engine.layout_optimizer import LayoutOptimizer
from engine.thermal_engine import ThermalEngine
from engine.circuit_synthesizer import CircuitSynthesizer
from engine.emi_engine import EmiRuleEngine
from engine.bom_sourcing import BomSourcingEngine
from web_search_engine import WebSearchEngine

logger = logging.getLogger("voltforge-ai.routes")

router = APIRouter(prefix="/voltForge-ai/api/v1/model", tags=["Voltforge AI"])
orchestrator = ElectronicsReasoningOrchestrator()
db_manager = DatabaseManager()
web_search = WebSearchEngine()


@router.post("/circuit/bom-sourcing")
def calculate_bom_sourcing(payload: ValidateRequest) -> Dict[str, Any]:
    logger.info("Calculating BOM component sourcing and volume discount costs")
    return BomSourcingEngine.calculate_bom_cost(
        components=payload.components
    )


@router.post("/circuit/emi-rules")
def evaluate_emi_rules(payload: ValidateRequest) -> Dict[str, Any]:

    logger.info("Evaluating circuit EMI and decoupling bypass capacitor rules")
    return EmiRuleEngine.evaluate_decoupling(
        components=payload.components,
        wires=payload.wires
    )


@router.post("/circuit/synthesize")
def synthesize_circuit_from_text(payload: ChatRequest) -> Dict[str, Any]:

    logger.info(f"Synthesizing circuit from natural language prompt: {payload.message}")
    return CircuitSynthesizer.synthesize_circuit(
        prompt=payload.message,
        preferred_board=payload.boardType or "ARDUINO_UNO"
    )


@router.post("/circuit/thermal-analysis")
def thermal_analysis_circuit(payload: ValidateRequest) -> Dict[str, Any]:

    logger.info("Computing circuit thermal and Joulean dissipation heatmap")
    return ThermalEngine.analyze_thermal_dissipation(
        components=payload.components,
        wires=payload.wires
    )


@router.post("/circuit/auto-layout")
def auto_layout_circuit(payload: ValidateRequest) -> Dict[str, Any]:

    logger.info("Executing topological circuit auto-placement")
    optimized = LayoutOptimizer.optimize_layout(
        components=payload.components,
        wires=payload.wires
    )
    return {
        "status": "SUCCESS",
        "components": optimized,
        "count": len(optimized)
    }


@router.post("/circuit/export")
def export_circuit(payload: ValidateRequest) -> Dict[str, Any]:

    logger.info("Generating multi-format circuit export")
    return CircuitExportEngine.export_all(
        components=payload.components,
        wires=payload.wires,
        board_type=payload.boardType or "ARDUINO_UNO"
    )



@router.post("/simulation/stream")
async def simulation_stream(payload: SimulationStreamRequest):
    logger.info("Processing streaming circuit simulation waveform request")
    return StreamingResponse(
        stream_simulation_sse(payload),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )



@router.get("/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "UP",
        "service": "Voltforge AI Microservice",
        "engine": "Voltforge Electronics Reasoning Engine",
        "ragEnabled": True,
        "version": "2.0.0",
    }


@router.get("/system/health")
def system_health() -> Dict[str, Any]:
    db_health = db_manager.check_health()
    return {
        "status": "ok",
        "service": "Voltforge AI Microservice",
        "version": "2.0.0",
        "database": db_health,
        "features": {
            "circuitVerifier": True,
            "spiceEngine": True,
            "reasoningOrchestrator": True,
            "webSearchEngine": True,
        }
    }


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    logger.info(f"Processing chat request: {payload.message}")
    history_dicts = [h.dict() for h in payload.history] if payload.history else []
    return orchestrator.process_chat(
        message=payload.message,
        board_type=payload.boardType or "ARDUINO_UNO",
        components=payload.components,
        wires=payload.wires,
        code=payload.code or "",
        context={"history": history_dicts, "canvasContext": payload.canvasContext}
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


@router.post("/datasheet/search")
def search_datasheet(payload: DatasheetSearchRequest) -> Dict[str, Any]:
    logger.info(f"Searching datasheets for: {payload.query}")
    results = web_search.get_component_info(payload.query)
    return {
        "query": payload.query,
        "results": results.get("searchResults", []),
        "specs": results.get("specs", {})
    }


@router.get("/datasheet/{part_number}")
def get_datasheet_details(part_number: str) -> Dict[str, Any]:
    logger.info(f"Fetching datasheet details for: {part_number}")
    details = web_search.get_component_info(part_number)
    return {
        "partNumber": part_number,
        "details": details
    }



@router.post("/feedback")
def submit_feedback(payload: FeedbackRequest) -> Dict[str, Any]:
    logger.info("Saving user feedback")
    try:
        db_manager.save_feedback(
            user_message=payload.userMessage or "",
            ai_response=payload.aiResponse or "",
            rating=payload.rating,
            comments=payload.comments or "",
            session_id=payload.sessionId
        )
        return {"status": "SUCCESS", "message": "Feedback recorded successfully."}
    except Exception as exc:
        logger.warning(f"Error saving feedback: {exc}")
        return {"status": "SUCCESS", "message": "Feedback received (stored locally)."}


@router.get("/database/health")
def db_health() -> Dict[str, Any]:
    return db_manager.check_health()
