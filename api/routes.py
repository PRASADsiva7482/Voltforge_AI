import asyncio
import logging
from pathlib import Path as FilePath
import time
import uuid
from typing import Annotated, Any, Dict, List, Literal, Optional
from fastapi import APIRouter, Depends, Header, HTTPException, Path, Query, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

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
from api.chat import _deterministic_response, stream_chat_sse
from api.copilot import prepare_grounded_context
from api.memory import (
    authenticated_memory_scope,
    memory_http_error,
    prepare_chat_memory,
)
from api.security import require_service_token
from api_contract import (
    CONTRACT_VERSION as API_CONTRACT_VERSION,
    ApiContractError,
    RequestStopped,
    artifact_identity,
    cancellation_registry,
    contract_health,
    error_response,
    wait_for_stage,
)
from api.sse import stream_simulation_sse
from api.database import DatabaseManager
from engineering_tools import (
    engineering_tools_health,
    run_authoritative_engineering_checks,
)
from local_retrieval import (
    RetrievalQuery,
    local_retrieval_health,
    search_curated_local,
)
from internet_retrieval import (
    InternetRetrievalQuery,
    internet_retrieval_health,
    search_internet,
)
from engine.code_generator import FirmwareCodeGenerator
from engine.export_engine import CircuitExportEngine
from engine.layout_optimizer import LayoutOptimizer
from engine.thermal_engine import ThermalEngine
from engine.circuit_synthesizer import CircuitSynthesizer
from engine.emi_engine import EmiRuleEngine
from engine.bom_sourcing import BomSourcingEngine
from engine.drc_engine import PcbDrcEngine
from engine.gerber_exporter import GerberExporter
from web_search_engine import WebSearchEngine
from context_compiler import context_compiler_health
from grounding import grounding_health
from memory_store import (
    MemoryCorrectionRequest,
    MemoryPreferenceRequest,
    MemoryStoreError,
    MemoryWriteRequest,
    memory_health,
)
from model.runtime_service import get_model_runtime_service
from model.generation_quality import (
    QUALITY_GATE_POLICY_ID,
    fallback_metadata,
    generation_quality_metrics,
)
from config import get_settings
from observability import observability
from feedback_governance import (
    FeedbackGovernanceError,
    feedback_health,
    get_feedback_store,
)
from hardware_coverage import get_component_coverage, get_hardware_coverage

logger = logging.getLogger("voltforge-ai.routes")

router = APIRouter(
    prefix="/voltForge-ai/api/v1/model",
    tags=["Voltforge AI"],
    dependencies=[Depends(require_service_token)],
)
db_manager = DatabaseManager()
model_runtime_service = get_model_runtime_service()


@router.get("/hardware-coverage")
def hardware_coverage() -> Dict[str, Any]:
    """Expose exact UI-board coverage without silently widening AI support."""

    return get_hardware_coverage()


@router.get("/component-coverage")
def component_coverage() -> Dict[str, Any]:
    """Expose UI component coverage without inventing exact electrical variants."""

    return get_component_coverage()


def _memory_health() -> Dict[str, Any]:
    settings = get_settings()
    return memory_health(
        settings.memory_database_path,
        authenticated_gateway_configured=bool(settings.api_token),
    )


def _feedback_database_path() -> FilePath:
    return FilePath(get_settings().runtime_directory) / "feedback-governance-v1" / "feedback.sqlite3"


def _feedback_health() -> Dict[str, Any]:
    return feedback_health(_feedback_database_path())


def _api_contract_health() -> Dict[str, Any]:
    settings = get_settings()
    return contract_health(
        routes=router.routes,
        model_health=model_runtime_service.health(),
        context_health=context_compiler_health(),
        engineering_health=engineering_tools_health(),
        local_retrieval_health=local_retrieval_health(),
        internet_health=internet_retrieval_health(settings),
        grounding_health=grounding_health(),
        memory_health=_memory_health(),
        feedback_health=_feedback_health(),
    )


def _chat_from_validate(payload: ValidateRequest) -> ChatRequest:
    return ChatRequest(
        message="Validate this circuit and project state.",
        boardType=payload.boardType or "ARDUINO_UNO",
        components=payload.components,
        wires=payload.wires,
        code=payload.code or "",
        diagnostics=[{"message": item} for item in payload.compilerDiagnostics[:50]],
        simulationState=payload.simulationState,
    )


def _compatibility_validation(report: Dict[str, Any]) -> Dict[str, Any]:
    findings = [
        finding
        for run in report.get("toolRuns", [])
        if isinstance(run, dict)
        for finding in run.get("findings", [])
        if isinstance(finding, dict)
    ]
    issues = [
        {
            "findingId": item.get("findingId"),
            "type": item.get("ruleId"),
            "severity": item.get("severity"),
            "message": item.get("summary"),
            "suggestedFix": (item.get("fix") or {}).get("summary"),
            "affectedProjectIds": item.get("affectedProjectIds") or [],
            "evidenceRefs": item.get("evidenceRefs") or [],
            "blocking": bool(item.get("blocking")),
            "modelOverridePolicy": item.get("modelOverridePolicy"),
        }
        for item in findings[:100]
    ]
    action_targets = {
        "wire-suggestion": "wireSuggestions",
        "component-addition": "additions",
        "component-removal": "removals",
        "value-change": "valueChanges",
        "code-fix": "codeFixes",
    }
    actions: Dict[str, List[Dict[str, Any]]] = {
        value: [] for value in action_targets.values()
    }
    for run in report.get("toolRuns", []):
        if not isinstance(run, dict):
            continue
        for action in run.get("approvedActions", []):
            if not isinstance(action, dict):
                continue
            target = action_targets.get(str(action.get("actionKind")))
            if target:
                actions[target].append(dict(action.get("payload") or {}))
    score = 100 - sum(
        {"CRITICAL": 30, "HIGH": 20, "WARNING": 10, "UNKNOWN": 5}.get(
            str(item.get("severity")), 0
        )
        for item in findings
    )
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    authority = {
        "policyId": report.get("policyId"),
        "policySha256": report.get("policySha256"),
        "reportId": report.get("reportId"),
        "sourceProjectRevision": report.get("sourceProjectRevision"),
        "status": summary.get("status"),
        "blockingFindingCount": summary.get("blockingFindings"),
        "criticalModelOverrideAllowed": False,
        "rawProjectContentStored": False,
    }
    return {
        "isValid": not bool(report.get("blockingFindingIds")),
        "safetyScore": max(0, min(100, score)),
        "generalFeedback": (
            f"Authoritative engineering checks completed with {len(findings)} finding(s) "
            f"and {len(report.get('blockingFindingIds') or [])} blocking finding(s)."
        ),
        "issues": issues,
        **{key: value[:25] for key, value in actions.items()},
        "engineeringAuthority": authority,
        "engineeringReport": report,
    }


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

    logger.info("Processing natural-language circuit synthesis request")
    return CircuitSynthesizer.synthesize_circuit(
        prompt=payload.message,
        preferred_board=payload.boardType or "ARDUINO_UNO"
    )


@router.post("/circuit/thermal-analysis")
def thermal_analysis_circuit(payload: ValidateRequest) -> Dict[str, Any]:

    logger.info("Computing circuit thermal and Joulean dissipation heatmap")
    return ThermalEngine.analyze_thermal_dissipation(
        components=payload.components,
        wires=payload.wires,
        simulation_state=payload.simulationState,
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
    settings = get_settings()
    local_model = model_runtime_service.health()
    return {
        "status": "UP",
        "service": "Voltforge AI Microservice",
        "engine": "Voltforge Electronics Reasoning Engine",
        "ragEnabled": True,
        "version": "2.0.0",
        "security": {
            "environment": settings.environment,
            "serviceAuthenticationRequired": settings.service_authentication_required,
            "artifactIntegrity": "signed-registry-and-checksum-verified",
            "rawPromptStorage": False,
            "rawModelOutputStorage": False,
            "arbitraryUrlFetchAllowed": False,
        },
        "localModel": local_model,
        "generation": {
            "mode": "deterministic-fallback",
            "networkRequired": False,
            "neuralOutputPolicy": "quality-gated-only",
            "qualityGatePolicyId": QUALITY_GATE_POLICY_ID,
            "fallbackSource": "voltforge-deterministic-tools",
        },
        "generationQuality": generation_quality_metrics.snapshot(),
        "contextCompiler": context_compiler_health(),
        "engineeringTools": engineering_tools_health(),
        "localRetrieval": local_retrieval_health(),
        "internetRetrieval": internet_retrieval_health(get_settings()),
        "grounding": grounding_health(),
        "memory": _memory_health(),
        "feedbackGovernance": _feedback_health(),
        "apiContract": _api_contract_health(),
        "observability": observability.health(),
    }


@router.get("/system/health")
def system_health() -> Dict[str, Any]:
    settings = get_settings()
    db_health = db_manager.check_health()
    local_model = model_runtime_service.health()
    return {
        "status": "ok",
        "service": "Voltforge AI Microservice",
        "version": "2.0.0",
        "security": {
            "environment": settings.environment,
            "serviceAuthenticationRequired": settings.service_authentication_required,
            "artifactIntegrity": "signed-registry-and-checksum-verified",
            "rawPromptStorage": False,
            "rawModelOutputStorage": False,
            "arbitraryUrlFetchAllowed": False,
        },
        "database": db_health,
        "localModel": local_model,
        "generation": {
            "mode": "deterministic-fallback",
            "networkRequired": False,
            "neuralOutputPolicy": "quality-gated-only",
            "qualityGatePolicyId": QUALITY_GATE_POLICY_ID,
            "fallbackSource": "voltforge-deterministic-tools",
        },
        "generationQuality": generation_quality_metrics.snapshot(),
        "contextCompiler": context_compiler_health(),
        "engineeringTools": engineering_tools_health(),
        "localRetrieval": local_retrieval_health(),
        "internetRetrieval": internet_retrieval_health(get_settings()),
        "grounding": grounding_health(),
        "memory": _memory_health(),
        "feedbackGovernance": _feedback_health(),
        "apiContract": _api_contract_health(),
        "observability": observability.health(),
        "features": {
            "circuitVerifier": True,
            "spiceEngine": True,
            "reasoningOrchestrator": True,
            "localKnowledgeRetrieval": True,
            "internetRetrievalEnabled": get_settings().internet_retrieval_enabled,
            "localModelReady": bool(local_model["ready"]),
            "authoritativeEngineeringTools": True,
            "claimLevelGrounding": grounding_health()["ready"],
            "boundedMemory": _memory_health()["ready"],
            "versionedAiContract": True,
        }
    }


@router.get("/contract")
def api_contract_status() -> Dict[str, Any]:
    """Expose content-free route readiness and the active compatibility contract."""
    return _api_contract_health()


@router.get("/metrics")
def metrics_status() -> Dict[str, Any]:
    """Expose content-free process metrics for the private service operator."""
    return observability.snapshot()


@router.post(
    "/chat",
    response_model=ChatResponse,
    dependencies=[Depends(require_service_token)],
)
async def chat_http(
    payload: ChatRequest,
    request: Request,
    x_voltforge_user_id: Annotated[
        str | None, Header(alias="X-Voltforge-User-Id")
    ] = None,
    x_voltforge_project_id: Annotated[
        str | None, Header(alias="X-Voltforge-Project-Id")
    ] = None,
    x_voltforge_session_id: Annotated[
        str | None, Header(alias="X-Voltforge-Session-Id")
    ] = None,
    x_voltforge_request_id: Annotated[
        str | None,
        Header(
            alias="X-Voltforge-Request-Id",
            min_length=3,
            max_length=160,
            pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{2,159}$",
        ),
    ] = None,
) -> ChatResponse | JSONResponse:
    logger.info("Processing local chat request")
    settings = get_settings()
    request_id = x_voltforge_request_id or getattr(
        request.state, "voltforge_request_id", None
    ) or str(uuid.uuid4())
    session_id = payload.sessionId or x_voltforge_session_id or str(uuid.uuid4())
    runtime_payload, binding, memory_metadata = prepare_chat_memory(
        payload,
        settings,
        user_id=x_voltforge_user_id,
        project_id=x_voltforge_project_id,
        session_id=x_voltforge_session_id,
    )
    try:
        token = cancellation_registry.register(request_id)
    except ApiContractError:
        return JSONResponse(
            status_code=409,
            content=error_response(
                "API_REQUEST_ID_CONFLICT",
                "The request identifier is already active.",
                retryable=True,
                request_id=request_id,
            ),
        )
    try:
        deadline = (
            asyncio.get_running_loop().time()
            + settings.chat_request_timeout_seconds
        )

        def remaining_deadline() -> float:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                token.cancel()
                raise RequestStopped(
                    "REQUEST_TIMEOUT",
                    "The local AI request exceeded its deadline.",
                )
            return remaining

        stage_started = time.perf_counter()
        try:
            grounded = await wait_for_stage(
                lambda: prepare_grounded_context(
                    runtime_payload, settings, check_cancelled=token.check
                ),
                token=token,
                disconnected=request.is_disconnected,
                deadline_seconds=remaining_deadline(),
            )
        finally:
            observability.record_stage(
                "grounding", (time.perf_counter() - stage_started) * 1000
            )
        def deterministic_stage() -> ChatResponse:
            token.check()
            result = _deterministic_response(
                runtime_payload,
                settings.internet_retrieval_enabled,
                grounded.engineering_report,
                grounded.retrieval_report,
                grounded.internet_retrieval_report,
                grounded.task_record,
                list(grounded.response_metadata.get("citations") or []),
            )
            token.check()
            return result

        stage_started = time.perf_counter()
        try:
            response = await wait_for_stage(
                deterministic_stage,
                token=token,
                disconnected=request.is_disconnected,
                deadline_seconds=remaining_deadline(),
            )
        finally:
            observability.record_stage(
                "generation", (time.perf_counter() - stage_started) * 1000
            )
        token.check()
        if not isinstance(response, ChatResponse) or not response.reply.strip():
            return JSONResponse(
                status_code=502,
                content=error_response(
                    "MALFORMED_GENERATION",
                    "The local generation result did not match the response contract.",
                    retryable=True,
                    request_id=request_id,
                ),
            )
        if binding is not None:
            await wait_for_stage(
                lambda: binding.record_turn(payload.message, response.reply),
                token=token,
                disconnected=request.is_disconnected,
                deadline_seconds=remaining_deadline(),
            )
            memory_metadata = await wait_for_stage(
                binding.refreshed_metadata,
                token=token,
                disconnected=request.is_disconnected,
                deadline_seconds=remaining_deadline(),
            )
        source_revision = str(
            grounded.task_record["input"]["projectContext"]["sourceProjectRevision"]
        )
        model_health = model_runtime_service.health()
        generation = fallback_metadata(
            model_health,
            context_compiler_ready=(
                grounded.context_metadata.get("status") == "compiled"
            ),
        )
        observability.record_generation(
            mode=generation.get("mode"),
            fallback_used=bool(generation.get("fallbackUsed")),
            fallback_reason=generation.get("fallbackReasonCode"),
            neural_attempted=bool(generation.get("neuralAttempted")),
            tool_grounded=bool(grounded.tool_events),
            artifact_id=model_health.get("activeArtifactId") or model_health.get("artifactId"),
            input_tokens=grounded.context_metadata.get("promptTokens"),
        )
        return response.model_copy(
            update={
                "requestId": request_id,
                "sessionId": session_id,
                "projectRevision": source_revision,
                "mode": generation["mode"],
                "artifact": artifact_identity(model_health).model_dump(mode="json"),
                "readiness": _api_contract_health()["subsystems"],
                "memory": memory_metadata,
            }
        )
    except RequestStopped as error:
        return JSONResponse(
            status_code=408 if error.code == "REQUEST_TIMEOUT" else 409,
            content=error_response(
                error.code,
                error.message,
                retryable=error.code == "REQUEST_TIMEOUT",
                request_id=request_id,
            ),
        )
    finally:
        cancellation_registry.release(request_id)


def chat(payload: ChatRequest) -> ChatResponse:
    """In-process compatibility helper; HTTP callers use ``chat_http`` above."""
    settings = get_settings()
    runtime_payload, _binding, memory_metadata = prepare_chat_memory(
        payload,
        settings,
        user_id=None,
        project_id=None,
        session_id=payload.sessionId,
    )
    grounded = prepare_grounded_context(runtime_payload, settings)
    response = _deterministic_response(
        runtime_payload,
        settings.internet_retrieval_enabled,
        grounded.engineering_report,
        grounded.retrieval_report,
        grounded.internet_retrieval_report,
        grounded.task_record,
        list(grounded.response_metadata.get("citations") or []),
    )
    model_health = model_runtime_service.health()
    generation = fallback_metadata(
        model_health,
        context_compiler_ready=grounded.context_metadata.get("status") == "compiled",
    )
    return response.model_copy(
        update={
            "requestId": str(uuid.uuid4()),
            "sessionId": payload.sessionId or str(uuid.uuid4()),
            "projectRevision": grounded.task_record["input"]["projectContext"][
                "sourceProjectRevision"
            ],
            "mode": generation["mode"],
            "artifact": artifact_identity(model_health).model_dump(mode="json"),
            "readiness": _api_contract_health()["subsystems"],
            "memory": memory_metadata,
        }
    )



@router.post("/chat/stream", dependencies=[Depends(require_service_token)])
async def chat_stream(
    payload: ChatRequest,
    request: Request,
    x_voltforge_user_id: Annotated[
        str | None, Header(alias="X-Voltforge-User-Id")
    ] = None,
    x_voltforge_project_id: Annotated[
        str | None, Header(alias="X-Voltforge-Project-Id")
    ] = None,
    x_voltforge_session_id: Annotated[
        str | None, Header(alias="X-Voltforge-Session-Id")
    ] = None,
    x_voltforge_request_id: Annotated[
        str | None,
        Header(
            alias="X-Voltforge-Request-Id",
            min_length=3,
            max_length=160,
            pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{2,159}$",
        ),
    ] = None,
):
    logger.info("Processing streaming chat request")
    settings = get_settings()
    request_id = x_voltforge_request_id or getattr(
        request.state, "voltforge_request_id", None
    ) or str(uuid.uuid4())
    session_id = payload.sessionId or x_voltforge_session_id or str(uuid.uuid4())
    runtime_payload, binding, memory_metadata = prepare_chat_memory(
        payload,
        settings,
        user_id=x_voltforge_user_id,
        project_id=x_voltforge_project_id,
        session_id=x_voltforge_session_id,
    )
    try:
        token = cancellation_registry.register(request_id)
    except ApiContractError:
        return JSONResponse(
            status_code=409,
            content=error_response(
                "API_REQUEST_ID_CONFLICT",
                "The request identifier is already active.",
                retryable=True,
                request_id=request_id,
            ),
        )
    return StreamingResponse(
        stream_chat_sse(
            runtime_payload,
            settings,
            memory_binding=binding,
            memory_metadata=memory_metadata,
            request_id=request_id,
            session_id=session_id,
            cancellation=token,
            disconnected=request.is_disconnected,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Voltforge-Request-Id": request_id,
            "X-Voltforge-Contract-Version": API_CONTRACT_VERSION,
        },
    )


@router.delete(
    "/chat/requests/{request_id}",
    dependencies=[Depends(require_service_token)],
)
def cancel_chat_request(
    request_id: Annotated[
        str,
        Path(
            min_length=3,
            max_length=160,
            pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{2,159}$",
        ),
    ],
) -> Response:
    if not cancellation_registry.cancel(request_id):
        return JSONResponse(
            status_code=404,
            content=error_response(
                "REQUEST_NOT_ACTIVE",
                "The local AI request is not active.",
                retryable=False,
                request_id=request_id,
            ),
        )
    return JSONResponse(
        status_code=202,
        content={
            "schemaVersion": 1,
            "contractVersion": API_CONTRACT_VERSION,
            "requestId": request_id,
            "status": "cancellation-requested",
        },
    )


@router.get("/memory", dependencies=[Depends(require_service_token)])
def inspect_memory(
    project_revision: Annotated[str | None, Query(alias="projectRevision")] = None,
    x_voltforge_user_id: Annotated[
        str | None, Header(alias="X-Voltforge-User-Id")
    ] = None,
    x_voltforge_project_id: Annotated[
        str | None, Header(alias="X-Voltforge-Project-Id")
    ] = None,
    x_voltforge_session_id: Annotated[
        str | None, Header(alias="X-Voltforge-Session-Id")
    ] = None,
) -> Dict[str, Any]:
    store, scope = authenticated_memory_scope(
        get_settings(),
        user_id=x_voltforge_user_id,
        project_id=x_voltforge_project_id,
        session_id=x_voltforge_session_id,
    )
    return store.inspect(scope, project_revision).model_dump(mode="json")


@router.put("/memory/preferences", dependencies=[Depends(require_service_token)])
def set_memory_preference(
    payload: MemoryPreferenceRequest,
    x_voltforge_user_id: Annotated[
        str | None, Header(alias="X-Voltforge-User-Id")
    ] = None,
    x_voltforge_project_id: Annotated[
        str | None, Header(alias="X-Voltforge-Project-Id")
    ] = None,
    x_voltforge_session_id: Annotated[
        str | None, Header(alias="X-Voltforge-Session-Id")
    ] = None,
) -> Dict[str, Any]:
    store, scope = authenticated_memory_scope(
        get_settings(),
        user_id=x_voltforge_user_id,
        project_id=x_voltforge_project_id,
        session_id=x_voltforge_session_id,
    )
    return store.set_enabled(
        scope, payload.enabled, clear_on_disable=payload.clearOnDisable
    ).model_dump(mode="json")


@router.post("/memory/entries", dependencies=[Depends(require_service_token)])
def create_memory_entry(
    payload: MemoryWriteRequest,
    x_voltforge_user_id: Annotated[
        str | None, Header(alias="X-Voltforge-User-Id")
    ] = None,
    x_voltforge_project_id: Annotated[
        str | None, Header(alias="X-Voltforge-Project-Id")
    ] = None,
    x_voltforge_session_id: Annotated[
        str | None, Header(alias="X-Voltforge-Session-Id")
    ] = None,
) -> Dict[str, Any]:
    store, scope = authenticated_memory_scope(
        get_settings(),
        user_id=x_voltforge_user_id,
        project_id=x_voltforge_project_id,
        session_id=x_voltforge_session_id,
    )
    try:
        entry, evicted = store.create(scope, payload)
    except MemoryStoreError as error:
        raise memory_http_error(error) from error
    state = store.inspect(scope, payload.projectRevision, evicted_count=evicted)
    return {"entry": entry.model_dump(mode="json"), "memory": state.model_dump(mode="json")}


@router.patch(
    "/memory/entries/{memory_id}", dependencies=[Depends(require_service_token)]
)
def correct_memory_entry(
    memory_id: str,
    payload: MemoryCorrectionRequest,
    x_voltforge_user_id: Annotated[
        str | None, Header(alias="X-Voltforge-User-Id")
    ] = None,
    x_voltforge_project_id: Annotated[
        str | None, Header(alias="X-Voltforge-Project-Id")
    ] = None,
    x_voltforge_session_id: Annotated[
        str | None, Header(alias="X-Voltforge-Session-Id")
    ] = None,
) -> Dict[str, Any]:
    store, scope = authenticated_memory_scope(
        get_settings(),
        user_id=x_voltforge_user_id,
        project_id=x_voltforge_project_id,
        session_id=x_voltforge_session_id,
    )
    try:
        entry = store.correct(scope, memory_id, payload)
    except MemoryStoreError as error:
        raise memory_http_error(error) from error
    return entry.model_dump(mode="json")


@router.delete(
    "/memory/entries/{memory_id}", dependencies=[Depends(require_service_token)]
)
def delete_memory_entry(
    memory_id: str,
    x_voltforge_user_id: Annotated[
        str | None, Header(alias="X-Voltforge-User-Id")
    ] = None,
    x_voltforge_project_id: Annotated[
        str | None, Header(alias="X-Voltforge-Project-Id")
    ] = None,
    x_voltforge_session_id: Annotated[
        str | None, Header(alias="X-Voltforge-Session-Id")
    ] = None,
) -> Dict[str, Any]:
    store, scope = authenticated_memory_scope(
        get_settings(),
        user_id=x_voltforge_user_id,
        project_id=x_voltforge_project_id,
        session_id=x_voltforge_session_id,
    )
    deleted = store.delete(scope, memory_id)
    if not deleted:
        raise memory_http_error(
            MemoryStoreError("MEMORY_NOT_FOUND", "The memory entry was not found.")
        )
    return {"deleted": True, "memoryId": memory_id}


@router.delete("/memory", dependencies=[Depends(require_service_token)])
def clear_memory(
    scope_name: Annotated[
        Literal["project", "session"], Query(alias="scope")
    ] = "session",
    x_voltforge_user_id: Annotated[
        str | None, Header(alias="X-Voltforge-User-Id")
    ] = None,
    x_voltforge_project_id: Annotated[
        str | None, Header(alias="X-Voltforge-Project-Id")
    ] = None,
    x_voltforge_session_id: Annotated[
        str | None, Header(alias="X-Voltforge-Session-Id")
    ] = None,
) -> Dict[str, Any]:
    store, memory_scope = authenticated_memory_scope(
        get_settings(),
        user_id=x_voltforge_user_id,
        project_id=x_voltforge_project_id,
        session_id=x_voltforge_session_id,
    )
    try:
        deleted = store.clear(memory_scope, scope_name)
    except MemoryStoreError as error:
        raise memory_http_error(error) from error
    return {"cleared": True, "scope": scope_name, "deletedCount": deleted}


@router.post("/validate-circuit")
def validate_circuit(payload: ValidateRequest) -> Dict[str, Any]:
    logger.info("Processing circuit validation")
    report = run_authoritative_engineering_checks(_chat_from_validate(payload))
    return _compatibility_validation(report.model_dump(mode="json"))


@router.post("/engineering/check")
def engineering_check(payload: ChatRequest) -> Dict[str, Any]:
    """Return the complete typed authoritative report for one project revision."""

    logger.info("Processing authoritative engineering checks")
    return run_authoritative_engineering_checks(payload).model_dump(mode="json")


@router.post("/retrieval/search")
def local_retrieval_search(payload: RetrievalQuery) -> Dict[str, Any]:
    """Return bounded chunks from the approved, offline lexical index."""

    logger.info("Processing curated local retrieval request")
    return search_curated_local(payload).model_dump(mode="json")


@router.post(
    "/internet/search",
    dependencies=[Depends(require_service_token)],
)
def internet_evidence_search(payload: InternetRetrievalQuery) -> Dict[str, Any]:
    """Search the one approved provider; request bodies cannot supply a URL."""

    logger.info("Processing optional internet evidence request")
    return search_internet(payload, get_settings()).model_dump(mode="json")


@router.post("/suggest-wiring")
def suggest_wiring(payload: ValidateRequest) -> Dict[str, Any]:
    logger.info("Processing wiring suggestions")
    result = _compatibility_validation(
        run_authoritative_engineering_checks(_chat_from_validate(payload)).model_dump(
            mode="json"
        )
    )
    return {
        "suggestions": result["wireSuggestions"],
        "wireSuggestions": result["wireSuggestions"],
        "confidence": 0.95 if result["wireSuggestions"] else 0.0,
        "engineeringAuthority": result["engineeringAuthority"],
    }


@router.post("/review-code")
def review_code(payload: CodeReviewRequest) -> Dict[str, Any]:
    logger.info("Processing code review")
    request = ChatRequest(
        message="Review this firmware against the supplied project.",
        boardType=payload.boardType or "ARDUINO_UNO",
        components=payload.components,
        wires=payload.wires,
        code=payload.code,
        diagnostics=[
            {"message": item} for item in payload.compilerDiagnostics[:50]
        ],
    )
    result = _compatibility_validation(
        run_authoritative_engineering_checks(request).model_dump(mode="json")
    )
    return {
        "summary": f"Code review complete for {payload.boardType}. Evaluated against active hardware components.",
        "score": result["safetyScore"],
        "confidence": 1.0,
        "issues": result["issues"],
        "engineeringAuthority": result["engineeringAuthority"],
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


@router.post("/datasheet/search", dependencies=[Depends(require_service_token)])
def search_datasheet(payload: DatasheetSearchRequest) -> Dict[str, Any]:
    logger.info("Processing datasheet search request")
    results = WebSearchEngine().get_component_info(payload.query)
    return {
        "query": payload.query,
        "results": results.get("searchResults", []),
        "specs": results.get("specs", {})
    }


@router.get("/datasheet/{part_number}", dependencies=[Depends(require_service_token)])
def get_datasheet_details(part_number: str) -> Dict[str, Any]:
    logger.info("Processing datasheet detail request")
    details = WebSearchEngine().get_component_info(part_number)
    return {
        "partNumber": part_number,
        "details": details
    }



@router.post("/feedback", status_code=202)
def submit_feedback(
    payload: FeedbackRequest,
    x_voltforge_user_id: Annotated[
        str | None, Header(alias="X-Voltforge-User-Id")
    ] = None,
    x_voltforge_project_id: Annotated[
        str | None, Header(alias="X-Voltforge-Project-Id")
    ] = None,
) -> Dict[str, Any]:
    """Admit bounded project feedback into the review queue only."""

    if not x_voltforge_user_id or not x_voltforge_project_id:
        raise HTTPException(
            status_code=401,
            detail={
                "code": "FEEDBACK_PROJECT_ACCESS_REQUIRED",
                "message": "Authenticated user and project access are required for feedback.",
            },
        )
    if payload.projectId != x_voltforge_project_id:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "FEEDBACK_PROJECT_SCOPE_MISMATCH",
                "message": "Feedback project does not match the authenticated project scope.",
            },
        )
    try:
        result = get_feedback_store(_feedback_database_path()).submit(
            user_id=x_voltforge_user_id,
            project_id=x_voltforge_project_id,
            project_revision=payload.projectRevision,
            request_id=payload.requestId,
            response_record_id=payload.responseRecordId,
            artifact_id=payload.artifactId,
            registry_revision=payload.registryRevision,
            feedback_kind=payload.feedbackKind,
            rating=payload.rating,
            evidence=payload.evidence,
            expected_behavior=payload.expectedBehavior,
            evidence_approved=payload.evidenceApproved,
            training_consent=payload.trainingConsent,
        )
    except FeedbackGovernanceError as error:
        status_code = 409 if error.code in {"FEEDBACK_DUPLICATE", "FEEDBACK_VERSION_CONFLICT"} else 422
        raise HTTPException(status_code=status_code, detail={"code": error.code, "message": error.message}) from error
    logger.info(
        "Feedback admitted for governed review (feedbackId=%s kind=%s rating=%s)",
        result["feedbackId"],
        result["feedbackKind"],
        result["rating"],
    )
    return result


@router.get("/database/health")
def db_health() -> Dict[str, Any]:
    return db_manager.check_health()


@router.post("/circuit/drc-check")
def run_pcb_drc_check(payload: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("Executing PCB Design Rule Check (DRC)")
    board_width = float(payload.get("boardWidth_mm", 100))
    board_height = float(payload.get("boardHeight_mm", 80))
    footprints = payload.get("footprints", [])
    traces = payload.get("traces", [])
    vias = payload.get("vias", [])
    wires = payload.get("wires", [])

    return PcbDrcEngine.run_drc(
        board_width_mm=board_width,
        board_height_mm=board_height,
        footprints=footprints,
        traces=traces,
        vias=vias,
        wires=wires,
    )


@router.post("/circuit/export-gerber")
def export_pcb_gerber(payload: Dict[str, Any]):
    logger.info("Generating RS-274X Gerber & Excellon drill archive")
    board_width = float(payload.get("boardWidth_mm", 100))
    board_height = float(payload.get("boardHeight_mm", 80))
    footprints = payload.get("footprints", [])
    traces = payload.get("traces", [])
    vias = payload.get("vias", [])
    project_name = payload.get("projectName", "VoltForge_PCB")

    zip_data = GerberExporter.generate_gerber_zip(
        board_width_mm=board_width,
        board_height_mm=board_height,
        footprints=footprints,
        traces=traces,
        vias=vias,
        project_name=project_name,
    )

    return Response(
        content=zip_data,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={project_name}_gerber.zip"}
    )
