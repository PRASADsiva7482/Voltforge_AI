from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    context: Optional[str] = ""
    canvasContext: Optional[str] = ""
    boardType: Optional[str] = "ARDUINO_UNO"
    components: List[Dict[str, Any]] = Field(default_factory=list)
    wires: List[Dict[str, Any]] = Field(default_factory=list)
    netlist: Optional[Dict[str, Any]] = None
    code: Optional[str] = ""
    canvasData: Optional[Dict[str, Any]] = None
    simulationState: Optional[Dict[str, Any]] = None
    history: List[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str
    hasCode: bool = False
    generatedCode: Optional[str] = None
    confidence: float = 0.8
    citations: List[Dict[str, str]] = Field(default_factory=list)
    wireSuggestions: List[Dict[str, str]] = Field(default_factory=list)
    additions: List[Dict[str, Any]] = Field(default_factory=list)
    removals: List[Dict[str, Any]] = Field(default_factory=list)
    valueChanges: List[Dict[str, Any]] = Field(default_factory=list)
    codeFixes: List[Dict[str, Any]] = Field(default_factory=list)


class ValidateRequest(BaseModel):
    boardType: Optional[str] = "ARDUINO_UNO"
    components: List[Dict[str, Any]] = Field(default_factory=list)
    wires: List[Dict[str, Any]] = Field(default_factory=list)
    code: Optional[str] = ""
    context: Optional[str] = ""
    compilerDiagnostics: List[str] = Field(default_factory=list)
    simulationState: Optional[Dict[str, Any]] = None


class CodeReviewRequest(BaseModel):
    boardType: Optional[str] = "ARDUINO_UNO"
    code: str = ""
    componentTypes: List[str] = Field(default_factory=list)
    components: List[Dict[str, Any]] = Field(default_factory=list)
    wires: List[Dict[str, Any]] = Field(default_factory=list)
    circuitDescription: Optional[str] = ""
    compilerDiagnostics: List[str] = Field(default_factory=list)


class SchematicToCodeRequest(BaseModel):
    boardType: Optional[str] = "ARDUINO_UNO"
    components: List[Dict[str, Any]] = Field(default_factory=list)
    wires: List[Dict[str, Any]] = Field(default_factory=list)
    code: Optional[str] = ""
    additionalInstructions: Optional[str] = ""


class GenerateCodeRequest(BaseModel):
    boardType: Optional[str] = "ARDUINO_UNO"
    components: List[Dict[str, Any]] = Field(default_factory=list)
    wires: List[Dict[str, Any]] = Field(default_factory=list)
    code: Optional[str] = ""
    prompt: Optional[str] = ""
    componentTypes: List[str] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    userMessage: Optional[str] = ""
    aiResponse: Optional[str] = ""
    rating: int = 5
    comments: Optional[str] = ""
    sessionId: Optional[str] = None


class DatasheetSearchRequest(BaseModel):
    query: str
    limit: int = 5


class SimulationStreamRequest(BaseModel):
    boardType: Optional[str] = "ARDUINO_UNO"
    components: List[Dict[str, Any]] = Field(default_factory=list)
    wires: List[Dict[str, Any]] = Field(default_factory=list)
    probes: List[str] = Field(default_factory=list)
    durationMs: int = 1000
    sampleRateHz: int = 100

