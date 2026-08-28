from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8_000)

    @field_validator("content")
    @classmethod
    def non_blank_content(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("chat message cannot be blank")
        return cleaned


class FirmwareSource(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    language: str = Field(min_length=1, max_length=32)
    content: str = Field(max_length=200_000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=6_000)
    sessionId: Optional[str] = Field(default=None, min_length=1, max_length=80)
    projectId: Optional[str] = Field(default=None, min_length=1, max_length=80)
    projectRevision: Optional[str] = Field(default=None, min_length=1, max_length=160)
    context: Optional[str] = Field(default="", max_length=60_000)
    canvasContext: Optional[str] = Field(default="", max_length=60_000)
    boardType: Optional[str] = Field(default="ARDUINO_UNO", max_length=80)
    components: List[Dict[str, Any]] = Field(default_factory=list, max_length=500)
    wires: List[Dict[str, Any]] = Field(default_factory=list, max_length=1_000)
    netlist: Optional[Dict[str, Any]] = Field(default_factory=dict)
    code: Optional[str] = Field(default="", max_length=200_000)
    canvasData: Optional[Dict[str, Any]] = None
    simulationState: Optional[Dict[str, Any]] = None
    history: List[ChatMessage] = Field(default_factory=list, max_length=20)
    files: List[FirmwareSource] = Field(default_factory=list, max_length=20)

    @field_validator("message")
    @classmethod
    def non_blank_message(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("message cannot be blank")
        return cleaned

    @model_validator(mode="after")
    def bounded_conversation_context(self) -> "ChatRequest":
        if sum(len(item.content) for item in self.history) > 40_000:
            raise ValueError("chat history is too large")
        if sum(len(item.content) for item in self.files) > 400_000:
            raise ValueError("firmware context is too large")
        return self


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
    durationMs: int = Field(default=1000, ge=1, le=60_000)
    sampleRateHz: int = Field(default=100, ge=1, le=2_000)
