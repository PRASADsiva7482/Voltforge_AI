"""
Pydantic Request and Response schemas for VoltForge AI microservice.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    message: str
    sessionId: Optional[str] = None
    projectId: Optional[str] = None
    projectRevision: Optional[str] = None
    boardType: Optional[str] = "ARDUINO_UNO"
    components: List[Dict[str, Any]] = Field(default_factory=list)
    wires: List[Dict[str, Any]] = Field(default_factory=list)
    code: Optional[str] = ""
    history: List[ChatMessage] = Field(default_factory=list)
    simulationState: Optional[Dict[str, Any]] = None
    context: Optional[str] = ""

class ChatResponse(BaseModel):
    schemaVersion: int = 1
    contractVersion: str = "1.0.0"
    requestId: Optional[str] = None
    sessionId: Optional[str] = None
    projectRevision: Optional[str] = None
    model: str = "Qwen-2.5-Coder-1.5B"
    mode: str = "neural"
    reply: str
    hasCode: bool = False
    generatedCode: Optional[str] = None
    confidence: float = 0.98
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    wireSuggestions: List[Dict[str, Any]] = Field(default_factory=list)
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

class GenerateCodeRequest(BaseModel):
    boardType: Optional[str] = "ARDUINO_UNO"
    components: List[Dict[str, Any]] = Field(default_factory=list)
    wires: List[Dict[str, Any]] = Field(default_factory=list)
    code: Optional[str] = ""
    prompt: Optional[str] = ""
    componentTypes: List[str] = Field(default_factory=list)

class AiGenerateResponse(BaseModel):
    status: str = "SUCCESS"
    message: str = "Operation completed successfully."
    generatedCode: Optional[str] = None
    wireSuggestions: List[Dict[str, Any]] = Field(default_factory=list)
    additions: List[Dict[str, Any]] = Field(default_factory=list)
    removals: List[Dict[str, Any]] = Field(default_factory=list)
    codeFixes: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: float = 0.98

class AiValidationResponse(BaseModel):
    isValid: bool = True
    safetyScore: int = 100
    generalFeedback: str = "Circuit analysis complete."
    issues: List[Dict[str, Any]] = Field(default_factory=list)
    suggestedFixes: List[str] = Field(default_factory=list)
