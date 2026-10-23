"""
Pydantic Request and Response schemas for VoltForge AI microservice.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from api.research import ResearchRequest, ResearchSource

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=12000)

class ChatRequest(BaseModel):
    research: ResearchRequest = Field(default_factory=ResearchRequest)
    # Bound by the authenticated gateway to an existing conversation attempt.
    attemptId: Optional[str] = Field(default=None, max_length=80)
    # Private service-authenticated BL transport only; browser values are ignored by BL.
    authorizedDocumentEvidence: List[ResearchSource] = Field(default_factory=list, max_length=5)

    @model_validator(mode='after')
    def document_scope(self):
        selected = set(self.research.documentIds)
        for source in self.authorizedDocumentEvidence:
            if source.kind != 'document_chunk' or source.documentId not in selected or not self.attemptId:
                raise ValueError('Document evidence does not match the authorized selection/attempt')
        return self

    @model_validator(mode='before')
    @classmethod
    def reject_external_evidence(cls, value):
        if isinstance(value, dict) and value.get('retrievedEvidence'):
            raise ValueError('Retrieval evidence must be produced by the retrieval service')
        return value

    message: str = Field(min_length=1, max_length=6000)
    sessionId: Optional[str] = None
    projectId: Optional[str] = None
    projectRevision: Optional[str] = None
    boardType: Optional[str] = None
    components: List[Dict[str, Any]] = Field(default_factory=list)
    wires: List[Dict[str, Any]] = Field(default_factory=list)
    code: Optional[str] = ""
    history: List[ChatMessage] = Field(default_factory=list, max_length=20)
    simulationState: Optional[Dict[str, Any]] = None
    context: Optional[str] = ""
    # Only the private authenticated gateway supplies stored, allowlisted preferences.
    approvedUserPreferences: Dict[str, str] = Field(default_factory=dict, max_length=3)

    @field_validator('message')
    @classmethod
    def nonblank_message(cls, value):
        if not value.strip():
            raise ValueError('Message must not be empty')
        return value.strip()

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
    confidence: Optional[float] = None
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    wireSuggestions: List[Dict[str, Any]] = Field(default_factory=list)
    additions: List[Dict[str, Any]] = Field(default_factory=list)
    removals: List[Dict[str, Any]] = Field(default_factory=list)
    valueChanges: List[Dict[str, Any]] = Field(default_factory=list)
    codeFixes: List[Dict[str, Any]] = Field(default_factory=list)

class ValidateRequest(BaseModel):
    boardType: Optional[str] = None
    components: List[Dict[str, Any]] = Field(default_factory=list)
    wires: List[Dict[str, Any]] = Field(default_factory=list)
    code: Optional[str] = ""
    context: Optional[str] = ""
    compilerDiagnostics: List[str] = Field(default_factory=list)
    simulationState: Optional[Dict[str, Any]] = None

class GenerateCodeRequest(BaseModel):
    boardType: Optional[str] = None
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
    confidence: Optional[float] = None

class AiValidationResponse(BaseModel):
    isValid: bool = True
    safetyScore: int = 100
    generalFeedback: str = "Circuit analysis complete."
    issues: List[Dict[str, Any]] = Field(default_factory=list)
    suggestedFixes: List[str] = Field(default_factory=list)
