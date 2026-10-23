"""Validated research contract and capability reporting for web/document adapters."""
from datetime import datetime
from typing import Literal

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, StrictBool, model_validator
from config import get_settings


class ResearchRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    mode: Literal['off', 'auto', 'on'] = 'off'
    # Deliberately separate from message/history/project context.
    publicQuery: str | None = Field(default=None, max_length=500)
    consentToExternalQuery: StrictBool = False
    documentIds: list[str] = Field(default_factory=list, max_length=10)

    @model_validator(mode='after')
    def validate_consent(self):
        if self.publicQuery is not None:
            self.publicQuery = self.publicQuery.strip()
        if self.mode == 'off':
            if self.publicQuery or self.consentToExternalQuery:
                raise ValueError('Research is off; remove the external query and consent')
        elif not self.publicQuery or not self.consentToExternalQuery:
            raise ValueError('Review a public query and explicitly permit sending it before web research')
        if self.publicQuery and len(self.publicQuery.split()) > 75:
            raise ValueError('Keep the public search query within 75 words')
        if any(not item.strip() or len(item) > 80 for item in self.documentIds):
            raise ValueError('Invalid document reference')
        return self


class ResearchSource(BaseModel):
    """Server-produced provenance, never an accepted browser evidence object."""
    model_config = ConfigDict(extra='forbid')
    sourceId: str = Field(min_length=1, max_length=80)
    kind: Literal['search_snippet', 'web_page', 'document_chunk']
    title: str = Field(min_length=1, max_length=300)
    url: HttpUrl | None = None
    documentId: str | None = Field(default=None, max_length=80)
    versionId: str | None = Field(default=None, max_length=80)
    retrievedAt: datetime
    excerpt: str = Field(min_length=1, max_length=8000)
    contentHash: str = Field(pattern=r'^[a-f0-9]{64}$')
    page: int | None = Field(default=None, ge=1)
    section: str | None = Field(default=None, max_length=300)

    @model_validator(mode='after')
    def validate_origin(self):
        if self.kind == 'document_chunk':
            if not self.documentId or not self.versionId or self.page is None or self.url is not None:
                raise ValueError('Document evidence requires an authorized document reference')
        elif self.url is None or self.url.scheme != 'https' or self.url.username or self.url.password or self.documentId:
            raise ValueError('Web evidence requires a public HTTPS source URL without credentials')
        if self.retrievedAt.tzinfo is None:
            raise ValueError('Retrieval time must include a timezone')
        return self


def research_capabilities():
    settings = get_settings()
    configured = bool(settings.brave_api_key.get_secret_value().strip())
    available = settings.web_search_enabled and configured
    return {
        'contractVersion': '1.0',
        'web': {'state': 'available' if available else 'disabled' if not settings.web_search_enabled else 'unavailable',
                'available': available, 'configured': configured,
                'code': 'AI_WEB_CONFIGURED' if available else 'AI_WEB_UNAVAILABLE',
                'supportedModes': ['off', 'on'] if available else ['off'],
                'message': 'Web research is configured; each query needs your approval. Provider connectivity is checked when searching.' if available else 'Web search is unavailable. Ordinary chat uses the local model.'},
        'retrieval': {'adapter': 'brave', 'implemented': True,
                      'enabled': settings.web_search_enabled, 'configured': configured,
                      'liveConnectivity': 'not_checked',
                      'state': 'disabled' if not settings.web_search_enabled else 'configured' if configured else 'unavailable'},
        'documents': {'state': 'available', 'available': True, 'configured': True,
                      'code': 'AI_DOCUMENTS_CONFIGURED',
                      'message': 'Select uploaded documents for this conversation. Only authorized matching text is used; no external search is needed.'},
    }


def require_supported_research(research: ResearchRequest, streaming=False):
    # Never silently fall back to an unresearched answer for an explicit request.
    if research.documentIds and not streaming:
        raise HTTPException(422, detail={'code': 'DOCUMENT_STREAM_REQUIRED', 'message': 'Document retrieval requires a saved streaming conversation.'})
    if research.mode == 'auto':
        raise HTTPException(422, detail={'code': 'WEB_AUTO_UNAVAILABLE', 'message': 'Choose explicit web research; automatic research is not supported yet.'})
    if research.mode != 'off' and not research_capabilities()['web']['available']:
        raise HTTPException(503, detail={'code': 'AI_WEB_UNAVAILABLE',
                                        'message': 'Web search is not available. No external query was sent.'})
    if research.mode != 'off' and not streaming:
        raise HTTPException(422, detail={'code': 'WEB_STREAM_REQUIRED', 'message': 'Web research requires the streaming chat endpoint.'})
