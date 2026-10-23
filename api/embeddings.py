"""Private BL-only encoder. Ownership is checked by BL before text reaches it."""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator
from api.security import require_service_token
from engine.embeddings import embedding_engine
from embedding_model import MODEL_ID, MANIFEST
from config import get_settings

router = APIRouter(prefix='/voltForge-ai/api/v1/model/embeddings', dependencies=[Depends(require_service_token)])


class EncodeRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    modelId: str = Field(pattern=r'^[a-f0-9]{64}$')
    texts: list[str] = Field(min_length=1, max_length=32)

    @field_validator('texts')
    @classmethod
    def bounded_text(cls, values):
        if any(not value.strip() or len(value)>2000 for value in values):
            raise ValueError('Each reference excerpt must contain 1–2000 characters')
        return values


@router.get('')
def status():
    return {'modelId':MODEL_ID, 'dimensions':MANIFEST['dimensions'], 'maxTokens':MANIFEST['max_tokens'],
            'available': get_settings().embeddings_enabled and embedding_engine.session is not None}


@router.post('')
def encode(request: EncodeRequest):
    if not status()['available']:
        raise HTTPException(503, detail='Reference embeddings unavailable. Check model setup or use lexical retrieval.')
    if request.modelId != MODEL_ID:
        raise HTTPException(409, detail='Embedding model changed; refresh its identity and rebuild derived vectors.')
    return {'modelId':MODEL_ID, 'dimensions':MANIFEST['dimensions'], 'vectors':embedding_engine.encode(request.texts)}
