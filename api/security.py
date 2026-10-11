"""
Service token authentication for VoltForge AI microservice.
"""

import hmac
from fastapi import Header, HTTPException, Request, status
from config import get_settings

def require_service_token(
    authorization: str | None = Header(default=None),
    x_voltforge_ai_token: str | None = Header(default=None),
    request: Request = None,
) -> None:
    settings = get_settings()
    expected = settings.api_token
    if not expected:
        return  # No token required if not configured
        
    token = x_voltforge_ai_token
    if not token and authorization:
        if authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
        else:
            token = authorization.strip()
            
    if not token or not hmac.compare_digest(str(token), str(expected)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Invalid or missing X-Voltforge-AI-Token header."
        )

