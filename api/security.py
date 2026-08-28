"""Optional private service-token authentication for the Python AI service."""

import hmac

from fastapi import Header, HTTPException, status

from config import get_settings


def require_service_token(
    authorization: str | None = Header(default=None),
    x_voltforge_ai_token: str | None = Header(default=None),
) -> None:
    """Require a constant-time token match whenever a token is configured.

    An empty token keeps existing local development compatible. Production
    deployments should configure the same non-empty value on this service and
    the Java gateway.
    """
    expected = get_settings().api_token
    if not expected:
        return

    supplied = x_voltforge_ai_token
    if not supplied and authorization and authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="AI service authentication required.",
        )
