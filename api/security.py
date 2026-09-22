"""Optional private service-token authentication for the Python AI service."""

import hmac
import re

from fastapi import Header, HTTPException, Request, status

from config import get_settings


_SECRET_PATTERNS = (
    re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE),
    re.compile(
        r"\b(?:password|passwd|pwd|api[_-]?key|access[_-]?token|secret)\s*[:=]\s*[^\s,;]{4,}",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:sk-[A-Za-z0-9_-]{12,}|github_pat_[A-Za-z0-9_]{12,}|AIza[A-Za-z0-9_-]{20,})\b"
    ),
    re.compile(r"\b(?:mysql|postgres(?:ql)?|mongodb)://[^\s]+", re.IGNORECASE),
)


def redact_sensitive_text(value: object, maximum_characters: int = 1_000) -> str:
    """Return bounded text safe for diagnostics; never use it for model context."""
    maximum = max(0, maximum_characters)
    redacted = str(value or "")[:maximum]
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return " ".join(redacted.replace("\r", " ").replace("\n", " ").split())[:maximum]


def require_service_token(
    authorization: str | None = Header(default=None),
    x_voltforge_ai_token: str | None = Header(default=None),
    *,
    request: Request = None,
) -> None:
    """Require a constant-time token match whenever a token is configured.

    Development and test deployments may omit the token for local compatibility.
    Production settings fail closed and this defense-in-depth branch returns a
    service-unavailable response if configuration is ever bypassed.
    """
    settings = get_settings()
    expected = settings.api_token
    # The contract document is content-free and may remain available to local
    # development tooling. Production still requires the same private token as
    # every other route, including health and contract metadata.
    if (
        request is not None
        and request.url.path.endswith("/contract")
        and not settings.service_authentication_required
    ):
        return
    if not expected:
        if settings.service_authentication_required:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="AI service authentication is not configured.",
            )
        return

    supplied = x_voltforge_ai_token
    if not supplied and authorization and authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="AI service authentication required.",
        )
