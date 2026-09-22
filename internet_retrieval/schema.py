"""Strict public contracts for optional internet evidence retrieval."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


POLICY_ID = "vfai023-secure-internet-evidence-v1"
POLICY_SHA256 = "3a7a34e0610fafc361ff4fb979e0bd49d3cfbd362413adb408ddf4ddd380c37c"
CONTRACT_VERSION = "1.0.0"
PROVIDER_ID = "duckduckgo-instant-answer-v1"
PACKAGE_DIR = Path(__file__).resolve().parent
POLICY_PATH = PACKAGE_DIR / "policy.v1.json"
SHA256_PATTERN = r"^[a-f0-9]{64}$"
UTC_TIMESTAMP_PATTERN = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$"


class InternetRetrievalError(RuntimeError):
    """Content-free retrieval error safe to expose as a reason code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InternetRetrievalQuery(ContractModel):
    text: str = Field(min_length=1, max_length=1_000)
    maximumResults: int = Field(default=3, ge=1, le=4)

    @field_validator("text")
    @classmethod
    def clean_query(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("internet retrieval query cannot be blank")
        if any(ord(character) == 0 for character in cleaned):
            raise ValueError("internet retrieval query contains a null byte")
        return cleaned


class InternetEvidence(ContractModel):
    evidenceId: str = Field(pattern=r"^evidence:web:[a-f0-9]{24}$")
    citationId: str = Field(pattern=r"^citation:web:[a-f0-9]{24}$")
    providerId: Literal["duckduckgo-instant-answer-v1"] = PROVIDER_ID
    rank: int = Field(ge=1, le=4)
    title: str = Field(min_length=1, max_length=300)
    snippet: str = Field(min_length=1, max_length=800)
    sourceUrl: str = Field(min_length=1, max_length=2_000)
    sourceDomain: str = Field(min_length=3, max_length=253)
    retrievedAt: str = Field(pattern=UTC_TIMESTAMP_PATTERN)
    sourcePublishedAt: str | None = Field(default=None, pattern=UTC_TIMESTAMP_PATTERN)
    contentSha256: str = Field(pattern=SHA256_PATTERN)
    authority: Literal["retrieved"] = "retrieved"
    untrustedContent: Literal[True] = True
    sanitized: Literal[True] = True
    promptInjectionDetected: Literal[False] = False
    trainingUseAllowed: Literal[False] = False

    @field_validator("sourceUrl")
    @classmethod
    def citation_url_is_safe_to_display(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            raise ValueError("internet citation URL must be a credential-free HTTPS URL")
        return value


InternetStatus = Literal[
    "not-requested",
    "disabled",
    "complete",
    "no-results",
    "degraded",
]
InternetTrigger = Literal[
    "not-triggered",
    "explicit-request",
    "local-evidence-insufficient",
    "direct-endpoint",
]


class InternetRetrievalResponse(ContractModel):
    schemaVersion: Literal[1] = 1
    contractVersion: Literal["1.0.0"] = CONTRACT_VERSION
    policyId: Literal["vfai023-secure-internet-evidence-v1"] = POLICY_ID
    policySha256: str = Field(pattern=SHA256_PATTERN)
    responseId: str = Field(pattern=r"^vf-internet-response-v1-[a-f0-9]{24}$")
    status: InternetStatus
    reasonCode: str = Field(pattern=r"^[A-Z][A-Z0-9_]{2,99}$")
    trigger: InternetTrigger
    querySha256: str = Field(pattern=SHA256_PATTERN)
    providerId: Literal["duckduckgo-instant-answer-v1"] = PROVIDER_ID
    providerDomain: Literal["api.duckduckgo.com"] = "api.duckduckgo.com"
    evidence: list[InternetEvidence] = Field(default_factory=list, max_length=4)
    returnedCount: int = Field(default=0, ge=0, le=4)
    blockedResultCount: int = Field(default=0, ge=0, le=32)
    networkAttempted: bool = False
    networkAccessed: bool = False
    cacheHit: bool = False
    cacheTtlSeconds: int = Field(ge=60, le=86_400)
    cacheExpiresAt: str | None = Field(default=None, pattern=UTC_TIMESTAMP_PATTERN)
    retrievedAt: str | None = Field(default=None, pattern=UTC_TIMESTAMP_PATTERN)
    degraded: bool = False
    authority: Literal["retrieved"] = "retrieved"
    untrustedContent: Literal[True] = True
    generationDependency: Literal[False] = False
    arbitraryUrlFetchAllowed: Literal[False] = False
    rawQueryStored: Literal[False] = False
    rawProviderPayloadStored: Literal[False] = False
    rawProjectContextSent: Literal[False] = False
    trainingUseAllowed: Literal[False] = False

    @model_validator(mode="after")
    def response_consistency(self) -> "InternetRetrievalResponse":
        if self.returnedCount != len(self.evidence):
            raise ValueError("returnedCount does not match internet evidence")
        if self.status == "complete" and not self.evidence:
            raise ValueError("complete internet retrieval requires evidence")
        if self.status != "complete" and self.evidence:
            raise ValueError("non-complete internet retrieval cannot expose evidence")
        if self.degraded != (self.status == "degraded"):
            raise ValueError("degraded state must be explicit")
        if self.status in {"not-requested", "disabled"} and (
            self.networkAttempted or self.networkAccessed
        ):
            raise ValueError("offline retrieval states cannot report network use")
        if self.cacheHit and (self.networkAttempted or self.networkAccessed):
            raise ValueError("a cache hit cannot report live network use")
        if self.cacheHit and not self.cacheExpiresAt:
            raise ValueError("a cache hit requires an expiry timestamp")
        if self.status in {"complete", "no-results"} and not self.retrievedAt:
            raise ValueError("completed retrieval requires a source timestamp")
        return self


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


@lru_cache(maxsize=1)
def load_policy() -> dict[str, Any]:
    try:
        policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise InternetRetrievalError(
            "INTERNET_POLICY_INVALID", "The internet retrieval policy is unreadable."
        ) from error
    if not isinstance(policy, dict) or policy.get("policyId") != POLICY_ID:
        raise InternetRetrievalError(
            "INTERNET_POLICY_INVALID", "The internet retrieval policy identity is invalid."
        )
    declared = policy.get("policySha256")
    unsigned = dict(policy)
    unsigned.pop("policySha256", None)
    if declared != POLICY_SHA256 or declared != sha256_json(unsigned):
        raise InternetRetrievalError(
            "INTERNET_POLICY_CHECKSUM_MISMATCH",
            "The internet retrieval policy checksum is invalid.",
        )
    providers = policy.get("providers")
    if not isinstance(providers, list) or len(providers) != 1:
        raise InternetRetrievalError(
            "INTERNET_POLICY_INVALID", "Exactly one checked provider must be configured."
        )
    provider = providers[0]
    invariants = (
        provider.get("providerId") == PROVIDER_ID,
        provider.get("endpoint") == "https://api.duckduckgo.com/",
        provider.get("allowedDomains") == ["api.duckduckgo.com"],
        provider.get("resultUrlFetchingAllowed") is False,
        policy.get("network", {}).get("redirectPolicy") == "reject-all",
        policy.get("network", {}).get("environmentProxyUseAllowed") is False,
        policy.get("training", {}).get("webContentTrainingAllowed") is False,
        policy.get("trust", {}).get("generationDependency") is False,
    )
    if not all(invariants):
        raise InternetRetrievalError(
            "INTERNET_POLICY_INVALID", "The internet retrieval safety invariants changed."
        )
    return policy


def utc_timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("internet retrieval timestamps must be timezone-aware")
    normalized = value.astimezone(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")
    return f"{normalized}Z"


def content_sha256(title: str, snippet: str, source_url: str) -> str:
    return sha256_json(
        {"title": title, "snippet": snippet, "sourceUrl": source_url}
    )
