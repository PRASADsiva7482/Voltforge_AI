"""Policy-gated, optional internet evidence retrieval service."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import hashlib
import logging
import threading
import time
from typing import Any, Callable, Mapping

from config import AiSettings, get_settings
from internet_retrieval.provider import (
    DuckDuckGoInstantAnswerProvider,
    ProviderResult,
    SafeJsonClient,
)
from internet_retrieval.schema import (
    InternetEvidence,
    InternetRetrievalError,
    InternetRetrievalQuery,
    InternetRetrievalResponse,
    POLICY_ID,
    POLICY_SHA256,
    PROVIDER_ID,
    content_sha256,
    load_policy,
    sha256_json,
    utc_timestamp,
)
from observability import observability


logger = logging.getLogger("voltforge-ai.internet-retrieval")
Clock = Callable[[], datetime]


@dataclass(frozen=True)
class _CacheEntry:
    response: InternetRetrievalResponse
    expires_at: datetime


class SanitizedEvidenceCache:
    def __init__(self, maximum_entries: int):
        self.maximum_entries = maximum_entries
        self._entries: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str, now: datetime) -> InternetRetrievalResponse | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if now >= entry.expires_at:
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return entry.response.model_copy(
                update={
                    "reasonCode": "INTERNET_EVIDENCE_CACHE_HIT",
                    "networkAttempted": False,
                    "networkAccessed": False,
                    "cacheHit": True,
                }
            )

    def put(
        self,
        key: str,
        response: InternetRetrievalResponse,
        expires_at: datetime,
    ) -> None:
        with self._lock:
            self._entries[key] = _CacheEntry(response=response, expires_at=expires_at)
            self._entries.move_to_end(key)
            while len(self._entries) > self.maximum_entries:
                self._entries.popitem(last=False)

    def size(self, now: datetime) -> int:
        with self._lock:
            expired = [key for key, value in self._entries.items() if now >= value.expires_at]
            for key in expired:
                del self._entries[key]
            return len(self._entries)


class InternetRetrievalService:
    def __init__(
        self,
        *,
        enabled: bool,
        timeout_seconds: int,
        maximum_response_bytes: int,
        cache_ttl_seconds: int,
        provider_id: str = PROVIDER_ID,
        provider: DuckDuckGoInstantAnswerProvider | None = None,
        cache: SanitizedEvidenceCache | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.policy = load_policy()
        if provider_id != PROVIDER_ID:
            raise InternetRetrievalError(
                "INTERNET_PROVIDER_NOT_APPROVED", "The selected provider is not approved."
            )
        limits = self.policy["limits"]
        cache_policy = self.policy["cache"]
        self.enabled = bool(enabled)
        self.timeout_seconds = max(
            int(limits["minimumTimeoutSeconds"]),
            min(int(timeout_seconds), int(limits["maximumTimeoutSeconds"])),
        )
        self.maximum_response_bytes = min(
            int(maximum_response_bytes), int(limits["maximumResponseBytes"])
        )
        self.cache_ttl_seconds = max(
            int(cache_policy["minimumTtlSeconds"]),
            min(int(cache_ttl_seconds), int(cache_policy["maximumTtlSeconds"])),
        )
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.cache = cache or SanitizedEvidenceCache(int(cache_policy["maximumEntries"]))
        if provider is None:
            provider_policy = self.policy["providers"][0]
            client = SafeJsonClient(
                provider_policy,
                timeout_seconds=self.timeout_seconds,
                maximum_response_bytes=self.maximum_response_bytes,
            )
            provider = DuckDuckGoInstantAnswerProvider(self.policy, client)
        self.provider = provider
        self._last_failure_code: str | None = None
        self._state_lock = threading.RLock()

    def search(
        self,
        query: InternetRetrievalQuery | Mapping[str, Any],
        *,
        trigger: str,
        bypass_cache: bool = False,
    ) -> InternetRetrievalResponse:
        """Search while recording only bounded status and timing facts."""
        started = time.perf_counter()
        try:
            response = self._search(query, trigger=trigger, bypass_cache=bypass_cache)
        except InternetRetrievalError as error:
            observability.record_retrieval(
                status="failed",
                failure_code=getattr(error, "code", "INTERNET_RETRIEVAL_UNAVAILABLE"),
                duration_ms=(time.perf_counter() - started) * 1000,
            )
            raise
        except Exception:
            observability.record_retrieval(
                status="failed",
                failure_code="PROVIDER_UNEXPECTED_FAILURE",
                duration_ms=(time.perf_counter() - started) * 1000,
            )
            raise
        observability.record_retrieval(
            status=response.status,
            cache_hit=bool(response.cacheHit),
            failure_code=response.reasonCode,
            duration_ms=(time.perf_counter() - started) * 1000,
        )
        logger.debug(
            "Internet retrieval completed status=%s cacheHit=%s durationMs=%s",
            response.status,
            response.cacheHit,
            round((time.perf_counter() - started) * 1000, 3),
        )
        return response

    def _search(
        self,
        query: InternetRetrievalQuery | Mapping[str, Any],
        *,
        trigger: str,
        bypass_cache: bool = False,
    ) -> InternetRetrievalResponse:
        request = (
            query
            if isinstance(query, InternetRetrievalQuery)
            else InternetRetrievalQuery.model_validate(query)
        )
        now = self._now()
        query_hash = hashlib.sha256(request.text.encode("utf-8")).hexdigest()
        if trigger == "not-triggered":
            return self._response(
                status="not-requested",
                reason_code="INTERNET_POLICY_NOT_TRIGGERED",
                trigger=trigger,
                query_hash=query_hash,
                now=now,
            )
        if trigger not in {
            "explicit-request",
            "local-evidence-insufficient",
            "direct-endpoint",
        }:
            raise ValueError("internet retrieval trigger is invalid")
        if not self.enabled:
            return self._response(
                status="disabled",
                reason_code="INTERNET_RETRIEVAL_DISABLED",
                trigger=trigger,
                query_hash=query_hash,
                now=now,
            )
        cache_key = sha256_json(
            {
                "policySha256": self.policy["policySha256"],
                "providerId": PROVIDER_ID,
                "querySha256": query_hash,
                "maximumResults": request.maximumResults,
            }
        )
        if not bypass_cache:
            cached = self.cache.get(cache_key, now)
            if cached is not None:
                return cached.model_copy(update={"trigger": trigger})
        try:
            provider_result = self.provider.search(request.text, request.maximumResults)
        except InternetRetrievalError as error:
            self._record_failure(error.code)
            logger.warning("Internet evidence retrieval degraded (code=%s)", error.code)
            return self._response(
                status="degraded",
                reason_code=error.code,
                trigger=trigger,
                query_hash=query_hash,
                now=now,
                network_attempted=True,
                network_accessed=True,
            )
        except Exception:
            self._record_failure("PROVIDER_UNEXPECTED_FAILURE")
            logger.exception("Internet evidence retrieval degraded without request content")
            return self._response(
                status="degraded",
                reason_code="PROVIDER_UNEXPECTED_FAILURE",
                trigger=trigger,
                query_hash=query_hash,
                now=now,
                network_attempted=True,
                network_accessed=True,
            )
        self._record_failure(None)
        response = self._successful_response(
            request,
            provider_result,
            trigger=trigger,
            query_hash=query_hash,
            now=now,
        )
        self.cache.put(
            cache_key,
            response,
            now + timedelta(seconds=self.cache_ttl_seconds),
        )
        return response

    def _successful_response(
        self,
        request: InternetRetrievalQuery,
        provider_result: ProviderResult,
        *,
        trigger: str,
        query_hash: str,
        now: datetime,
    ) -> InternetRetrievalResponse:
        retrieved_at = utc_timestamp(now)
        expires_at = utc_timestamp(now + timedelta(seconds=self.cache_ttl_seconds))
        evidence = []
        for rank, item in enumerate(provider_result.evidence[: request.maximumResults], start=1):
            digest = content_sha256(item.title, item.snippet, item.source_url)
            identity = {"providerId": PROVIDER_ID, "contentSha256": digest}
            evidence.append(
                InternetEvidence(
                    evidenceId=f"evidence:web:{sha256_json(identity)[:24]}",
                    citationId=f"citation:web:{sha256_json({'citation': identity})[:24]}",
                    rank=rank,
                    title=item.title,
                    snippet=item.snippet,
                    sourceUrl=item.source_url,
                    sourceDomain=item.source_domain,
                    retrievedAt=retrieved_at,
                    contentSha256=digest,
                )
            )
        return self._response(
            status="complete" if evidence else "no-results",
            reason_code=(
                "INTERNET_EVIDENCE_RETRIEVED"
                if evidence
                else "INTERNET_PROVIDER_NO_SAFE_RESULTS"
            ),
            trigger=trigger,
            query_hash=query_hash,
            now=now,
            evidence=evidence,
            blocked_result_count=provider_result.blocked_count,
            network_attempted=True,
            network_accessed=True,
            retrieved_at=retrieved_at,
            cache_expires_at=expires_at,
        )

    def _response(
        self,
        *,
        status: str,
        reason_code: str,
        trigger: str,
        query_hash: str,
        now: datetime,
        evidence: list[InternetEvidence] | None = None,
        blocked_result_count: int = 0,
        network_attempted: bool = False,
        network_accessed: bool = False,
        retrieved_at: str | None = None,
        cache_expires_at: str | None = None,
    ) -> InternetRetrievalResponse:
        bounded_evidence = evidence or []
        identity = {
            "policySha256": self.policy["policySha256"],
            "querySha256": query_hash,
            "trigger": trigger,
            "status": status,
            "reasonCode": reason_code,
            "retrievedAt": retrieved_at,
            "evidenceIds": [item.evidenceId for item in bounded_evidence],
        }
        return InternetRetrievalResponse(
            policySha256=self.policy["policySha256"],
            responseId=f"vf-internet-response-v1-{sha256_json(identity)[:24]}",
            status=status,
            reasonCode=reason_code,
            trigger=trigger,
            querySha256=query_hash,
            evidence=bounded_evidence,
            returnedCount=len(bounded_evidence),
            blockedResultCount=blocked_result_count,
            networkAttempted=network_attempted,
            networkAccessed=network_accessed,
            cacheHit=False,
            cacheTtlSeconds=self.cache_ttl_seconds,
            cacheExpiresAt=cache_expires_at,
            retrievedAt=retrieved_at,
            degraded=status == "degraded",
        )

    def health(self) -> dict[str, Any]:
        now = self._now()
        with self._state_lock:
            failure = self._last_failure_code
        state = "offline" if not self.enabled else "degraded" if failure else "ready"
        return {
            "enabled": self.enabled,
            "ready": self.enabled and failure is None,
            "state": state,
            "code": (
                "INTERNET_RETRIEVAL_DISABLED"
                if not self.enabled
                else failure or "INTERNET_RETRIEVAL_READY"
            ),
            "policy": "optional-evidence-only",
            "policyId": POLICY_ID,
            "policySha256": self.policy["policySha256"],
            "providerId": PROVIDER_ID,
            "approvedDomains": ["api.duckduckgo.com"],
            "generationDependency": False,
            "arbitraryUrlFetchAllowed": False,
            "redirectsAllowed": False,
            "binaryContentAllowed": False,
            "environmentProxyUseAllowed": False,
            "cacheTtlSeconds": self.cache_ttl_seconds,
            "cacheEntryCount": self.cache.size(now),
            "maximumResponseBytes": self.maximum_response_bytes,
            "webContentTrainingAllowed": False,
            "rawQueryStored": False,
        }

    def _record_failure(self, code: str | None) -> None:
        with self._state_lock:
            self._last_failure_code = code

    def _now(self) -> datetime:
        value = self.clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("internet retrieval clock must be timezone-aware")
        return value.astimezone(timezone.utc)


def retrieval_trigger(message: str, local_status: str) -> str:
    policy = load_policy()["triggerPolicy"]
    normalized = " ".join(message.casefold().split())
    if any(term in normalized for term in policy["explicitRequest"]):
        return "explicit-request"
    local_insufficient = local_status in set(policy["localEvidenceInsufficientStatuses"])
    evidence_intent = any(term in normalized for term in policy["evidenceIntentTerms"])
    if local_insufficient and evidence_intent:
        return "local-evidence-insufficient"
    return "not-triggered"


@lru_cache(maxsize=8)
def get_internet_retrieval_service(
    enabled: bool,
    timeout_seconds: int,
    maximum_response_bytes: int,
    cache_ttl_seconds: int,
    provider_id: str,
) -> InternetRetrievalService:
    return InternetRetrievalService(
        enabled=enabled,
        timeout_seconds=timeout_seconds,
        maximum_response_bytes=maximum_response_bytes,
        cache_ttl_seconds=cache_ttl_seconds,
        provider_id=provider_id,
    )


def service_from_settings(settings: AiSettings | None = None) -> InternetRetrievalService:
    configured = settings or get_settings()
    return get_internet_retrieval_service(
        configured.internet_retrieval_enabled,
        configured.internet_retrieval_timeout_seconds,
        configured.internet_retrieval_max_response_bytes,
        configured.internet_retrieval_cache_ttl_seconds,
        configured.internet_retrieval_provider,
    )


def search_for_request(
    request: Any,
    local_response: Any,
    settings: AiSettings | None = None,
) -> InternetRetrievalResponse:
    message = str(getattr(request, "message", ""))
    bounded = " ".join(
        "".join(
            character
            for character in message
            if character in "\t\n\r" or (ord(character) >= 32 and ord(character) != 127)
        ).split()
    )[:1_000]
    if not bounded:
        bounded = "unusable search text"
    try:
        trigger = retrieval_trigger(
            message, str(getattr(local_response, "status", "unavailable"))
        )
        return service_from_settings(settings).search(
            InternetRetrievalQuery(text=bounded, maximumResults=3),
            trigger=trigger,
        )
    except Exception as error:
        return degraded_without_service(bounded, error)


def search_internet(
    query: InternetRetrievalQuery | Mapping[str, Any],
    settings: AiSettings | None = None,
) -> InternetRetrievalResponse:
    request = (
        query
        if isinstance(query, InternetRetrievalQuery)
        else InternetRetrievalQuery.model_validate(query)
    )
    try:
        return service_from_settings(settings).search(request, trigger="direct-endpoint")
    except Exception as error:
        return degraded_without_service(request.text, error, trigger="direct-endpoint")


def degraded_without_service(
    query_text: str,
    error: Exception,
    *,
    trigger: str = "not-triggered",
) -> InternetRetrievalResponse:
    """Keep local processing alive when policy/service construction itself fails."""

    query_hash = hashlib.sha256(query_text.encode("utf-8")).hexdigest()
    raw_code = str(getattr(error, "code", "INTERNET_RETRIEVAL_UNAVAILABLE"))
    reason_code = "".join(
        character if character.isascii() and (character.isupper() or character.isdigit()) else "_"
        for character in raw_code.upper()
    ).strip("_")[:100]
    if len(reason_code) < 3:
        reason_code = "INTERNET_RETRIEVAL_UNAVAILABLE"
    identity = {
        "policySha256": POLICY_SHA256,
        "querySha256": query_hash,
        "trigger": trigger,
        "reasonCode": reason_code,
    }
    return InternetRetrievalResponse(
        policySha256=POLICY_SHA256,
        responseId=f"vf-internet-response-v1-{sha256_json(identity)[:24]}",
        status="degraded",
        reasonCode=reason_code,
        trigger=trigger,
        querySha256=query_hash,
        cacheTtlSeconds=900,
        degraded=True,
    )


def internet_retrieval_health(settings: AiSettings | None = None) -> dict[str, Any]:
    try:
        return service_from_settings(settings).health()
    except InternetRetrievalError as error:
        return {
            "enabled": False,
            "ready": False,
            "state": "degraded",
            "code": error.code,
            "policy": "optional-evidence-only",
            "policyId": POLICY_ID,
            "providerId": PROVIDER_ID,
            "approvedDomains": [],
            "generationDependency": False,
            "arbitraryUrlFetchAllowed": False,
            "redirectsAllowed": False,
            "binaryContentAllowed": False,
            "environmentProxyUseAllowed": False,
            "webContentTrainingAllowed": False,
            "rawQueryStored": False,
        }
    except Exception:
        return {
            "enabled": False,
            "ready": False,
            "state": "degraded",
            "code": "INTERNET_RETRIEVAL_UNAVAILABLE",
            "policy": "optional-evidence-only",
            "policyId": POLICY_ID,
            "providerId": PROVIDER_ID,
            "approvedDomains": [],
            "generationDependency": False,
            "arbitraryUrlFetchAllowed": False,
            "redirectsAllowed": False,
            "binaryContentAllowed": False,
            "environmentProxyUseAllowed": False,
            "webContentTrainingAllowed": False,
            "rawQueryStored": False,
        }


def internet_tool_event(response: InternetRetrievalResponse) -> dict[str, Any]:
    evidence = [
        {
            "evidenceId": item.evidenceId,
            "citationId": item.citationId,
            "title": item.title,
            "snippet": item.snippet,
            "sourceUrl": item.sourceUrl,
            "sourceDomain": item.sourceDomain,
            "retrievedAt": item.retrievedAt,
            "contentSha256": item.contentSha256,
            "untrustedContent": True,
            "trainingUseAllowed": False,
        }
        for item in response.evidence[:2]
    ]
    return {
        "name": "secure-internet-evidence",
        "version": response.contractVersion,
        "status": (
            "complete" if response.status in {"complete", "no-results"} else "unavailable"
        ),
        "authority": "retrieved",
        "summary": (
            f"Retrieved {response.returnedCount} sanitized untrusted internet evidence item(s)."
            if response.status == "complete"
            else "The approved internet provider returned no safe evidence."
            if response.status == "no-results"
            else "Optional internet evidence is offline or degraded; local processing continues."
        ),
        "evidence": {
            "policyId": response.policyId,
            "policySha256": response.policySha256,
            "providerId": response.providerId,
            "providerDomain": response.providerDomain,
            "retrievalStatus": response.status,
            "reasonCode": response.reasonCode,
            "trigger": response.trigger,
            "results": evidence,
            "returnedCount": response.returnedCount,
            "blockedResultCount": response.blockedResultCount,
            "networkAccessed": response.networkAccessed,
            "cacheHit": response.cacheHit,
            "untrustedContent": True,
            "generationDependency": False,
            "arbitraryUrlFetchAllowed": False,
            "rawQueryStored": False,
            "rawProviderPayloadStored": False,
            "rawProjectContextSent": False,
            "trainingUseAllowed": False,
        },
    }


def response_metadata(response: InternetRetrievalResponse) -> dict[str, Any]:
    return {
        "policyId": response.policyId,
        "providerId": response.providerId,
        "providerDomain": response.providerDomain,
        "status": response.status,
        "reasonCode": response.reasonCode,
        "trigger": response.trigger,
        "returnedCount": response.returnedCount,
        "blockedResultCount": response.blockedResultCount,
        "networkAttempted": response.networkAttempted,
        "networkAccessed": response.networkAccessed,
        "cacheHit": response.cacheHit,
        "retrievedAt": response.retrievedAt,
        "degraded": response.degraded,
        "untrustedContent": True,
        "generationDependency": False,
        "arbitraryUrlFetchAllowed": False,
        "rawQueryStored": False,
        "rawProviderPayloadStored": False,
        "rawProjectContextSent": False,
        "trainingUseAllowed": False,
    }


def citations_from_response(response: InternetRetrievalResponse) -> list[dict[str, str]]:
    return [
        {
            "citationId": item.citationId,
            "sourceId": f"web-provider:{item.providerId}",
            "sourceRevision": item.retrievedAt,
            "title": item.title,
            "locator": item.sourceUrl,
            "url": item.sourceUrl,
            "source": item.sourceDomain,
            "contentSha256": item.contentSha256,
            "retrievedAt": item.retrievedAt,
            "authority": "retrieved",
            "untrustedContent": "true",
            "trainingUseAllowed": "false",
        }
        for item in response.evidence
    ]


def render_internet_summary(
    response: InternetRetrievalResponse, maximum_results: int = 3
) -> str:
    if response.status != "complete":
        return ""
    lines = [
        "Internet evidence (untrusted; verify against the cited source and deterministic checks):"
    ]
    for item in response.evidence[:maximum_results]:
        lines.append(f"- [{item.citationId}] {item.title}: {item.snippet}")
    return "\n".join(lines)
