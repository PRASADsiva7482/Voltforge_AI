"""One checked JSON evidence provider behind a bounded network client."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Iterable, Mapping

import requests

from internet_retrieval.schema import InternetRetrievalError
from internet_retrieval.security import (
    ProviderUrlGuard,
    build_provider_url,
    contains_prompt_injection,
    safe_citation_url,
    sanitize_text,
)


@dataclass(frozen=True)
class ProviderEvidence:
    title: str
    snippet: str
    source_url: str
    source_domain: str


@dataclass(frozen=True)
class ProviderResult:
    evidence: tuple[ProviderEvidence, ...]
    blocked_count: int


class SafeJsonClient:
    """Fetch one allow-listed JSON document without redirects, proxies, or retries."""

    def __init__(
        self,
        provider_policy: Mapping[str, Any],
        *,
        timeout_seconds: int,
        maximum_response_bytes: int,
        session: requests.Session | None = None,
        guard: ProviderUrlGuard | None = None,
    ) -> None:
        self.provider_policy = provider_policy
        self.timeout_seconds = timeout_seconds
        self.maximum_response_bytes = maximum_response_bytes
        self.session = session or requests.Session()
        self.session.trust_env = False
        self.guard = guard or ProviderUrlGuard(provider_policy)

    def get(self, url: str) -> dict[str, Any]:
        self.guard.validate(url)
        try:
            response = self.session.get(
                url,
                allow_redirects=False,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "VoltForge-AI-Evidence/1.0",
                },
                stream=True,
                timeout=(self.timeout_seconds, self.timeout_seconds),
            )
        except requests.RequestException as error:
            raise InternetRetrievalError(
                "PROVIDER_REQUEST_FAILED", "The approved evidence provider was unavailable."
            ) from error
        if 300 <= response.status_code < 400:
            response.close()
            raise InternetRetrievalError(
                "UNSAFE_PROVIDER_REDIRECT", "Provider redirects are not permitted."
            )
        if response.status_code != 200:
            response.close()
            raise InternetRetrievalError(
                "PROVIDER_HTTP_ERROR", "The approved evidence provider returned an error."
            )
        content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type not in set(self.provider_policy["acceptedContentTypes"]):
            response.close()
            raise InternetRetrievalError(
                "PROVIDER_CONTENT_TYPE_BLOCKED", "Non-JSON provider content was blocked."
            )
        declared_length = response.headers.get("Content-Length")
        if declared_length:
            try:
                parsed_length = int(declared_length)
            except ValueError as error:
                response.close()
                raise InternetRetrievalError(
                    "PROVIDER_CONTENT_LENGTH_INVALID",
                    "The provider content length was malformed.",
                ) from error
            if parsed_length < 0:
                response.close()
                raise InternetRetrievalError(
                    "PROVIDER_CONTENT_LENGTH_INVALID",
                    "The provider content length was malformed.",
                )
            if parsed_length > self.maximum_response_bytes:
                response.close()
                raise InternetRetrievalError(
                    "PROVIDER_RESPONSE_TOO_LARGE", "Provider content exceeded the byte limit."
                )
        chunks: list[bytes] = []
        total = 0
        try:
            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                total += len(chunk)
                if total > self.maximum_response_bytes:
                    raise InternetRetrievalError(
                        "PROVIDER_RESPONSE_TOO_LARGE", "Provider content exceeded the byte limit."
                    )
                chunks.append(chunk)
        finally:
            response.close()
        try:
            decoded = b"".join(chunks).decode("utf-8", errors="strict")
            payload = json.loads(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise InternetRetrievalError(
                "PROVIDER_JSON_INVALID", "The provider returned malformed JSON."
            ) from error
        if not isinstance(payload, dict):
            raise InternetRetrievalError(
                "PROVIDER_JSON_INVALID", "The provider JSON root must be an object."
            )
        return payload


class DuckDuckGoInstantAnswerProvider:
    provider_id = "duckduckgo-instant-answer-v1"

    def __init__(self, policy: Mapping[str, Any], client: SafeJsonClient):
        self.policy = policy
        self.provider_policy = policy["providers"][0]
        self.client = client

    def search(self, query: str, maximum_results: int) -> ProviderResult:
        url = build_provider_url(self.provider_policy["endpoint"], query)
        payload = self.client.get(url)
        limits = self.policy["limits"]
        candidates = list(self._candidate_rows(payload))[
            : int(limits["maximumProviderItemsExamined"])
        ]
        accepted: list[ProviderEvidence] = []
        blocked = 0
        seen: set[tuple[str, str]] = set()
        for raw_title, raw_snippet, raw_url in candidates:
            title = sanitize_text(raw_title, int(limits["maximumTitleCharacters"]))
            snippet = sanitize_text(raw_snippet, int(limits["maximumSnippetCharacters"]))
            citation = safe_citation_url(
                raw_url, int(limits["maximumSourceUrlCharacters"])
            )
            if not title or not snippet or citation is None:
                blocked += 1
                continue
            if contains_prompt_injection(
                title,
                snippet,
                maximum_scan=int(
                    self.policy["sanitization"]["maximumInjectionScanCharacters"]
                ),
            ):
                blocked += 1
                continue
            source_url, source_domain = citation
            identity = (source_url, snippet.casefold())
            if identity in seen:
                continue
            seen.add(identity)
            accepted.append(
                ProviderEvidence(
                    title=title,
                    snippet=snippet,
                    source_url=source_url,
                    source_domain=source_domain,
                )
            )
            if len(accepted) >= maximum_results:
                break
        return ProviderResult(tuple(accepted), blocked)

    def _candidate_rows(self, payload: Mapping[str, Any]) -> Iterable[tuple[Any, Any, Any]]:
        abstract = payload.get("AbstractText")
        abstract_url = payload.get("AbstractURL")
        if abstract and abstract_url:
            yield (
                payload.get("Heading") or payload.get("AbstractSource") or "Internet evidence",
                abstract,
                abstract_url,
            )
        for item in payload.get("Results") or []:
            if isinstance(item, Mapping):
                yield (item.get("Text"), item.get("Text"), item.get("FirstURL"))
        stack = list(reversed(payload.get("RelatedTopics") or []))
        examined = 0
        while stack and examined < 32:
            examined += 1
            item = stack.pop()
            if not isinstance(item, Mapping):
                continue
            topics = item.get("Topics")
            if isinstance(topics, list):
                stack.extend(reversed(topics))
                continue
            yield (item.get("Text"), item.get("Text"), item.get("FirstURL"))
