"""SSRF and untrusted-content boundary for internet evidence retrieval."""

from __future__ import annotations

from html.parser import HTMLParser
import ipaddress
import re
import socket
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from internet_retrieval.schema import InternetRetrievalError


Resolver = Callable[..., list[tuple[Any, ...]]]


_HOST_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_INJECTION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in (
        r"ignore\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|system|developer)\s+instructions?",
        r"(?:reveal|print|show|return)\s+(?:the\s+)?(?:system|developer)\s+prompt",
        r"(?:system|developer)\s+(?:prompt|message|instructions?)",
        r"do\s+not\s+follow\s+(?:the\s+)?(?:previous|system|developer)",
        r"execute\s+(?:a\s+|the\s+)?(?:command|tool|shell|code)",
        r"you\s+are\s+(?:chatgpt|an?\s+ai\s+assistant|the\s+assistant)",
        r"<\|(?:system|assistant|developer)\|>",
        r"\[(?:/?inst|system|developer)\]",
    )
)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def sanitize_text(value: Any, maximum: int) -> str:
    extractor = _TextExtractor()
    extractor.feed(str(value or ""))
    extractor.close()
    visible = " ".join(extractor.parts)
    visible = "".join(
        character
        for character in visible
        if character in "\t\n\r" or (ord(character) >= 32 and ord(character) != 127)
    )
    return " ".join(visible.split())[:maximum].strip()


def contains_prompt_injection(*values: str, maximum_scan: int = 4096) -> bool:
    candidate = "\n".join(values)[:maximum_scan]
    return any(pattern.search(candidate) is not None for pattern in _INJECTION_PATTERNS)


def _normalized_hostname(hostname: str) -> str:
    try:
        host = hostname.rstrip(".").encode("idna").decode("ascii").lower()
    except UnicodeError as error:
        raise InternetRetrievalError(
            "UNSAFE_PROVIDER_HOST", "The provider hostname is invalid."
        ) from error
    if len(host) > 253 or not host or host == "localhost":
        raise InternetRetrievalError(
            "UNSAFE_PROVIDER_HOST", "The provider hostname is not publicly routable."
        )
    labels = host.split(".")
    if len(labels) < 2 or any(not _HOST_LABEL.fullmatch(label) for label in labels):
        raise InternetRetrievalError(
            "UNSAFE_PROVIDER_HOST", "The provider hostname is malformed."
        )
    return host


def _public_ip(address: str) -> str:
    try:
        parsed = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError as error:
        raise InternetRetrievalError(
            "UNSAFE_PROVIDER_ADDRESS", "Provider DNS returned a malformed address."
        ) from error
    if (
        not parsed.is_global
        or parsed.is_private
        or parsed.is_loopback
        or parsed.is_link_local
        or parsed.is_multicast
        or parsed.is_reserved
        or parsed.is_unspecified
    ):
        raise InternetRetrievalError(
            "UNSAFE_PROVIDER_ADDRESS",
            "Provider DNS resolved to a private, reserved, or non-global address.",
        )
    return parsed.compressed


class ProviderUrlGuard:
    """Validate the one policy-owned provider endpoint before every request."""

    def __init__(self, provider_policy: Mapping[str, Any], resolver: Resolver | None = None):
        self.provider_policy = provider_policy
        self.resolver = resolver or socket.getaddrinfo

    def validate(self, url: str) -> tuple[str, ...]:
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            raise InternetRetrievalError(
                "UNSAFE_PROVIDER_URL", "The provider URL violates the HTTPS boundary."
            )
        host = _normalized_hostname(parsed.hostname)
        if host not in self.provider_policy["allowedDomains"]:
            raise InternetRetrievalError(
                "UNAPPROVED_PROVIDER_DOMAIN", "The provider domain is not approved."
            )
        if parsed.port not in {None, 443}:
            raise InternetRetrievalError(
                "UNSAFE_PROVIDER_PORT", "The provider port is not approved."
            )
        if parsed.path not in self.provider_policy["allowedPaths"]:
            raise InternetRetrievalError(
                "UNAPPROVED_PROVIDER_PATH", "The provider path is not approved."
            )
        query_keys = {key for key, _ in parse_qsl(parsed.query, keep_blank_values=True)}
        if not query_keys.issubset(set(self.provider_policy["allowedQueryParameters"])):
            raise InternetRetrievalError(
                "UNAPPROVED_PROVIDER_QUERY", "The provider query contract is invalid."
            )
        try:
            resolved = self.resolver(host, 443, type=socket.SOCK_STREAM)
        except OSError as error:
            raise InternetRetrievalError(
                "PROVIDER_DNS_UNAVAILABLE", "The approved provider could not be resolved."
            ) from error
        addresses = sorted(
            {
                _public_ip(str(item[4][0]))
                for item in resolved
                if len(item) >= 5 and item[4]
            }
        )
        if not addresses:
            raise InternetRetrievalError(
                "PROVIDER_DNS_UNAVAILABLE", "Provider DNS returned no usable address."
            )
        return tuple(addresses)


def build_provider_url(endpoint: str, query: str) -> str:
    parsed = urlsplit(endpoint)
    parameters = {
        "format": "json",
        "no_html": "1",
        "no_redirect": "1",
        "q": query,
        "skip_disambig": "1",
    }
    return urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path or "/", urlencode(parameters), "")
    )


def safe_citation_url(value: Any, maximum: int = 2000) -> tuple[str, str] | None:
    raw = str(value or "").strip()
    if not raw or len(raw) > maximum:
        return None
    try:
        parsed = urlsplit(raw)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port not in {None, 443}
        ):
            return None
        host = _normalized_hostname(parsed.hostname)
        try:
            literal = ipaddress.ip_address(host)
        except ValueError:
            literal = None
        if literal is not None and not literal.is_global:
            return None
        if all(character.isdigit() or character == "." for character in host):
            return None
        safe = urlunsplit(("https", host, parsed.path or "/", parsed.query, ""))
        return safe[:maximum], host
    except (InternetRetrievalError, ValueError):
        return None
