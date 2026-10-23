"""Bounded public web retrieval. No model invocation, private context or URL tools.

Uses the already-pinned HTTPCore transport to bind connections to validated IPs.
Source text remains untrusted reference material, not tool instructions.
"""
import asyncio
import hashlib
import ipaddress
import json
import re
import socket
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import urlencode, urlsplit, urlunsplit

import httpcore

from api.research import ResearchRequest, ResearchSource
from config import get_settings

BRAVE_URL = 'https://api.search.brave.com/res/v1/web/search'
PRIMARY_TECHNICAL_DOMAINS = ('arduino.cc', 'microchip.com', 'espressif.com', 'st.com', 'ti.com',
                             'analog.com', 'nxp.com', 'nordicsemi.com', 'raspberrypi.com')
TRANSITION_RANGES = tuple(ipaddress.ip_network(n) for n in
                          ('64:ff9b::/96', '64:ff9b:1::/48', '2002::/16', '2001::/32'))


class ResearchError(Exception):
    """Only static safe descriptions cross the API boundary."""
    def __init__(self, code, message, status=503):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


def public_ip(value):
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        raise ResearchError('WEB_ADDRESS_BLOCKED', 'The destination is not a public IP address.', 422) from None
    if (not ip.is_global or ip.is_multicast or ip.is_reserved
            or isinstance(ip, ipaddress.IPv6Address) and
            (ip.ipv4_mapped is not None or any(ip in net for net in TRANSITION_RANGES))):
        raise ResearchError('WEB_ADDRESS_BLOCKED', 'Non-public and transition addresses are blocked.', 422)
    return str(ip)


def public_url(value, max_length=2048):
    if not isinstance(value, str) or len(value) > max_length or re.search(r'[\x00-\x20\x7f\\]', value):
        raise ResearchError('WEB_URL_BLOCKED', 'The source URL is not supported.', 422)
    try:
        parts = urlsplit(value)
        host = (parts.hostname or '').encode('idna').decode('ascii').lower().rstrip('.')
        port = parts.port
    except (ValueError, UnicodeError):
        raise ResearchError('WEB_URL_BLOCKED', 'The source URL is not supported.', 422) from None
    if (parts.scheme != 'https' or not host or parts.username is not None or parts.password is not None
            or port not in (None, 443) or '%' in host or host == 'localhost'
            or host.endswith(('.localhost', '.local', '.internal', '.test', '.invalid'))):
        raise ResearchError('WEB_URL_BLOCKED', 'Only public HTTPS sources on port 443 are supported.', 422)
    try:
        ipaddress.ip_address(host)
    except ValueError:
        if '.' not in host or not re.fullmatch(r'[a-z0-9.-]+', host):
            raise ResearchError('WEB_URL_BLOCKED', 'The source hostname is not supported.', 422)
    else:
        public_ip(host)
    authority = f'[{host}]' if ':' in host else host
    # Fragments are not fetched and must not create duplicate source identities.
    return urlunsplit(('https', authority, parts.path or '/', parts.query, ''))


class PublicNetworkBackend(httpcore.AsyncNetworkBackend):
    def __init__(self):
        self.backend = httpcore.AnyIOBackend()

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        if port != 443:
            raise ResearchError('WEB_ADDRESS_BLOCKED', 'Only HTTPS connections are permitted.', 422)
        async with asyncio.timeout(timeout or 5):
            rows = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
            # Reject mixed public/private DNS answers too. The original hostname is
            # never passed to connect_tcp again; HTTPCore retains it for TLS/SNI.
            addresses = list(dict.fromkeys(public_ip(row[4][0]) for row in rows))
            if not addresses:
                raise ResearchError('WEB_DNS_FAILED', 'The public source could not be resolved.')
            last_error = None
            for address in addresses[:4]:
                try:
                    return await self.backend.connect_tcp(address, port, timeout, socket_options=socket_options)
                except (httpcore.ConnectError, httpcore.ConnectTimeout) as error:
                    last_error = error
            raise last_error

    async def connect_unix_socket(self, *args, **kwargs):
        raise ResearchError('WEB_ADDRESS_BLOCKED', 'Local sockets are not supported.', 422)

    async def sleep(self, seconds):
        await asyncio.sleep(seconds)


@dataclass
class Download:
    body: bytes
    media_type: str


class PublicHTTP:
    """One scoped pool per research attempt; redirects and decompression disabled."""
    def __init__(self, settings, pool=None):
        self.settings = settings
        self.pool = pool or httpcore.AsyncConnectionPool(
            network_backend=PublicNetworkBackend(), max_connections=2,
            max_keepalive_connections=0, retries=0,
        )

    async def __aenter__(self):
        await self.pool.__aenter__()
        return self

    async def __aexit__(self, *args):
        return await self.pool.__aexit__(*args)

    async def get(self, url, *, api_key=None):
        url = public_url(url, max_length=8192 if api_key else 2048)
        headers = {'Accept-Encoding': 'identity', 'User-Agent': 'VoltForge-Research/1.0',
                   'Accept': 'application/json' if api_key else 'text/html, text/plain'}
        if api_key:
            # Secrets can only be sent to the fixed provider endpoint. Never to
            # a search result, redirect target or configurable endpoint URL.
            if urlsplit(url)._replace(query='', fragment='').geturl() != BRAVE_URL:
                raise ResearchError('WEB_CREDENTIAL_SCOPE', 'Invalid search-provider destination.', 422)
            headers['X-Subscription-Token'] = api_key
        seconds = self.settings.web_page_timeout_seconds
        try:
            async with asyncio.timeout(seconds):
                async with self.pool.stream('GET', url, headers=headers,
                        extensions={'timeout': dict(connect=seconds, read=seconds, write=seconds, pool=seconds)}) as response:
                    if response.status == 429:
                        raise ResearchError('WEB_RATE_LIMITED', 'The source is rate limited. Retry later.', 429)
                    if response.status in (401, 403) and api_key:
                        raise ResearchError('WEB_PROVIDER_AUTH', 'Search-provider credentials or subscription were rejected.')
                    if 300 <= response.status < 400:
                        raise ResearchError('WEB_REDIRECT_BLOCKED', 'Redirected sources are not fetched in this release.', 422)
                    if response.status != 200:
                        raise ResearchError('WEB_UPSTREAM_FAILED' if response.status >= 500 else 'WEB_SOURCE_UNAVAILABLE',
                                            'The public source could not be read.')
                    metadata = {key.decode('ascii').lower(): value.decode('latin-1') for key, value in response.headers}
                    if metadata.get('content-encoding', 'identity').lower() != 'identity':
                        raise ResearchError('WEB_ENCODING_UNSUPPORTED', 'Compressed responses are not accepted.', 422)
                    limit = self.settings.web_max_response_bytes
                    try:
                        if int(metadata.get('content-length', '0')) > limit:
                            raise ResearchError('WEB_SOURCE_TOO_LARGE', 'The source exceeds the download limit.', 422)
                    except ValueError:
                        raise ResearchError('WEB_RESPONSE_INVALID', 'The source sent invalid response headers.', 422) from None
                    media = metadata.get('content-type', '').split(';')[0].strip().lower()
                    allowed = ('application/json',) if api_key else ('text/html', 'text/plain')
                    if media not in allowed:
                        raise ResearchError('WEB_CONTENT_UNSUPPORTED', 'Only supported text sources can be read.', 422)
                    body = bytearray()
                    async for chunk in response.aiter_stream():
                        if len(body) + len(chunk) > limit:
                            raise ResearchError('WEB_SOURCE_TOO_LARGE', 'The source exceeds the download limit.', 422)
                        body.extend(chunk)
                    return Download(bytes(body), media)
        except TimeoutError:
            raise ResearchError('WEB_TIMEOUT', 'The public source timed out.', 504) from None
        except (httpcore.NetworkError, httpcore.TimeoutException, httpcore.ProtocolError, OSError):
            raise ResearchError('WEB_NETWORK_FAILED', 'The public source could not be reached.') from None


class TextExtractor(HTMLParser):
    IGNORED = {'script', 'style', 'noscript', 'svg', 'head', 'nav', 'footer', 'form', 'template'}
    def __init__(self, limit=8000):
        super().__init__(convert_charrefs=True)
        self.ignored = []
        self.parts = []
        self.remaining = limit

    def handle_starttag(self, tag, attrs):
        if tag in self.IGNORED:
            self.ignored.append(tag)

    def handle_endtag(self, tag):
        if self.ignored and tag == self.ignored[-1]:
            self.ignored.pop()

    def handle_data(self, data):
        if not self.ignored and self.remaining:
            text = ' '.join(data.split())[:self.remaining]
            if text:
                self.parts.append(text)
                self.remaining = max(0, self.remaining - len(text) - 1)


def extract_text(value, limit=8000):
    parser = TextExtractor(limit)
    parser.feed(value)
    parser.close()
    return ' '.join(parser.parts)[:limit]


def source(url, title, excerpt, kind, raw):
    return ResearchSource(
        sourceId='web-' + hashlib.sha256(url.encode()).hexdigest()[:24],
        kind=kind, title=title[:300] or urlsplit(url).hostname, url=url,
        retrievedAt=datetime.now(timezone.utc), excerpt=excerpt[:8000],
        contentHash=hashlib.sha256(raw).hexdigest(),
    )


class BraveSearch:
    def __init__(self, settings, http, before_retry=None):
        self.settings, self.http, self.before_retry = settings, http, before_retry

    async def search(self, query):
        # Exactly one reviewed public query. No history, project data or memory.
        url = BRAVE_URL + '?' + urlencode({'q': query, 'count': self.settings.web_max_results,
                                           'text_decorations': 'false', 'safesearch': 'moderate',
                                           'result_filter': 'web', 'spellcheck': 'false'})
        for attempt in range(2):
            try:
                result = await self.http.get(url, api_key=self.settings.brave_api_key.get_secret_value())
                break
            except ResearchError as error:
                if attempt or error.code not in ('WEB_NETWORK_FAILED', 'WEB_UPSTREAM_FAILED'):
                    raise
                await asyncio.sleep(self.settings.web_min_search_interval_seconds)
                if self.before_retry:
                    self.before_retry()
        try:
            data = json.loads(result.body)
            rows = data.get('web', {}).get('results', [])
            if not isinstance(rows, list):
                raise ValueError()
        except (ValueError, AttributeError, TypeError, RecursionError):
            raise ResearchError('WEB_PROVIDER_RESPONSE_INVALID', 'The search provider returned an invalid response.') from None
        sources, seen = [], set()
        for row in rows[:self.settings.web_max_results]:
            if not isinstance(row, dict):
                continue
            try:
                url = public_url(row.get('url'))
            except ResearchError:
                continue
            title, description = row.get('title', ''), row.get('description', '')
            if not isinstance(title, str) or not isinstance(description, str) or url in seen:
                continue
            text = extract_text(description, 2000)
            if not text:
                continue
            seen.add(url)
            sources.append(source(url, extract_text(title, 300), text, 'search_snippet', text.encode()))
        # Prefer original technical publishers, without asserting factual authority.
        # Preserve provider order within each group; unknown publishers remain usable.
        def publisher_priority(item):
            host = urlsplit(str(item.url)).hostname
            return not any(host == domain or host.endswith('.' + domain) for domain in PRIMARY_TECHNICAL_DOMAINS)
        return sorted(sources, key=publisher_priority)


class WebResearch:
    def __init__(self, settings_provider=get_settings, http_factory=PublicHTTP):
        self.settings_provider, self.http_factory = settings_provider, http_factory
        self.lock = threading.Lock()
        self.active = 0
        self.next_search = 0.0

    def reserve_retry(self, settings):
        with self.lock:
            now = time.monotonic()
            if now < self.next_search:
                raise ResearchError('WEB_BUSY', 'Research capacity is busy. Retry shortly.', 429)
            self.next_search = now + settings.web_min_search_interval_seconds

    async def search(self, request: ResearchRequest):
        settings = self.settings_provider()
        # Documents have their own future ownership-validated retrieval pipeline.
        if request.documentIds:
            raise ResearchError('AI_DOCUMENTS_UNAVAILABLE', 'Document retrieval is not available.')
        if request.mode == 'off':
            return {'status': 'disabled', 'sources': [], 'warnings': []}
        if not settings.web_search_enabled:
            raise ResearchError('WEB_DISABLED', 'Web research is disabled by server configuration.')
        if not settings.brave_api_key.get_secret_value().strip():
            raise ResearchError('WEB_NOT_CONFIGURED', 'A search-provider credential must be configured.')
        with self.lock:
            now = time.monotonic()
            if self.active >= settings.web_max_concurrent or now < self.next_search:
                raise ResearchError('WEB_BUSY', 'Research capacity is busy. Retry shortly.', 429)
            self.active += 1
            self.next_search = now + settings.web_min_search_interval_seconds
        try:
            async with asyncio.timeout(settings.web_timeout_seconds):
                async with self.http_factory(settings) as http:
                    sources = await BraveSearch(settings, http, lambda: self.reserve_retry(settings)).search(request.publicQuery)
                    warnings = []
                    for index, item in enumerate(sources[:settings.web_max_pages]):
                        try:
                            page = await http.get(str(item.url))
                            decoded = page.body.decode('utf-8', errors='replace')
                            text = extract_text(decoded) if page.media_type == 'text/html' else decoded.strip()[:8000]
                            if not text:
                                raise ResearchError('WEB_SOURCE_EMPTY', 'No readable text was found.', 422)
                            sources[index] = source(str(item.url), item.title, text, 'web_page', page.body)
                        except ResearchError as error:
                            # Retain the snippet as a snippet; never call a failed fetch a page.
                            warnings.append({'sourceId': item.sourceId, 'code': error.code})
                    return {'status': 'partial' if warnings else 'complete' if sources else 'no_results',
                            'sources': [s.model_dump(mode='json') for s in sources], 'warnings': warnings,
                            'evidencePolicy': 'untrusted_reference_only'}
        except ResearchError as error:
            if error.code == 'WEB_RATE_LIMITED':
                with self.lock:
                    self.next_search = max(self.next_search, time.monotonic() + 60)
            raise
        except TimeoutError:
            raise ResearchError('WEB_DEADLINE', 'Research exceeded its time limit. Retry with a narrower query.', 504) from None
        finally:
            with self.lock:
                self.active -= 1


web_research = WebResearch()
