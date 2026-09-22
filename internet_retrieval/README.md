# Secure optional internet evidence retrieval

VFAI-023 is an opt-in evidence tool. It is not a model, is never required by
local generation, and cannot override deterministic engineering findings.

The checked policy permits one internally constructed HTTPS request to the
DuckDuckGo Instant Answer JSON API. User input supplies only bounded query
text; it can never select a URL, host, path, port, redirect, HTTP method, or
content type. The client resolves the approved host before every request,
rejects every non-global address, ignores environment proxies, rejects all
redirects, streams at most 256 KiB, accepts JSON content types only, and makes
no retries. Result URLs are citations only and are never fetched by the AI
service.

Automatic retrieval is allowed only when the curated local index reports
`no-results` or `unavailable` and the request contains a documented evidence
intent such as `datasheet`, `pinout`, `specifications`, `release notes`, or
`library api`. Explicit phrases such as `search the web` also trigger it. All
other requests return `not-requested` without touching DNS or HTTP. When the
feature flag is off, a triggered request returns the explicit `disabled`
state. Provider, DNS, timeout, format, sanitization, and size failures return a
content-free `degraded` state; local tools and local generation continue.

Only sanitized excerpts are cached, in memory, for a bounded TTL and entry
count. Raw queries are hashed rather than persisted or logged. Raw provider
payloads and project context are not stored or sent. Markup/control characters
are removed and results containing instruction-like prompt injection are
discarded. Every accepted item carries its retrieval timestamp, content hash,
source URL/domain, `retrieved` authority, and explicit untrusted/not-for-
training flags.

No retrieved web content may enter `electronics_corpus`, a training shard, or
a model dataset through this feature. Corpus admission requires a separate
source-provenance and licensing approval workflow outside VFAI-023.
