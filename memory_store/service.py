"""Authenticated, bounded, revision-aware SQLite memory store."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import threading
from typing import Any, Callable, Iterator, Mapping

from memory_store.schema import (
    CONTRACT_VERSION,
    POLICY_ID,
    MemoryContractError,
    MemoryCorrectionRequest,
    MemoryEntry,
    MemoryState,
    MemoryWriteRequest,
    checked_state_schema,
    load_policy,
)


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@/+\-]{0,159}$")
_WHITESPACE = re.compile(r"\s+")
_INJECTION = re.compile(
    r"(?:ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions?|"
    r"reveal\s+(?:the\s+)?(?:system\s+prompt|hidden\s+reasoning)|"
    r"(?:system|developer)\s+message\s*:|<\|(?:system|assistant|developer)\|>|"
    r"\[/?inst\]|<script\b|follow\s+these\s+instructions?)",
    re.IGNORECASE,
)
_HIDDEN_REASONING = re.compile(
    r"(?:<\/?think>|chain[- ]of[- ]thought|hidden\s+reasoning|internal\s+reasoning|scratchpad)",
    re.IGNORECASE,
)
_SNAPSHOT_TERMS = re.compile(
    r'"(?:components|wires|canvasData|netlist|simulationState|codeFiles)"\s*:',
    re.IGNORECASE,
)
_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.IGNORECASE | re.DOTALL),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE),
    re.compile(r"\b(?:sk-[A-Za-z0-9_-]{12,}|github_pat_[A-Za-z0-9_]{12,}|AIza[A-Za-z0-9_-]{20,})\b"),
    re.compile(r"\b(?:password|passwd|pwd|api[_-]?key|access[_-]?token|secret)\s*[:=]\s*[^\s,;]{4,}", re.IGNORECASE),
    re.compile(r"\b(?:mysql|postgres(?:ql)?|mongodb)://[^\s:@/]+:[^\s@/]+@", re.IGNORECASE),
)
_TURN_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_+.-]{2,}")
_TURN_STOP = {
    "about", "assistant", "build", "could", "explain", "from", "have", "keep",
    "local", "please", "private", "project", "response", "should", "that", "testing",
    "this", "use", "using", "value", "with",
}


class MemoryStoreError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class MemoryScope:
    owner_key: str
    project_key: str
    session_key: str | None

    @classmethod
    def from_identifiers(
        cls, user_id: str, project_id: str, session_id: str | None = None
    ) -> "MemoryScope":
        user = _validated_identifier(user_id, "user")
        project = _validated_identifier(project_id, "project")
        session = (
            _validated_identifier(session_id, "session") if session_id else None
        )
        owner_key = _scope_hash("owner", user)
        project_key = _scope_hash("project", f"{owner_key}:{project}")
        session_key = (
            _scope_hash("session", f"{owner_key}:{project_key}:{session}")
            if session
            else None
        )
        return cls(owner_key, project_key, session_key)


def _validated_identifier(value: str, label: str) -> str:
    cleaned = str(value or "").strip()
    if not _IDENTIFIER.fullmatch(cleaned):
        raise MemoryStoreError(
            "MEMORY_IDENTITY_INVALID", f"The authenticated {label} identity is invalid."
        )
    return cleaned


def _scope_hash(domain: str, value: str) -> str:
    return hashlib.sha256(f"vfai025:{domain}:v1:{value}".encode("utf-8")).hexdigest()


def _timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _content_sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sanitize_content(value: str, maximum: int) -> tuple[str, bool]:
    content = _WHITESPACE.sub(" ", str(value or "")).strip()
    if not content:
        raise MemoryStoreError("MEMORY_CONTENT_EMPTY", "Memory content cannot be empty.")
    if len(content) > maximum:
        raise MemoryStoreError(
            "MEMORY_CONTENT_TOO_LARGE", "Memory content exceeds the bounded limit."
        )
    if _INJECTION.search(content):
        raise MemoryStoreError(
            "MEMORY_INSTRUCTION_CONTENT_REJECTED",
            "Instruction-like or prompt-injection content cannot be remembered.",
        )
    if _HIDDEN_REASONING.search(content):
        raise MemoryStoreError(
            "MEMORY_HIDDEN_REASONING_REJECTED",
            "Hidden reasoning cannot be remembered.",
        )
    if _SNAPSHOT_TERMS.search(content):
        raise MemoryStoreError(
            "MEMORY_PROJECT_SNAPSHOT_REJECTED",
            "Full project snapshots cannot be remembered.",
        )
    if content[:1] in {"{", "["}:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, (dict, list)):
            raise MemoryStoreError(
                "MEMORY_PROJECT_SNAPSHOT_REJECTED",
                "Structured project snapshots cannot be remembered.",
            )
    redacted = content
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    redacted = _WHITESPACE.sub(" ", redacted).strip()
    if not redacted or redacted == "[REDACTED]":
        raise MemoryStoreError(
            "MEMORY_SECRET_ONLY_REJECTED",
            "Secret-only content cannot be remembered.",
        )
    return redacted, redacted != content


class BoundedMemoryStore:
    def __init__(
        self,
        database_path: str | Path,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.path = Path(database_path).resolve()
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.policy = load_policy()
        self._lock = threading.RLock()
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS memory_preferences (
                    owner_key TEXT NOT NULL,
                    project_key TEXT NOT NULL,
                    enabled INTEGER NOT NULL CHECK (enabled IN (0, 1)),
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (owner_key, project_key)
                );
                CREATE TABLE IF NOT EXISTS memory_entries (
                    memory_id TEXT PRIMARY KEY,
                    owner_key TEXT NOT NULL,
                    project_key TEXT NOT NULL,
                    session_key TEXT,
                    scope TEXT NOT NULL CHECK (scope IN ('project', 'session')),
                    kind TEXT NOT NULL CHECK (kind IN ('fact', 'decision', 'summary', 'recent-turn')),
                    content TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    project_revision TEXT NOT NULL,
                    version INTEGER NOT NULL CHECK (version >= 1),
                    source TEXT NOT NULL CHECK (source IN ('user-approved', 'recent-turn-summary')),
                    redaction_applied INTEGER NOT NULL CHECK (redaction_applied IN (0, 1)),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    CHECK ((scope = 'project' AND session_key IS NULL) OR
                           (scope = 'session' AND session_key IS NOT NULL))
                );
                CREATE INDEX IF NOT EXISTS idx_memory_scope
                    ON memory_entries(owner_key, project_key, session_key, expires_at);
                CREATE INDEX IF NOT EXISTS idx_memory_eviction
                    ON memory_entries(owner_key, project_key, kind, created_at, memory_id);
                PRAGMA user_version = 1;
                """
            )
        try:
            self.path.chmod(0o600)
        except OSError:
            pass

    def is_enabled(self, scope: MemoryScope) -> bool:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT enabled FROM memory_preferences WHERE owner_key = ? AND project_key = ?",
                (scope.owner_key, scope.project_key),
            ).fetchone()
        return bool(row and row["enabled"])

    def set_enabled(
        self, scope: MemoryScope, enabled: bool, *, clear_on_disable: bool = False
    ) -> MemoryState:
        now = _timestamp(self.clock())
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO memory_preferences(owner_key, project_key, enabled, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(owner_key, project_key) DO UPDATE SET
                    enabled = excluded.enabled,
                    updated_at = excluded.updated_at
                """,
                (scope.owner_key, scope.project_key, int(enabled), now),
            )
            if not enabled and clear_on_disable:
                connection.execute(
                    "DELETE FROM memory_entries WHERE owner_key = ? AND project_key = ?",
                    (scope.owner_key, scope.project_key),
                )
        return self.inspect(scope)

    def create(
        self, scope: MemoryScope, request: MemoryWriteRequest
    ) -> tuple[MemoryEntry, int]:
        if not self.is_enabled(scope):
            raise MemoryStoreError(
                "MEMORY_DISABLED", "Memory must be explicitly enabled before storing entries."
            )
        if request.scope == "session" and scope.session_key is None:
            raise MemoryStoreError(
                "MEMORY_SESSION_REQUIRED", "Session-scoped memory requires a session identity."
            )
        maximum = int(self.policy["limits"]["maximumContentCharacters"])
        content, redacted = _sanitize_content(request.content, maximum)
        entry, evicted = self._insert(
            scope,
            kind=request.kind,
            entry_scope=request.scope,
            content=content,
            project_revision=request.projectRevision,
            expires_in_hours=request.expiresInHours,
            source="user-approved",
            redacted=redacted,
        )
        return entry, evicted

    def _insert(
        self,
        scope: MemoryScope,
        *,
        kind: str,
        entry_scope: str,
        content: str,
        project_revision: str,
        expires_in_hours: int | None,
        source: str,
        redacted: bool,
    ) -> tuple[MemoryEntry, int]:
        revision = _validated_identifier(project_revision, "project revision")
        now_value = self.clock().astimezone(timezone.utc)
        now = _timestamp(now_value)
        ttl = self._ttl(kind, expires_in_hours)
        expires = _timestamp(now_value + timedelta(hours=ttl))
        session_key = scope.session_key if entry_scope == "session" else None
        with self._lock, self._connect() as connection:
            sequence = int(
                connection.execute(
                    "SELECT COALESCE(MAX(rowid), 0) + 1 AS next_id FROM memory_entries"
                ).fetchone()["next_id"]
            )
            identity = (
                f"{scope.owner_key}:{scope.project_key}:{session_key}:{kind}:"
                f"{_content_sha(content)}:{revision}:{now}:{sequence}"
            )
            memory_id = f"memory:v1:{hashlib.sha256(identity.encode()).hexdigest()[:24]}"
            connection.execute(
                """
                INSERT INTO memory_entries(
                    memory_id, owner_key, project_key, session_key, scope, kind,
                    content, content_sha256, project_revision, version, source,
                    redaction_applied, created_at, updated_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    scope.owner_key,
                    scope.project_key,
                    session_key,
                    entry_scope,
                    kind,
                    content,
                    _content_sha(content),
                    revision,
                    source,
                    int(redacted),
                    now,
                    now,
                    expires,
                ),
            )
            evicted = self._enforce_limits(connection, scope, now)
            row = connection.execute(
                "SELECT * FROM memory_entries WHERE memory_id = ?", (memory_id,)
            ).fetchone()
            if row is None:
                raise MemoryStoreError(
                    "MEMORY_ENTRY_EVICTED", "The new memory entry exceeded storage limits."
                )
        return self._entry(row, revision), evicted

    def correct(
        self,
        scope: MemoryScope,
        memory_id: str,
        request: MemoryCorrectionRequest,
    ) -> MemoryEntry:
        content, redacted = _sanitize_content(
            request.content, int(self.policy["limits"]["maximumContentCharacters"])
        )
        revision = _validated_identifier(request.projectRevision, "project revision")
        now_value = self.clock().astimezone(timezone.utc)
        now = _timestamp(now_value)
        with self._lock, self._connect() as connection:
            row = self._owned_entry(connection, scope, memory_id)
            if row is None:
                raise MemoryStoreError("MEMORY_NOT_FOUND", "The memory entry was not found.")
            if int(row["version"]) != request.expectedVersion:
                raise MemoryStoreError(
                    "MEMORY_VERSION_CONFLICT",
                    "The memory entry changed; inspect it before correcting again.",
                )
            expires = _timestamp(
                now_value
                + timedelta(hours=self._ttl(str(row["kind"]), request.expiresInHours))
            )
            connection.execute(
                """
                UPDATE memory_entries SET content = ?, content_sha256 = ?,
                    project_revision = ?, version = version + 1,
                    redaction_applied = ?, updated_at = ?, expires_at = ?
                WHERE memory_id = ?
                """,
                (
                    content,
                    _content_sha(content),
                    revision,
                    int(redacted),
                    now,
                    expires,
                    memory_id,
                ),
            )
            updated = connection.execute(
                "SELECT * FROM memory_entries WHERE memory_id = ?", (memory_id,)
            ).fetchone()
        return self._entry(updated, revision)

    def delete(self, scope: MemoryScope, memory_id: str) -> bool:
        with self._lock, self._connect() as connection:
            row = self._owned_entry(connection, scope, memory_id)
            if row is None:
                return False
            connection.execute(
                "DELETE FROM memory_entries WHERE memory_id = ?", (memory_id,)
            )
        return True

    def clear(self, scope: MemoryScope, clear_scope: str) -> int:
        with self._lock, self._connect() as connection:
            if clear_scope == "session":
                if scope.session_key is None:
                    raise MemoryStoreError(
                        "MEMORY_SESSION_REQUIRED",
                        "Clearing session memory requires a session identity.",
                    )
                cursor = connection.execute(
                    """
                    DELETE FROM memory_entries
                    WHERE owner_key = ? AND project_key = ? AND session_key = ?
                    """,
                    (scope.owner_key, scope.project_key, scope.session_key),
                )
            elif clear_scope == "project":
                cursor = connection.execute(
                    "DELETE FROM memory_entries WHERE owner_key = ? AND project_key = ?",
                    (scope.owner_key, scope.project_key),
                )
            else:
                raise MemoryStoreError(
                    "MEMORY_CLEAR_SCOPE_INVALID", "Memory clear scope is invalid."
                )
        return max(0, int(cursor.rowcount))

    def inspect(
        self,
        scope: MemoryScope,
        project_revision: str | None = None,
        *,
        context_only: bool = False,
        evicted_count: int = 0,
    ) -> MemoryState:
        revision = (
            _validated_identifier(project_revision, "project revision")
            if project_revision
            else None
        )
        now = _timestamp(self.clock())
        with self._lock, self._connect() as connection:
            purged = self._purge_expired(connection, scope, now)
            rows = self._visible_rows(connection, scope)
        entries = [self._entry(row, revision) for row in rows]
        if context_only:
            entries = [item for item in entries if not item.staleForProjectRevision]
            entries, omitted = self._bounded_context(entries)
        else:
            omitted = 0
        enabled = self.is_enabled(scope)
        return MemoryState(
            policySha256=self.policy["policySha256"],
            status="ready" if enabled else "disabled",
            reasonCode="MEMORY_READY" if enabled else "MEMORY_DISABLED",
            enabled=enabled,
            projectRevision=revision,
            entries=entries,
            entryCount=len(entries),
            staleEntryCount=sum(item.staleForProjectRevision for item in entries),
            omittedEntryCount=omitted,
            expiredPurgedCount=purged,
            evictedCount=evicted_count,
            totalBytes=sum(len(item.content.encode("utf-8")) for item in entries),
        )

    def context_entries(
        self, scope: MemoryScope, project_revision: str | None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not self.is_enabled(scope) or not project_revision:
            state = self.inspect(scope, project_revision, context_only=True)
            return [], self.public_metadata(state)
        state = self.inspect(scope, project_revision, context_only=True)
        context = [
            {
                "id": item.memoryId,
                "kind": item.kind,
                "content": item.content[: int(self.policy["limits"]["maximumContextContentCharacters"])],
                "projectRevision": item.projectRevision,
                "version": item.version,
                "source": item.source,
                "authority": item.authority,
                "modelEvidenceAllowed": False,
                "trainingUseAllowed": False,
            }
            for item in state.entries
        ]
        return context, self.public_metadata(state)

    def record_turn(
        self,
        scope: MemoryScope,
        project_revision: str | None,
        user_message: str,
        assistant_reply: str,
    ) -> bool:
        if not self.is_enabled(scope) or scope.session_key is None or not project_revision:
            return False
        try:
            bounded_user, redacted = _sanitize_content(
                _WHITESPACE.sub(" ", user_message).strip()[:600], 600
            )
            terms: list[str] = []
            for token in _TURN_TOKEN.findall(bounded_user):
                normalized = token.casefold()
                if normalized in _TURN_STOP or normalized == "redacted" or normalized in terms:
                    continue
                terms.append(normalized)
                if len(terms) >= 16:
                    break
            if not terms:
                return False
            lowered = assistant_reply.casefold()
            categories = ["engineering-guidance"]
            if "[citation:" in lowered:
                categories.append("cited")
            if any(value in lowered for value in ("cannot verify", "uncertain", "unknown")):
                categories.append("uncertain")
            if "evidence conflicts" in lowered:
                categories.append("conflicted")
            if any(value in lowered for value in ("warning", "blocking", "unsafe")):
                categories.append("safety")
            if "```" in assistant_reply:
                categories.append("code")
            content = (
                f"Recent topic terms: {', '.join(terms)}. "
                f"Outcome categories: {', '.join(categories)}."
            )
            self._insert(
                scope,
                kind="recent-turn",
                entry_scope="session",
                content=content,
                project_revision=project_revision,
                expires_in_hours=None,
                source="recent-turn-summary",
                redacted=redacted,
            )
        except MemoryStoreError:
            return False
        return True

    def public_metadata(self, state: MemoryState) -> dict[str, Any]:
        return {
            "policyId": POLICY_ID,
            "contractVersion": CONTRACT_VERSION,
            "status": state.status,
            "reasonCode": state.reasonCode,
            "enabled": state.enabled,
            "selectedEntryCount": state.entryCount,
            "staleEntryCount": state.staleEntryCount,
            "omittedEntryCount": state.omittedEntryCount,
            "expiredPurgedCount": state.expiredPurgedCount,
            "evictedCount": state.evictedCount,
            "trainingUseAllowed": False,
            "rawIdentifiersStored": False,
            "rawContentStoredInMetadata": False,
        }

    def _ttl(self, kind: str, requested: int | None) -> int:
        default = int(self.policy["kinds"][kind]["defaultTtlHours"])
        maximum = int(self.policy["limits"]["maximumTtlHours"])
        minimum = int(self.policy["limits"]["minimumTtlHours"])
        return min(maximum, max(minimum, int(requested or default)))

    def _visible_rows(
        self, connection: sqlite3.Connection, scope: MemoryScope
    ) -> list[sqlite3.Row]:
        if scope.session_key:
            return list(
                connection.execute(
                    """
                    SELECT * FROM memory_entries
                    WHERE owner_key = ? AND project_key = ?
                      AND (scope = 'project' OR session_key = ?)
                    ORDER BY kind, created_at, memory_id
                    """,
                    (scope.owner_key, scope.project_key, scope.session_key),
                ).fetchall()
            )
        return list(
            connection.execute(
                """
                SELECT * FROM memory_entries
                WHERE owner_key = ? AND project_key = ? AND scope = 'project'
                ORDER BY kind, created_at, memory_id
                """,
                (scope.owner_key, scope.project_key),
            ).fetchall()
        )

    def _owned_entry(
        self, connection: sqlite3.Connection, scope: MemoryScope, memory_id: str
    ) -> sqlite3.Row | None:
        row = connection.execute(
            """
            SELECT * FROM memory_entries
            WHERE memory_id = ? AND owner_key = ? AND project_key = ?
            """,
            (memory_id, scope.owner_key, scope.project_key),
        ).fetchone()
        if row is None:
            return None
        if row["scope"] == "session" and row["session_key"] != scope.session_key:
            return None
        return row

    def _purge_expired(
        self, connection: sqlite3.Connection, scope: MemoryScope, now: str
    ) -> int:
        cursor = connection.execute(
            """
            DELETE FROM memory_entries
            WHERE owner_key = ? AND project_key = ? AND expires_at <= ?
            """,
            (scope.owner_key, scope.project_key, now),
        )
        return max(0, int(cursor.rowcount))

    def _enforce_limits(
        self, connection: sqlite3.Connection, scope: MemoryScope, now: str
    ) -> int:
        evicted = self._purge_expired(connection, scope, now)
        maximum_session = int(self.policy["limits"]["maximumSessionEntries"])
        if scope.session_key:
            session_rows = connection.execute(
                """
                SELECT memory_id FROM memory_entries
                WHERE owner_key = ? AND project_key = ? AND session_key = ?
                ORDER BY created_at, memory_id
                """,
                (scope.owner_key, scope.project_key, scope.session_key),
            ).fetchall()
            for row in session_rows[: max(0, len(session_rows) - maximum_session)]:
                connection.execute(
                    "DELETE FROM memory_entries WHERE memory_id = ?", (row["memory_id"],)
                )
                evicted += 1
        maximum_entries = int(self.policy["limits"]["maximumProjectEntries"])
        maximum_bytes = int(self.policy["limits"]["maximumProjectBytes"])
        while True:
            aggregate = connection.execute(
                """
                SELECT COUNT(*) AS count, COALESCE(SUM(LENGTH(CAST(content AS BLOB))), 0) AS bytes
                FROM memory_entries WHERE owner_key = ? AND project_key = ?
                """,
                (scope.owner_key, scope.project_key),
            ).fetchone()
            if int(aggregate["count"]) <= maximum_entries and int(aggregate["bytes"]) <= maximum_bytes:
                break
            victim = connection.execute(
                """
                SELECT memory_id FROM memory_entries
                WHERE owner_key = ? AND project_key = ?
                ORDER BY CASE kind
                    WHEN 'recent-turn' THEN 1 WHEN 'summary' THEN 2
                    WHEN 'decision' THEN 3 ELSE 4 END,
                    created_at, memory_id
                LIMIT 1
                """,
                (scope.owner_key, scope.project_key),
            ).fetchone()
            if victim is None:
                break
            connection.execute(
                "DELETE FROM memory_entries WHERE memory_id = ?", (victim["memory_id"],)
            )
            evicted += 1
        return evicted

    def _bounded_context(
        self, entries: list[MemoryEntry]
    ) -> tuple[list[MemoryEntry], int]:
        maximum_entries = int(self.policy["limits"]["maximumContextEntries"])
        maximum_bytes = int(self.policy["limits"]["maximumContextBytes"])
        priority = {"fact": 1, "decision": 2, "summary": 3, "recent-turn": 4}
        selected: list[MemoryEntry] = []
        total = 0
        for entry in sorted(
            entries,
            key=lambda item: (priority[item.kind], item.updatedAt, item.memoryId),
        ):
            size = len(entry.content.encode("utf-8"))
            if len(selected) >= maximum_entries or total + size > maximum_bytes:
                continue
            selected.append(entry)
            total += size
        return selected, len(entries) - len(selected)

    @staticmethod
    def _entry(row: sqlite3.Row, revision: str | None) -> MemoryEntry:
        return MemoryEntry(
            memoryId=row["memory_id"],
            kind=row["kind"],
            scope=row["scope"],
            content=row["content"],
            contentSha256=row["content_sha256"],
            projectRevision=row["project_revision"],
            version=row["version"],
            source=row["source"],
            createdAt=row["created_at"],
            updatedAt=row["updated_at"],
            expiresAt=row["expires_at"],
            staleForProjectRevision=bool(
                revision and row["project_revision"] != revision
            ),
            redactionApplied=bool(row["redaction_applied"]),
        )


@lru_cache(maxsize=4)
def get_memory_store(database_path: str) -> BoundedMemoryStore:
    return BoundedMemoryStore(database_path)


def memory_health(
    database_path: str | Path, *, authenticated_gateway_configured: bool
) -> dict[str, Any]:
    try:
        policy = load_policy()
        checked_state_schema()
        BoundedMemoryStore(database_path)
    except (MemoryContractError, MemoryStoreError, OSError, sqlite3.Error) as error:
        return {
            "ready": False,
            "status": "unavailable",
            "code": getattr(error, "code", "MEMORY_STORE_UNAVAILABLE"),
            "policyId": POLICY_ID,
            "contractVersion": CONTRACT_VERSION,
        }
    return {
        "ready": bool(authenticated_gateway_configured),
        "status": "ready" if authenticated_gateway_configured else "authentication-required",
        "code": "MEMORY_READY" if authenticated_gateway_configured else "MEMORY_GATEWAY_AUTH_REQUIRED",
        "policyId": policy["policyId"],
        "policySha256": policy["policySha256"],
        "contractVersion": CONTRACT_VERSION,
        "defaultEnabled": False,
        "trainingUseAllowed": False,
        "rawIdentifiersStored": False,
    }
