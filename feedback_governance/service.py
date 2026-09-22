"""Fail-closed feedback admission and reproducible retraining preparation.

Feedback is never a training shortcut. The service stores only bounded,
explicitly approved evidence plus safe release identities. A reviewer must
admit a regression to the protected held-out set before a separately consented
and reviewed, non-leaking training example can be scheduled. Scheduling writes
candidate inputs and provenance only; it never loads or mutates model weights.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import threading
from typing import Any, Callable, Iterator, Mapping, Sequence

from evaluation.leakage import HeldOutRegistry, exclude_held_out_records, get_held_out_registry, load_cases
from task_schema.schema import TaskContractError, validate_task_record


AI_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = Path(__file__).resolve().parent / "policy.v1.json"
DEFAULT_RUNTIME_ROOT = AI_ROOT / "runtime" / "feedback-governance-v1"
DEFAULT_DATABASE_PATH = DEFAULT_RUNTIME_ROOT / "feedback.sqlite3"
POLICY_ID = "vfai034-feedback-governance-v1"
SCHEMA_VERSION = 1
FEEDBACK_KINDS = frozenset({"useful", "incorrect", "unsafe"})
FEEDBACK_STATES = frozenset({"pending-review", "rejected", "held-out", "training-approved"})
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{2,159}$")
ARTIFACT_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._:-]{2,199}$")
SHA256 = re.compile(r"^[a-f0-9]{64}$")
SECRET = re.compile(
    r"(?:-----BEGIN [A-Z ]*PRIVATE KEY-----|\bBearer\s+|\b(?:password|passwd|pwd|api[_-]?key|access[_-]?token|secret)\s*[:=]|\b(?:sk-|github_pat_|AIza))",
    re.IGNORECASE,
)
INSTRUCTION = re.compile(
    r"(?:ignore\s+(?:all\s+)?previous|system\s+message|developer\s+message|hidden\s+reasoning|chain[- ]of[- ]thought|auto(?:matically)?\s+(?:apply|execute))",
    re.IGNORECASE,
)
SNAPSHOT = re.compile(
    r"(?:full\s+(?:project|conversation|chat)\s+snapshot|<voltforge_project_data>|\[SYS\]|\[USER\]|\[ASSISTANT\])",
    re.IGNORECASE,
)
WHITESPACE = re.compile(r"\s+")


class FeedbackGovernanceError(RuntimeError):
    """Precise, public-safe feedback and retraining failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha256(value: Any) -> str:
    payload = value if isinstance(value, bytes) else _canonical(value)
    return hashlib.sha256(payload).hexdigest()


def _utc_now(clock: Callable[[], datetime] | None = None) -> str:
    current = (clock or (lambda: datetime.now(timezone.utc)))().astimezone(timezone.utc)
    return current.isoformat(timespec="seconds").replace("+00:00", "Z")


def _read_json(path: Path, code: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FeedbackGovernanceError(code, f"Feedback governance document is invalid: {path.name}.") from error


def load_policy(path: Path = POLICY_PATH) -> dict[str, Any]:
    policy = _read_json(path, "FEEDBACK_POLICY_INVALID")
    if not isinstance(policy, dict) or policy.get("schemaVersion") != SCHEMA_VERSION or policy.get("policyId") != POLICY_ID:
        raise FeedbackGovernanceError("FEEDBACK_POLICY_INVALID", "The feedback governance policy schema is unsupported.")
    if set(policy.get("feedbackKinds", [])) != FEEDBACK_KINDS or set(policy.get("states", [])) != FEEDBACK_STATES:
        raise FeedbackGovernanceError("FEEDBACK_POLICY_INVALID", "The feedback governance state or kind set is incomplete.")
    return policy


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or IDENTIFIER.fullmatch(value.strip()) is None:
        raise FeedbackGovernanceError("FEEDBACK_IDENTITY_INVALID", f"{field} is invalid.")
    return value.strip()


def _artifact_id(value: Any) -> str:
    if not isinstance(value, str) or ARTIFACT_IDENTIFIER.fullmatch(value.strip()) is None:
        raise FeedbackGovernanceError("FEEDBACK_ARTIFACT_INVALID", "artifactId is invalid.")
    return value.strip()


def _bounded_text(value: Any, field: str, maximum: int, *, required: bool = True) -> str:
    if not isinstance(value, str):
        if required:
            raise FeedbackGovernanceError("FEEDBACK_EVIDENCE_INVALID", f"{field} must be text.")
        return ""
    text = WHITESPACE.sub(" ", value).strip()
    if required and not text:
        raise FeedbackGovernanceError("FEEDBACK_EVIDENCE_EMPTY", f"{field} cannot be empty.")
    if len(text) > maximum:
        raise FeedbackGovernanceError("FEEDBACK_EVIDENCE_TOO_LARGE", f"{field} exceeds its bounded limit.")
    if SECRET.search(text):
        raise FeedbackGovernanceError("FEEDBACK_SECRET_REJECTED", f"{field} contains secret-like material.")
    if INSTRUCTION.search(text):
        raise FeedbackGovernanceError("FEEDBACK_INSTRUCTION_REJECTED", f"{field} contains instruction-like material.")
    if SNAPSHOT.search(text):
        raise FeedbackGovernanceError("FEEDBACK_PROJECT_CONTENT_REJECTED", f"{field} contains transcript or project-snapshot material.")
    if text[:1] in {"{", "["}:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, (dict, list)):
            raise FeedbackGovernanceError("FEEDBACK_PROJECT_CONTENT_REJECTED", f"{field} cannot be a structured project snapshot.")
    return text


def _scope_hash(kind: str, value: str) -> str:
    return _sha256(f"vfai034:{kind}:v1:{value}".encode("utf-8"))


def _atomic_write(path: Path, data: bytes) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join((_canonical(record) + b"\n") for record in records)


def _feedback_task_record(
    *,
    record_id: str,
    user_text: str,
    expected_text: str,
    revision: str,
    source_kind: str = "evaluation",
) -> dict[str, Any]:
    record = {
        "schemaVersion": 1,
        "contractVersion": "1.0.0",
        "recordId": record_id,
        "recordKind": "example",
        "task": "domain_chat",
        "input": {
            "system": {
                "type": "system",
                "text": "Answer this reviewed VoltForge regression using deterministic project authority and bounded evidence.",
                "policyVersion": "vfai034-feedback-policy-v1",
            },
            "user": {"type": "user", "text": user_text},
            "projectContext": {
                "type": "project-context",
                "projectId": "feedback-regression-suite",
                "sourceProjectRevision": revision,
                "revisionSource": "evaluation",
                "boardType": None,
                "payload": {},
            },
            "toolEvidence": [],
        },
        "output": {
            "assistantText": {
                "type": "assistant-text",
                "text": expected_text,
                "evidenceRefs": [],
            },
            "structuredActions": [],
            "citations": [],
        },
        "metadata": {"sourceKind": source_kind, "sourceIds": []},
    }
    try:
        return validate_task_record(record)
    except TaskContractError as error:
        raise FeedbackGovernanceError("FEEDBACK_TASK_RECORD_INVALID", "The reviewed example does not match the task contract.") from error


def _heldout_registry(extra_records: Sequence[Mapping[str, Any]]) -> HeldOutRegistry:
    base = get_held_out_registry()
    extra_cases = [
        {
            "id": f"vf-feedback-heldout-{_sha256(item)[:24]}",
            # Protect the authored regression prompt and expected behavior,
            # but not the shared policy text or operational metadata. This
            # prevents false collisions while preserving the actual leakage
            # boundary.
            "input": {"user": item.get("input", {}).get("user", {})},
            "assertions": [item.get("output", {}).get("assistantText", {}).get("text", "")],
        }
        for item in extra_records
    ]
    return HeldOutRegistry([*load_cases(), *extra_cases], base.semantic_threshold)


class FeedbackStore:
    """SQLite-backed review queue with content-free public summaries."""

    def __init__(
        self,
        database_path: str | Path = DEFAULT_DATABASE_PATH,
        *,
        artifact_directory: str | Path | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.path = Path(database_path).resolve()
        self.artifact_directory = Path(artifact_directory or self.path.parent / "artifacts").resolve()
        self.clock = clock
        self.policy = load_policy()
        self._lock = threading.RLock()
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA foreign_keys = ON")
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
                CREATE TABLE IF NOT EXISTS feedback_records (
                    feedback_id TEXT PRIMARY KEY,
                    owner_key TEXT NOT NULL,
                    project_key TEXT NOT NULL,
                    project_revision TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    response_record_id TEXT NOT NULL,
                    artifact_id TEXT NOT NULL,
                    registry_revision INTEGER,
                    feedback_kind TEXT NOT NULL CHECK (feedback_kind IN ('useful', 'incorrect', 'unsafe')),
                    rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
                    evidence TEXT NOT NULL,
                    expected_behavior TEXT NOT NULL,
                    evidence_sha256 TEXT NOT NULL,
                    expected_behavior_sha256 TEXT NOT NULL,
                    evidence_approved INTEGER NOT NULL CHECK (evidence_approved = 1),
                    training_consent INTEGER NOT NULL CHECK (training_consent IN (0, 1)),
                    state TEXT NOT NULL CHECK (state IN ('pending-review', 'rejected', 'held-out', 'training-approved')),
                    version INTEGER NOT NULL CHECK (version >= 1),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    reviewer_id TEXT,
                    review_note TEXT,
                    heldout_path TEXT,
                    heldout_sha256 TEXT,
                    training_record TEXT,
                    training_sha256 TEXT,
                    UNIQUE (project_key, request_id, feedback_kind)
                );
                CREATE INDEX IF NOT EXISTS idx_feedback_review ON feedback_records(state, created_at, feedback_id);
                CREATE TABLE IF NOT EXISTS retraining_runs (
                    run_id TEXT PRIMARY KEY,
                    run_path TEXT NOT NULL,
                    training_sha256 TEXT NOT NULL,
                    state TEXT NOT NULL CHECK (state IN ('scheduled', 'verified')),
                    created_at TEXT NOT NULL
                );
                """
            )
        try:
            self.path.chmod(0o600)
        except OSError:
            pass

    def _row(self, row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        for field in ("evidence_approved", "training_consent"):
            result[field] = bool(result[field])
        return result

    def _public(self, row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "feedbackId": row["feedback_id"],
            "state": row["state"],
            "feedbackKind": row["feedback_kind"],
            "rating": row["rating"],
            "requestId": row["request_id"],
            "responseRecordId": row["response_record_id"],
            "artifactId": row["artifact_id"],
            "registryRevision": row["registry_revision"],
            "trainingEligible": row["state"] == "training-approved",
            "rawContentStored": False,
        }

    def submit(
        self,
        *,
        user_id: str,
        project_id: str,
        project_revision: str,
        request_id: str,
        response_record_id: str,
        artifact_id: str,
        registry_revision: int | None,
        feedback_kind: str,
        rating: int,
        evidence: str,
        expected_behavior: str | None,
        evidence_approved: bool,
        training_consent: bool,
    ) -> dict[str, Any]:
        policy_limits = self.policy["limits"]
        user = _identifier(user_id, "authenticated user")
        project = _identifier(project_id, "projectId")
        revision = _identifier(project_revision, "projectRevision")
        request = _identifier(request_id, "requestId")
        response = _identifier(response_record_id, "responseRecordId")
        artifact = _artifact_id(artifact_id)
        if feedback_kind not in FEEDBACK_KINDS:
            raise FeedbackGovernanceError("FEEDBACK_KIND_INVALID", "Feedback kind is unsupported.")
        if isinstance(rating, bool) or not isinstance(rating, int) or not 1 <= rating <= 5:
            raise FeedbackGovernanceError("FEEDBACK_RATING_INVALID", "Feedback rating must be between 1 and 5.")
        if not evidence_approved:
            raise FeedbackGovernanceError("FEEDBACK_APPROVAL_REQUIRED", "Explicit approval is required before evidence admission.")
        safe_evidence = _bounded_text(evidence, "evidence", int(policy_limits["maximumEvidenceCharacters"]))
        safe_expected = _bounded_text(
            expected_behavior,
            "expectedBehavior",
            int(policy_limits["maximumExpectedBehaviorCharacters"]),
            required=feedback_kind in {"incorrect", "unsafe"},
        )
        if feedback_kind == "useful":
            safe_expected = ""
            training_consent = False
        owner_key = _scope_hash("owner", user)
        project_key = _scope_hash("project", f"{owner_key}:{project}")
        now = _utc_now(self.clock)
        evidence_hash = _sha256(safe_evidence)
        expected_hash = _sha256(safe_expected)
        feedback_id = f"vf-feedback-v1-{_sha256({ 'projectKey': project_key, 'requestId': request, 'kind': feedback_kind, 'evidenceSha256': evidence_hash, 'createdAt': now })[:24]}"
        try:
            with self._lock, self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO feedback_records(
                        feedback_id, owner_key, project_key, project_revision,
                        request_id, response_record_id, artifact_id, registry_revision,
                        feedback_kind, rating, evidence, expected_behavior,
                        evidence_sha256, expected_behavior_sha256, evidence_approved,
                        training_consent, state, version, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, 'pending-review', 1, ?, ?)
                    """,
                    (
                        feedback_id,
                        owner_key,
                        project_key,
                        revision,
                        request,
                        response,
                        artifact,
                        registry_revision,
                        feedback_kind,
                        rating,
                        safe_evidence,
                        safe_expected,
                        evidence_hash,
                        expected_hash,
                        int(bool(training_consent)),
                        now,
                        now,
                    ),
                )
                row = connection.execute("SELECT * FROM feedback_records WHERE feedback_id = ?", (feedback_id,)).fetchone()
        except sqlite3.IntegrityError as error:
            raise FeedbackGovernanceError("FEEDBACK_DUPLICATE", "Feedback for this request and kind was already admitted.") from error
        if row is None:
            raise FeedbackGovernanceError("FEEDBACK_STORE_FAILURE", "Feedback admission did not produce a record.")
        return self._public(self._row(row)) | {"status": "PENDING_REVIEW", "reviewRequired": True}

    def _get(self, connection: sqlite3.Connection, feedback_id: str) -> dict[str, Any]:
        row = connection.execute("SELECT * FROM feedback_records WHERE feedback_id = ?", (feedback_id,)).fetchone()
        if row is None:
            raise FeedbackGovernanceError("FEEDBACK_NOT_FOUND", "Feedback record was not found.")
        return self._row(row)

    def _heldout_document(self, row: Mapping[str, Any]) -> dict[str, Any]:
        record_id = f"vf-task-v1-{_sha256({'feedbackId': row['feedback_id'], 'stage': 'held-out'})[:24]}"
        revision = f"feedback:{row['feedback_id']}"
        record = _feedback_task_record(
            record_id=record_id,
            user_text=f"Use this reviewed VoltForge regression evidence: {row['evidence']}",
            expected_text=row["expected_behavior"],
            revision=revision,
        )
        return {
            "schemaVersion": 1,
            "documentKind": "vfai034-held-out-regression-v1",
            "feedbackId": row["feedback_id"],
            "feedbackKind": row["feedback_kind"],
            "record": record,
            "evidenceSha256": row["evidence_sha256"],
            "expectedBehaviorSha256": row["expected_behavior_sha256"],
            "createdAtUtc": _utc_now(self.clock),
        }

    def review(
        self,
        feedback_id: str,
        *,
        decision: str,
        reviewer_id: str,
        review_note: str = "",
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        feedback = _identifier(feedback_id, "feedbackId")
        reviewer = _identifier(reviewer_id, "reviewerId")
        note = _bounded_text(review_note, "reviewNote", int(self.policy["limits"]["maximumReviewNoteCharacters"]), required=False)
        if decision not in {"reject", "approve-heldout"}:
            raise FeedbackGovernanceError("FEEDBACK_REVIEW_DECISION_INVALID", "Review decision must be reject or approve-heldout.")
        with self._lock, self._connect() as connection:
            row = self._get(connection, feedback)
            if expected_version is not None and row["version"] != expected_version:
                raise FeedbackGovernanceError("FEEDBACK_VERSION_CONFLICT", "Feedback changed before review.")
            if row["state"] != "pending-review":
                raise FeedbackGovernanceError("FEEDBACK_INVALID_TRANSITION", "Only pending feedback may be reviewed.")
            if decision == "approve-heldout" and row["feedback_kind"] == "useful":
                raise FeedbackGovernanceError("FEEDBACK_NOT_A_REGRESSION", "Useful feedback is retained for product review and cannot enter the regression suite.")
            now = _utc_now(self.clock)
            if decision == "reject":
                connection.execute(
                    "UPDATE feedback_records SET state = 'rejected', version = version + 1, updated_at = ?, reviewer_id = ?, review_note = ? WHERE feedback_id = ? AND version = ?",
                    (now, reviewer, note, feedback, row["version"]),
                )
                updated = self._get(connection, feedback)
                return self._public(updated) | {"status": "REJECTED"}
            document = self._heldout_document(row)
            heldout_path = self.artifact_directory / "heldout" / f"{feedback}.json"
            heldout_bytes = _json_bytes(document)
            heldout_sha = _sha256(heldout_bytes)
            _atomic_write(heldout_path, heldout_bytes)
            connection.execute(
                "UPDATE feedback_records SET state = 'held-out', version = version + 1, updated_at = ?, reviewer_id = ?, review_note = ?, heldout_path = ?, heldout_sha256 = ? WHERE feedback_id = ? AND version = ?",
                (now, reviewer, note, str(heldout_path), heldout_sha, feedback, row["version"]),
            )
            updated = self._get(connection, feedback)
        return self._public(updated) | {"status": "HELD_OUT_ADMITTED", "heldoutSha256": heldout_sha}

    def approve_training(
        self,
        feedback_id: str,
        *,
        training_example: Mapping[str, Any],
        reviewer_id: str,
        review_note: str = "",
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        feedback = _identifier(feedback_id, "feedbackId")
        reviewer = _identifier(reviewer_id, "reviewerId")
        note = _bounded_text(review_note, "reviewNote", int(self.policy["limits"]["maximumReviewNoteCharacters"]), required=False)
        if not isinstance(training_example, Mapping):
            raise FeedbackGovernanceError("FEEDBACK_TRAINING_EXAMPLE_INVALID", "A distinct reviewed training example is required.")
        user_text = _bounded_text(training_example.get("message"), "trainingExample.message", int(self.policy["limits"]["maximumTrainingExampleCharacters"]))
        expected_text = _bounded_text(training_example.get("expectedBehavior"), "trainingExample.expectedBehavior", int(self.policy["limits"]["maximumTrainingExampleCharacters"]))
        with self._lock, self._connect() as connection:
            row = self._get(connection, feedback)
            if expected_version is not None and row["version"] != expected_version:
                raise FeedbackGovernanceError("FEEDBACK_VERSION_CONFLICT", "Feedback changed before training approval.")
            if row["state"] != "held-out":
                raise FeedbackGovernanceError("FEEDBACK_HELDOUT_REQUIRED", "A feedback regression must enter the held-out set before training approval.")
            if not row["training_consent"]:
                raise FeedbackGovernanceError("FEEDBACK_TRAINING_CONSENT_REQUIRED", "Explicit training consent is required for this feedback.")
            heldout = self._load_heldout_record(row)
            training_record = _feedback_task_record(
                record_id=f"vf-task-v1-{_sha256({'feedbackId': feedback, 'stage': 'training', 'message': user_text, 'expected': expected_text})[:24]}",
                user_text=user_text,
                expected_text=expected_text,
                revision=f"feedback-training:{feedback}",
                source_kind="evaluation",
            )
            registry = _heldout_registry([heldout["record"]])
            collisions = registry.match_record(training_record)
            if collisions is not None:
                raise FeedbackGovernanceError("FEEDBACK_TRAINING_LEAKAGE", "The proposed training example overlaps a protected held-out case.")
            training_json = _canonical(training_record)
            training_sha = _sha256(training_json)
            now = _utc_now(self.clock)
            connection.execute(
                "UPDATE feedback_records SET state = 'training-approved', version = version + 1, updated_at = ?, reviewer_id = ?, review_note = ?, training_record = ?, training_sha256 = ? WHERE feedback_id = ? AND version = ?",
                (now, reviewer, note, training_json.decode("utf-8"), training_sha, feedback, row["version"]),
            )
            updated = self._get(connection, feedback)
        return self._public(updated) | {"status": "TRAINING_APPROVED", "trainingSha256": training_sha, "heldoutSha256": row["heldout_sha256"]}

    def _load_heldout_record(self, row: Mapping[str, Any]) -> dict[str, Any]:
        path_value = row.get("heldout_path")
        declared = row.get("heldout_sha256")
        if not isinstance(path_value, str) or not isinstance(declared, str) or SHA256.fullmatch(declared) is None:
            raise FeedbackGovernanceError("FEEDBACK_HELDOUT_INVALID", "The admitted held-out record is incomplete.")
        path = Path(path_value).resolve()
        try:
            data = path.read_bytes()
        except OSError as error:
            raise FeedbackGovernanceError("FEEDBACK_HELDOUT_UNAVAILABLE", "The admitted held-out record is unavailable.") from error
        if _sha256(data) != declared:
            raise FeedbackGovernanceError("FEEDBACK_HELDOUT_CHECKSUM_MISMATCH", "The admitted held-out record changed.")
        document = _read_json(path, "FEEDBACK_HELDOUT_INVALID")
        if not isinstance(document, dict) or document.get("feedbackId") != row["feedback_id"]:
            raise FeedbackGovernanceError("FEEDBACK_HELDOUT_INVALID", "The admitted held-out identity is invalid.")
        record = document.get("record")
        if not isinstance(record, dict):
            raise FeedbackGovernanceError("FEEDBACK_HELDOUT_INVALID", "The admitted held-out task record is missing.")
        try:
            document["record"] = validate_task_record(record)
        except TaskContractError as error:
            raise FeedbackGovernanceError("FEEDBACK_HELDOUT_INVALID", "The admitted held-out task record is invalid.") from error
        return document

    def schedule_retraining(
        self,
        feedback_ids: Sequence[str],
        *,
        base_artifact_id: str,
        base_registry_revision: int,
        seed: int = 34034,
        hyperparameters: Mapping[str, Any] | None = None,
        output_directory: str | Path | None = None,
    ) -> dict[str, Any]:
        limits = self.policy["limits"]
        if not feedback_ids or len(feedback_ids) > int(limits["maximumFeedbackPerRun"]):
            raise FeedbackGovernanceError("FEEDBACK_RETRAIN_INPUT_INVALID", "A retraining run must contain a bounded non-empty feedback set.")
        artifact = _artifact_id(base_artifact_id)
        if isinstance(base_registry_revision, bool) or not isinstance(base_registry_revision, int) or base_registry_revision < 1:
            raise FeedbackGovernanceError("FEEDBACK_RETRAIN_BASE_INVALID", "baseRegistryRevision must be a positive integer.")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise FeedbackGovernanceError("FEEDBACK_RETRAIN_SEED_INVALID", "Retraining seed must be a non-negative integer.")
        selected: list[dict[str, Any]] = []
        heldout_records: list[dict[str, Any]] = []
        seen: set[str] = set()
        with self._lock, self._connect() as connection:
            for value in feedback_ids:
                feedback = _identifier(value, "feedbackId")
                if feedback in seen:
                    raise FeedbackGovernanceError("FEEDBACK_RETRAIN_INPUT_INVALID", "Feedback IDs must be unique.")
                seen.add(feedback)
                row = self._get(connection, feedback)
                if row["state"] != "training-approved" or not row["training_record"]:
                    raise FeedbackGovernanceError("FEEDBACK_RETRAIN_NOT_APPROVED", "Every retraining input must be training-approved.")
                heldout = self._load_heldout_record(row)
                try:
                    training = validate_task_record(json.loads(row["training_record"]))
                except (json.JSONDecodeError, TaskContractError) as error:
                    raise FeedbackGovernanceError("FEEDBACK_TRAINING_RECORD_INVALID", "A training-approved record is invalid.") from error
                if _sha256(_canonical(training)) != row["training_sha256"]:
                    raise FeedbackGovernanceError("FEEDBACK_TRAINING_CHECKSUM_MISMATCH", "A training-approved record changed.")
                selected.append({"feedbackId": feedback, "heldoutSha256": row["heldout_sha256"], "trainingSha256": row["training_sha256"], "record": training})
                heldout_records.append(heldout["record"])
        registry = _heldout_registry(heldout_records)
        accepted, rejected = exclude_held_out_records([item["record"] for item in selected], registry)
        if rejected or len(accepted) != len(selected):
            raise FeedbackGovernanceError("FEEDBACK_RETRAIN_LEAKAGE", "Retraining inputs overlap the frozen or admitted held-out suite.")
        training_records = [dict(item) for item in accepted]
        dependency_paths = [
            Path("feedback_governance/policy.v1.json"),
            Path("feedback_governance/service.py"),
            Path("feedback_training/policy.v1.json"),
            Path("feedback_training/executor.py"),
            Path("evaluation/leakage.py"),
            Path("gen1_training/data.py"),
            Path("gen1_training/trainer.py"),
            Path("model/tokenizers/vfdlm-byte-bpe-v1.1.0/tokenizer_manifest.json"),
            Path("task_schema/task-record.schema.json"),
            Path("synthetic_data/pipeline-lock.v1.json"),
            Path("model/release-policy.v1.json"),
        ]
        dependencies = []
        for relative in dependency_paths:
            path = AI_ROOT / relative
            if not path.is_file():
                raise FeedbackGovernanceError("FEEDBACK_RETRAIN_DEPENDENCY_MISSING", f"Retraining dependency is missing: {relative.as_posix()}.")
            dependencies.append({"path": relative.as_posix(), "sha256": _sha256(path.read_bytes())})
        compatibility = _current_compatibility()
        training_bytes = _jsonl_bytes(training_records)
        training_sha = _sha256(training_bytes)
        core = {
            "schemaVersion": 1,
            "runFormat": "vfai034-retraining-run-v1",
            "baseModel": {"artifactId": artifact, "registryRevision": base_registry_revision},
            "feedbackInputs": [{key: item[key] for key in ("feedbackId", "heldoutSha256", "trainingSha256")} for item in selected],
            "trainingRecordCount": len(training_records),
            "trainingShardSha256": training_sha,
            "compatibility": compatibility,
            "deterministic": {"seed": seed, "hyperparameters": dict(hyperparameters or {})},
            "dependencies": dependencies,
            "requiredGates": self.policy["retraining"]["requiredReleaseGates"],
            "liveChatWeightMutation": False,
        }
        run_id = f"vf-retrain-v1-{_sha256(core)[:24]}"
        run = {
            **core,
            "runId": run_id,
            "state": "scheduled",
            "createdAtUtc": _utc_now(self.clock),
        }
        run["runSha256"] = _sha256({key: value for key, value in run.items() if key != "runSha256"})
        destination = Path(output_directory or self.artifact_directory / "runs").resolve()
        run_path = destination / f"{run_id}.json"
        training_path = destination / f"{run_id}.training.jsonl"
        _atomic_write(training_path, training_bytes)
        _atomic_write(run_path, _json_bytes({**run, "trainingPath": str(training_path), "trainingSha256": training_sha}))
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO retraining_runs(run_id, run_path, training_sha256, state, created_at) VALUES (?, ?, ?, 'scheduled', ?)",
                (run_id, str(run_path), training_sha, run["createdAtUtc"]),
            )
        return {"runId": run_id, "state": "SCHEDULED", "trainingRecordCount": len(training_records), "trainingSha256": training_sha, "runSha256": run["runSha256"], "requiredGates": run["requiredGates"], "liveChatWeightMutation": False}

    def verify_execution_admission(
        self,
        run: Mapping[str, Any],
        *,
        run_path: str | Path,
    ) -> dict[str, Any]:
        """Revalidate consent and leakage immediately before model allocation.

        This method is deliberately read-only. A failed or successful admission
        check does not change feedback records, the scheduled-run row, memory,
        artifacts, or the model registry.
        """

        run_id = _identifier(run.get("runId"), "runId")
        base = run.get("baseModel")
        inputs = run.get("feedbackInputs")
        if not isinstance(base, Mapping) or not isinstance(inputs, list) or not inputs:
            raise FeedbackGovernanceError(
                "FEEDBACK_EXECUTION_INPUT_INVALID",
                "The scheduled execution inputs are incomplete.",
            )
        base_artifact = _artifact_id(base.get("artifactId"))
        base_revision = base.get("registryRevision")
        if (
            isinstance(base_revision, bool)
            or not isinstance(base_revision, int)
            or base_revision < 1
        ):
            raise FeedbackGovernanceError(
                "FEEDBACK_EXECUTION_BASE_INVALID",
                "The scheduled base registry revision is invalid.",
            )
        manifest_path = Path(run_path).resolve()
        selected_records: list[dict[str, Any]] = []
        selected_hashes: list[dict[str, str]] = []
        with self._lock, self._connect() as connection:
            scheduled = connection.execute(
                "SELECT * FROM retraining_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if scheduled is None:
                raise FeedbackGovernanceError(
                    "FEEDBACK_EXECUTION_RUN_NOT_REGISTERED",
                    "The retraining run is not registered in the feedback store.",
                )
            if (
                Path(str(scheduled["run_path"])).resolve() != manifest_path
                or scheduled["training_sha256"] != run.get("trainingSha256")
                or scheduled["state"] != "scheduled"
            ):
                raise FeedbackGovernanceError(
                    "FEEDBACK_EXECUTION_RUN_MISMATCH",
                    "The registered retraining run does not match the supplied manifest.",
                )
            seen: set[str] = set()
            for item in inputs:
                if not isinstance(item, Mapping) or set(item) != {
                    "feedbackId",
                    "heldoutSha256",
                    "trainingSha256",
                }:
                    raise FeedbackGovernanceError(
                        "FEEDBACK_EXECUTION_INPUT_INVALID",
                        "A feedback execution input is malformed.",
                    )
                feedback_id = _identifier(item.get("feedbackId"), "feedbackId")
                if feedback_id in seen:
                    raise FeedbackGovernanceError(
                        "FEEDBACK_EXECUTION_INPUT_INVALID",
                        "Feedback execution inputs must be unique.",
                    )
                seen.add(feedback_id)
                row = self._get(connection, feedback_id)
                if (
                    row["state"] != "training-approved"
                    or row["evidence_approved"] is not True
                    or row["training_consent"] is not True
                    or row["artifact_id"] != base_artifact
                    or row["registry_revision"] != base_revision
                    or row["heldout_sha256"] != item.get("heldoutSha256")
                    or row["training_sha256"] != item.get("trainingSha256")
                ):
                    raise FeedbackGovernanceError(
                        "FEEDBACK_EXECUTION_CONSENT_OR_LINEAGE_INVALID",
                        "Current feedback consent, review state, or base lineage does not match the run.",
                    )
                heldout = self._load_heldout_record(row)
                try:
                    training = validate_task_record(json.loads(row["training_record"]))
                except (TypeError, json.JSONDecodeError, TaskContractError) as error:
                    raise FeedbackGovernanceError(
                        "FEEDBACK_EXECUTION_TRAINING_RECORD_INVALID",
                        "An execution training record is invalid.",
                    ) from error
                if _sha256(_canonical(training)) != row["training_sha256"]:
                    raise FeedbackGovernanceError(
                        "FEEDBACK_EXECUTION_TRAINING_CHECKSUM_MISMATCH",
                        "An execution training record changed after approval.",
                    )
                selected_records.append(training)
                selected_hashes.append(
                    {
                        "feedbackId": feedback_id,
                        "heldoutSha256": str(row["heldout_sha256"]),
                        "trainingSha256": str(row["training_sha256"]),
                    }
                )
            heldout_rows = connection.execute(
                "SELECT * FROM feedback_records WHERE state IN ('held-out', 'training-approved') ORDER BY feedback_id"
            ).fetchall()
            all_heldout_records = [
                self._load_heldout_record(self._row(row))["record"] for row in heldout_rows
            ]
        registry = _heldout_registry(all_heldout_records)
        accepted, rejected = exclude_held_out_records(selected_records, registry)
        if rejected or len(accepted) != len(selected_records):
            raise FeedbackGovernanceError(
                "FEEDBACK_EXECUTION_HELDOUT_LEAKAGE",
                "Execution inputs overlap the current frozen or admitted held-out suite.",
            )
        return {
            "runId": run_id,
            "baseArtifactId": base_artifact,
            "baseRegistryRevision": base_revision,
            "feedbackInputCount": len(selected_records),
            "currentHeldoutRecordCount": len(all_heldout_records),
            "inputSetSha256": _sha256(selected_hashes),
            "consentAndReviewReverified": True,
            "currentHeldoutLeakageRejected": True,
            "stateMutated": False,
        }

    def _counts(self) -> dict[str, int]:
        with self._lock, self._connect() as connection:
            rows = connection.execute("SELECT state, COUNT(*) AS count FROM feedback_records GROUP BY state").fetchall()
            runs = connection.execute("SELECT COUNT(*) AS count FROM retraining_runs").fetchone()["count"]
        counts = {state: 0 for state in FEEDBACK_STATES}
        counts.update({str(row["state"]): int(row["count"]) for row in rows})
        counts["retrainingRuns"] = int(runs)
        return counts

    def health(self) -> dict[str, Any]:
        try:
            counts = self._counts()
            return {
                "policyId": POLICY_ID,
                "state": "ready",
                "databaseAvailable": True,
                "counts": counts,
                "rawPromptStored": False,
                "rawModelResponseStored": False,
                "rawProjectSnapshotStored": False,
                "liveChatWeightMutation": False,
            }
        except (OSError, sqlite3.Error) as error:
            return {"policyId": POLICY_ID, "state": "unavailable", "databaseAvailable": False, "errorType": type(error).__name__, "rawPromptStored": False, "rawModelResponseStored": False, "rawProjectSnapshotStored": False, "liveChatWeightMutation": False}


def _current_compatibility() -> dict[str, Any]:
    try:
        from model.release_lifecycle import current_release_compatibility

        return current_release_compatibility()
    except Exception as error:
        raise FeedbackGovernanceError("FEEDBACK_RETRAIN_COMPATIBILITY_UNAVAILABLE", "The serving compatibility contract could not be captured.") from error


def get_feedback_store(
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    *,
    artifact_directory: str | Path | None = None,
) -> FeedbackStore:
    return FeedbackStore(database_path, artifact_directory=artifact_directory)


def feedback_health(database_path: str | Path = DEFAULT_DATABASE_PATH) -> dict[str, Any]:
    try:
        return FeedbackStore(database_path).health()
    except (FeedbackGovernanceError, OSError, sqlite3.Error) as error:
        return {"policyId": POLICY_ID, "state": "unavailable", "databaseAvailable": False, "errorType": type(error).__name__, "rawPromptStored": False, "rawModelResponseStored": False, "rawProjectSnapshotStored": False, "liveChatWeightMutation": False}


def verify_retraining_run(path: str | Path) -> dict[str, Any]:
    run_path = Path(path).resolve()
    run = _read_json(run_path, "FEEDBACK_RETRAIN_RUN_INVALID")
    if not isinstance(run, dict) or run.get("schemaVersion") != 1 or run.get("runFormat") != "vfai034-retraining-run-v1":
        raise FeedbackGovernanceError("FEEDBACK_RETRAIN_RUN_INVALID", "Retraining run schema is unsupported.")
    declared = run.get("runSha256")
    expected = _sha256({key: value for key, value in run.items() if key != "runSha256" and key not in {"trainingPath", "trainingSha256"}})
    if declared != expected:
        raise FeedbackGovernanceError("FEEDBACK_RETRAIN_RUN_CHECKSUM_MISMATCH", "Retraining run provenance checksum is invalid.")
    training_path = Path(str(run.get("trainingPath") or "")).resolve()
    training_bytes = training_path.read_bytes()
    if _sha256(training_bytes) != run.get("trainingSha256") or _sha256(training_bytes) != run.get("trainingShardSha256"):
        raise FeedbackGovernanceError("FEEDBACK_RETRAIN_TRAINING_CHECKSUM_MISMATCH", "Retraining candidate input changed.")
    records = []
    for line in training_bytes.splitlines():
        if line.strip():
            try:
                records.append(validate_task_record(json.loads(line)))
            except (json.JSONDecodeError, TaskContractError) as error:
                raise FeedbackGovernanceError("FEEDBACK_RETRAIN_TRAINING_RECORD_INVALID", "Retraining candidate input is invalid.") from error
    if len(records) != run.get("trainingRecordCount"):
        raise FeedbackGovernanceError("FEEDBACK_RETRAIN_COUNT_MISMATCH", "Retraining candidate input count changed.")
    if run.get("liveChatWeightMutation") is not False:
        raise FeedbackGovernanceError("FEEDBACK_RETRAIN_MUTATION_POLICY_INVALID", "Retraining runs must prohibit live weight mutation.")
    return {"runId": run.get("runId"), "state": run.get("state"), "trainingRecordCount": len(records), "trainingSha256": run.get("trainingSha256"), "runSha256": declared, "verified": True}
