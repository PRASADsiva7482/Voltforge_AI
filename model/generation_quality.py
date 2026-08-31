"""Fail-closed neural generation quality gates and privacy-safe counters.

The gate accepts only a strict JSON envelope.  It renders visible text and
constructs typed task output after all evidence, revision, safety, repetition,
truncation, and confidence checks pass.  Raw model text is never interpreted as
a circuit or firmware action.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import re
import threading
from typing import Any, Callable, Mapping, Sequence

from grounding import NeuralGroundingError, validate_neural_claim
from task_schema.schema import TaskContractError, validate_task_record


QUALITY_GATE_POLICY_ID = "vfai019-generation-quality-policy-v1"
QUALITY_GATE_SCHEMA_VERSION = 1
MAX_CANDIDATE_BYTES = 64_000
MAX_VISIBLE_CHARACTERS = 8_000
MAX_SEGMENTS = 16
MAX_ACTIONS = 8

_SEGMENT_TYPES = {"grounded-claim", "safety-warning", "uncertainty"}
_FINISH_REASONS = {"stop", "eos", "length"}
_AUTHORITATIVE_ENGINEERING_POLICY_ID = "vfai021-authoritative-engineering-tools-v1"
_CONTRADICTORY_ASSURANCE = re.compile(
    r"\b(no (?:critical|blocking|electrical|firmware|safety) (?:issue|issues|finding|findings)|"
    r"safe to (?:connect|power|upload|use)|ignore (?:the )?(?:tool|finding|warning)|"
    r"override (?:the )?(?:tool|finding|warning)|false positive)\b",
    re.IGNORECASE,
)
_UNCERTAINTY_MARKERS = (
    "uncertain",
    "unknown",
    "cannot verify",
    "can't verify",
    "insufficient evidence",
    "not available",
    "not provided",
)
_OUT_OF_DOMAIN = re.compile(
    r"\b(?:medical diagnosis|legal advice|financial advice|credential theft|password cracking|weapon construction)\b",
    re.IGNORECASE,
)
_PROMPT_INJECTION = re.compile(
    r"\b(?:ignore|override|reveal)\s+(?:all\s+|any\s+|the\s+)?(?:previous|system|hidden)\s+instructions?\b",
    re.IGNORECASE,
)
_DANGEROUS_OPERATION = re.compile(
    r"\b(?:bypass|disable|remove|short|bridge)\s+(?:the\s+)?(?:fuse|protection|interlock|isolation|current[- ]limiting resistor|supply rails?)\b|"
    r"\bconnect\s+(?:directly\s+)?(?:to\s+)?(?:mains|line voltage)\b",
    re.IGNORECASE,
)
_DOMAIN_LANGUAGE = re.compile(
    r"\b(?:adc|arduino|board|circuit|component|current|datasheet|electronic|firmware|gpio|ground|i2c|"
    r"led|microcontroller|netlist|pin|power|pwm|resistor|sensor|signal|spi|uart|voltage|wire)\b",
    re.IGNORECASE,
)
_SUPPORT_STOP_WORDS = {
    "about",
    "after",
    "because",
    "before",
    "could",
    "from",
    "have",
    "into",
    "must",
    "requires",
    "should",
    "that",
    "their",
    "there",
    "these",
    "this",
    "through",
    "with",
    "would",
}
_HIDDEN_EXECUTION_KEYS = {
    "autoapply",
    "applyautomatically",
    "executeimmediately",
    "skipconfirmation",
    "bypassconfirmation",
}


class GenerationQualityError(ValueError):
    """Stable rejection code without embedding model output in the message."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class QualityGateDecision:
    accepted: bool
    code: str
    message: str
    response_text: str | None = None
    confidence: float = 0.0
    response_record: dict[str, Any] | None = None
    evidence_refs: tuple[str, ...] = ()
    action_count: int = 0

    def public_metadata(self) -> dict[str, Any]:
        return {
            "policyId": QUALITY_GATE_POLICY_ID,
            "status": "accepted" if self.accepted else "rejected",
            "code": self.code,
            "confidence": self.confidence if self.accepted else None,
            "evidenceCount": len(self.evidence_refs),
            "actionCount": self.action_count if self.accepted else 0,
            "rawOutputStored": False,
        }


@dataclass(frozen=True, slots=True)
class GenerationResolution:
    """One honest generation outcome: accepted neural output or deterministic fallback."""

    metadata: dict[str, Any]
    response_text: str | None = None
    response_record: dict[str, Any] | None = None
    deterministic_value: Any = None


class GenerationQualityMetrics:
    """Process-local counters that never retain prompt or output content."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._evaluated = 0
        self._accepted = 0
        self._rejected = 0
        self._fallbacks = 0
        self._neural_attempts = 0
        self._rejection_codes: Counter[str] = Counter()
        self._fallback_reasons: Counter[str] = Counter()

    def record_gate(self, decision: QualityGateDecision) -> None:
        with self._lock:
            self._evaluated += 1
            self._neural_attempts += 1
            if decision.accepted:
                self._accepted += 1
            else:
                self._rejected += 1
                self._rejection_codes[decision.code] += 1

    def record_fallback(self, reason_code: str, *, neural_attempted: bool) -> None:
        with self._lock:
            self._fallbacks += 1
            if neural_attempted:
                self._neural_attempts += 1
            self._fallback_reasons[reason_code] += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "policyId": QUALITY_GATE_POLICY_ID,
                "evaluatedCandidates": self._evaluated,
                "acceptedCandidates": self._accepted,
                "rejectedCandidates": self._rejected,
                "deterministicFallbacks": self._fallbacks,
                "neuralAttempts": self._neural_attempts,
                "rejectionCodes": dict(sorted(self._rejection_codes.items())),
                "fallbackReasons": dict(sorted(self._fallback_reasons.items())),
                "rawPromptStored": False,
                "rawOutputStored": False,
            }

    def reset_for_tests(self) -> None:
        with self._lock:
            self._evaluated = 0
            self._accepted = 0
            self._rejected = 0
            self._fallbacks = 0
            self._neural_attempts = 0
            self._rejection_codes.clear()
            self._fallback_reasons.clear()


generation_quality_metrics = GenerationQualityMetrics()


class GenerationQualityGate:
    """Validate one bounded model envelope against one typed request record."""

    def __init__(self, metrics: GenerationQualityMetrics | None = None) -> None:
        self.metrics = metrics

    def evaluate(
        self,
        candidate_text: str,
        request_record: Mapping[str, Any],
    ) -> QualityGateDecision:
        try:
            decision = self._evaluate(candidate_text, request_record)
        except GenerationQualityError as error:
            decision = QualityGateDecision(False, error.code, error.message)
        if self.metrics is not None:
            self.metrics.record_gate(decision)
        return decision

    def _evaluate(
        self,
        candidate_text: str,
        request_record: Mapping[str, Any],
    ) -> QualityGateDecision:
        if not isinstance(candidate_text, str) or not candidate_text.strip():
            self._reject("QG_OUTPUT_MISSING", "The neural candidate is empty.")
        try:
            encoded = candidate_text.encode("utf-8", errors="strict")
        except UnicodeError:
            self._reject("QG_SCHEMA_INVALID", "The neural candidate is not valid UTF-8 text.")
        if len(encoded) > MAX_CANDIDATE_BYTES or "\x00" in candidate_text:
            self._reject("QG_OUTPUT_OVERSIZED", "The neural candidate exceeds its bound.")
        try:
            envelope = json.loads(candidate_text)
        except (UnicodeError, json.JSONDecodeError, RecursionError):
            self._reject("QG_SCHEMA_INVALID", "The neural candidate is not a valid JSON envelope.")
        if not isinstance(envelope, dict):
            self._reject("QG_SCHEMA_INVALID", "The neural candidate envelope must be an object.")
        expected_top = {
            "schemaVersion",
            "domain",
            "finishReason",
            "confidence",
            "segments",
            "structuredActions",
            "citationEvidenceIds",
        }
        if set(envelope) != expected_top or envelope.get("schemaVersion") != 1:
            self._reject("QG_SCHEMA_INVALID", "The neural candidate envelope is incompatible.")
        if envelope.get("domain") != "voltforge-electronics":
            self._reject("QG_OUT_OF_DOMAIN", "The neural candidate is outside VoltForge scope.")
        finish_reason = envelope.get("finishReason")
        if finish_reason not in _FINISH_REASONS:
            self._reject("QG_SCHEMA_INVALID", "The neural finish reason is invalid.")
        if finish_reason == "length":
            self._reject("QG_TRUNCATED", "The neural candidate ended at its token limit.")

        try:
            request = validate_task_record(request_record)
        except TaskContractError as error:
            raise GenerationQualityError(
                "QG_CONTEXT_INVALID",
                "Quality gates require a valid typed inference request.",
            ) from error
        if request["recordKind"] != "inference-request":
            self._reject("QG_CONTEXT_INVALID", "Quality gates require an inference request.")
        evidence = {
            item["evidenceId"]: item for item in request["input"]["toolEvidence"]
        }
        segments = envelope.get("segments")
        if not isinstance(segments, list) or not 1 <= len(segments) <= MAX_SEGMENTS:
            self._reject("QG_SCHEMA_INVALID", "Neural segments are missing or out of bounds.")

        rendered: list[str] = []
        referenced: set[str] = set()
        uncertainty_present = False
        seen_segments: set[str] = set()
        for segment in segments:
            if not isinstance(segment, dict) or set(segment) != {
                "type",
                "text",
                "evidenceRefs",
            }:
                self._reject("QG_SCHEMA_INVALID", "A neural segment is malformed.")
            kind = segment.get("type")
            text = segment.get("text")
            refs = segment.get("evidenceRefs")
            if kind not in _SEGMENT_TYPES or not isinstance(text, str):
                self._reject("QG_SCHEMA_INVALID", "A neural segment type or text is invalid.")
            text = text.strip()
            if not text or len(text) > 2_000 or not isinstance(refs, list):
                self._reject("QG_SCHEMA_INVALID", "A neural segment exceeds its contract.")
            if any(not isinstance(item, str) for item in refs) or len(refs) != len(set(refs)):
                self._reject("QG_CITATION_INVALID", "Neural evidence references are invalid.")
            normalized = self._normalize(text)
            if normalized in seen_segments:
                self._reject("QG_REPETITION", "The neural candidate repeats a segment.")
            seen_segments.add(normalized)
            if _OUT_OF_DOMAIN.search(text):
                self._reject("QG_OUT_OF_DOMAIN", "The neural candidate is outside VoltForge scope.")
            if _PROMPT_INJECTION.search(text):
                self._reject("QG_DANGEROUS_INSTRUCTION", "The neural candidate contains instruction override text.")
            if kind == "uncertainty":
                uncertainty_present = True
                if refs or not any(marker in text.casefold() for marker in _UNCERTAINTY_MARKERS):
                    self._reject("QG_UNCERTAINTY_INVALID", "Uncertainty is not represented honestly.")
            else:
                if _DANGEROUS_OPERATION.search(text):
                    if kind != "safety-warning" or not self._is_warning(text):
                        self._reject(
                            "QG_DANGEROUS_INSTRUCTION",
                            "The neural candidate contains a dangerous instruction.",
                        )
                if not _DOMAIN_LANGUAGE.search(text):
                    self._reject("QG_OUT_OF_DOMAIN", "A neural claim lacks electronics scope.")
                if not refs:
                    self._reject("QG_UNSUPPORTED_CLAIM", "A neural claim has no evidence.")
                self._validate_evidence_refs(refs, evidence)
                self._validate_claim_support(text, refs, evidence)
                try:
                    validate_neural_claim(text, refs, evidence)
                except NeuralGroundingError as error:
                    self._reject(error.code, error.message)
                referenced.update(refs)
            rendered.append(text)

        self._validate_authoritative_engineering_findings(segments, evidence)

        response_text = "\n".join(rendered)
        if len(response_text) > MAX_VISIBLE_CHARACTERS:
            self._reject("QG_OUTPUT_OVERSIZED", "Visible neural output exceeds its bound.")
        self._validate_repetition(response_text)
        self._validate_completion(response_text)

        citation_ids = envelope.get("citationEvidenceIds")
        if (
            not isinstance(citation_ids, list)
            or any(not isinstance(item, str) for item in citation_ids)
            or len(citation_ids) != len(set(citation_ids))
            or set(citation_ids) != referenced
        ):
            self._reject("QG_CITATION_INVALID", "Neural citations do not exactly cover evidence use.")

        confidence = self._validate_confidence(
            envelope.get("confidence"),
            referenced,
            evidence,
            uncertainty_present,
        )
        actions = envelope.get("structuredActions")
        if not isinstance(actions, list) or len(actions) > MAX_ACTIONS:
            self._reject("QG_SCHEMA_INVALID", "Neural structured actions exceed their bound.")
        if actions and (uncertainty_present or confidence < 0.7):
            self._reject("QG_ACTION_UNSAFE", "Uncertain neural output cannot create actions.")
        action_refs = self._validate_actions(
            actions,
            evidence,
            request["input"]["projectContext"]["sourceProjectRevision"],
        )
        if not action_refs.issubset(referenced):
            self._reject("QG_CITATION_INVALID", "Neural action evidence is not cited in visible output.")

        response_record = self._build_response_record(
            request,
            response_text,
            confidence,
            actions,
            sorted(referenced),
            evidence,
        )
        return QualityGateDecision(
            True,
            "QG_ACCEPTED",
            "The neural candidate passed all generation quality gates.",
            response_text=response_text,
            confidence=confidence,
            response_record=response_record,
            evidence_refs=tuple(sorted(referenced)),
            action_count=len(actions),
        )

    def _validate_authoritative_engineering_findings(
        self,
        segments: Sequence[Mapping[str, Any]],
        evidence: Mapping[str, Mapping[str, Any]],
    ) -> None:
        authoritative_indexes = {
            evidence_id: item
            for evidence_id, item in evidence.items()
            if item.get("toolName") == "tool:engineering-authority-index"
            and item.get("payload", {}).get("policyId")
            == _AUTHORITATIVE_ENGINEERING_POLICY_ID
            and item.get("payload", {}).get("blockingFindingIds")
        }
        if not authoritative_indexes:
            return
        index_ids = set(authoritative_indexes)
        warnings = [
            segment
            for segment in segments
            if segment.get("type") == "safety-warning"
            and index_ids.intersection(segment.get("evidenceRefs") or [])
        ]
        warned_ids = {
            reference
            for segment in warnings
            for reference in segment.get("evidenceRefs") or []
        }
        if not warnings or not index_ids.issubset(warned_ids):
            self._reject(
                "QG_AUTHORITATIVE_FINDING_OMITTED",
                "Neural output omitted an authoritative blocking engineering finding.",
            )
        for segment in segments:
            refs = set(segment.get("evidenceRefs") or [])
            text = str(segment.get("text") or "")
            if refs.intersection(index_ids) and segment.get("type") != "safety-warning":
                self._reject(
                    "QG_AUTHORITATIVE_OVERRIDE",
                    "Blocking engineering evidence may only be rendered as a safety warning.",
                )
            if (
                segment.get("type") != "safety-warning"
                and _CONTRADICTORY_ASSURANCE.search(text)
            ):
                self._reject(
                    "QG_AUTHORITATIVE_OVERRIDE",
                    "Neural output contradicts authoritative blocking engineering findings.",
                )

    @staticmethod
    def _validate_evidence_refs(
        refs: Sequence[str], evidence: Mapping[str, Mapping[str, Any]]
    ) -> None:
        for reference in refs:
            item = evidence.get(reference)
            if item is None:
                raise GenerationQualityError(
                    "QG_CITATION_INVALID", "The neural candidate cites unknown evidence."
                )
            if item.get("status") not in {"complete", "reported"}:
                raise GenerationQualityError(
                    "QG_UNSUPPORTED_CLAIM", "The neural candidate relies on failed evidence."
                )

    @classmethod
    def _validate_claim_support(
        cls,
        text: str,
        refs: Sequence[str],
        evidence: Mapping[str, Mapping[str, Any]],
    ) -> None:
        claim_tokens = cls._support_tokens(text)
        evidence_tokens: set[str] = set()
        for reference in refs:
            item = evidence[reference]
            evidence_tokens.update(cls._support_tokens(str(item.get("summary", ""))))
            evidence_tokens.update(
                cls._support_tokens(
                    json.dumps(item.get("payload", {}), sort_keys=True, default=str)
                )
            )
        if not claim_tokens.intersection(evidence_tokens):
            raise GenerationQualityError(
                "QG_UNSUPPORTED_CLAIM",
                "The neural claim has no lexical support in its cited evidence.",
            )

    def _validate_confidence(
        self,
        value: Any,
        referenced: set[str],
        evidence: Mapping[str, Mapping[str, Any]],
        uncertainty_present: bool,
    ) -> float:
        if not isinstance(value, dict) or set(value) != {"score", "basis"}:
            self._reject("QG_CONFIDENCE_INVALID", "Neural confidence metadata is malformed.")
        score = value.get("score")
        basis = value.get("basis")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
            or basis not in {"evidence-aligned", "uncertain"}
        ):
            self._reject("QG_CONFIDENCE_INVALID", "Neural confidence metadata is invalid.")
        if uncertainty_present and basis != "uncertain":
            self._reject("QG_CONFIDENCE_UNSUPPORTED", "Uncertainty and confidence basis conflict.")
        if not uncertainty_present and basis != "evidence-aligned":
            self._reject("QG_CONFIDENCE_UNSUPPORTED", "Grounded output has an unsupported confidence basis.")
        maximum = 0.5 if not referenced else 0.95
        authorities = {evidence[item].get("authority") for item in referenced}
        if "client-reported" in authorities:
            maximum = min(maximum, 0.65)
        if "retrieved" in authorities:
            maximum = min(maximum, 0.75)
        if uncertainty_present:
            maximum = min(maximum, 0.6)
        if float(score) > maximum + 1e-12:
            self._reject("QG_CONFIDENCE_UNSUPPORTED", "Neural confidence exceeds its evidence.")
        return float(score)

    def _validate_actions(
        self,
        actions: Sequence[Any],
        evidence: Mapping[str, Mapping[str, Any]],
        revision: str,
    ) -> set[str]:
        references: set[str] = set()
        action_ids: set[str] = set()
        for action in actions:
            if not isinstance(action, dict):
                self._reject("QG_ACTION_SCHEMA_INVALID", "A neural action is malformed.")
            expected = {
                "type",
                "actionId",
                "actionKind",
                "sourceProjectRevision",
                "applicationMode",
                "requiresUserConfirmation",
                "evidenceRefs",
                "payload",
            }
            if set(action) != expected:
                self._reject("QG_ACTION_SCHEMA_INVALID", "A neural action has unexpected fields.")
            if (
                action.get("type") != "structured-action"
                or action.get("sourceProjectRevision") != revision
                or action.get("applicationMode") != "proposal-only"
                or action.get("requiresUserConfirmation") is not True
            ):
                self._reject("QG_ACTION_UNSAFE", "A neural action bypasses review or revision binding.")
            action_id = action.get("actionId")
            refs = action.get("evidenceRefs")
            if not isinstance(action_id, str) or action_id in action_ids or not isinstance(refs, list):
                self._reject("QG_ACTION_SCHEMA_INVALID", "Neural action identity is invalid.")
            if any(not isinstance(item, str) for item in refs) or len(refs) != len(set(refs)):
                self._reject("QG_ACTION_SCHEMA_INVALID", "Neural action evidence is invalid.")
            action_ids.add(action_id)
            self._validate_evidence_refs(refs, evidence)
            if not refs or any(evidence[item].get("authority") != "deterministic" for item in refs):
                self._reject("QG_ACTION_UNSAFE", "Neural actions require deterministic evidence.")
            if self._contains_hidden_execution_control(action.get("payload")):
                self._reject(
                    "QG_ACTION_SCHEMA_INVALID",
                    "A neural action contains a forbidden execution control.",
                )
            payload_text = json.dumps(action.get("payload"), sort_keys=True, default=str)
            if _PROMPT_INJECTION.search(payload_text) or _DANGEROUS_OPERATION.search(payload_text):
                self._reject("QG_DANGEROUS_INSTRUCTION", "A neural action contains unsafe instructions.")
            if action.get("actionKind") == "code-fix" and not any(
                evidence[item].get("payload", {}).get("compiled") is True for item in refs
            ):
                self._reject("QG_ACTION_UNVERIFIED_CODE", "Neural code actions require compiled evidence.")
            if not self._is_deterministically_approved(action, refs, evidence):
                self._reject(
                    "QG_ACTION_UNSUPPORTED",
                    "A neural action does not exactly match a deterministic tool proposal.",
                )
            references.update(refs)
        return references

    @staticmethod
    def _is_deterministically_approved(
        action: Mapping[str, Any],
        refs: Sequence[str],
        evidence: Mapping[str, Mapping[str, Any]],
    ) -> bool:
        expected = {
            "actionKind": action.get("actionKind"),
            "payload": action.get("payload"),
        }
        for reference in refs:
            approved = evidence[reference].get("payload", {}).get("approvedActions", [])
            if isinstance(approved, list) and expected in approved:
                return True
        return False

    @classmethod
    def _contains_hidden_execution_control(cls, value: Any) -> bool:
        stack = [value]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                for key, child in item.items():
                    normalized = re.sub(r"[^a-z0-9]", "", str(key).casefold())
                    if normalized in _HIDDEN_EXECUTION_KEYS:
                        return True
                    stack.append(child)
            elif isinstance(item, list):
                stack.extend(item)
        return False

    @staticmethod
    def _build_response_record(
        request: Mapping[str, Any],
        response_text: str,
        confidence: float,
        actions: Sequence[Mapping[str, Any]],
        evidence_refs: list[str],
        evidence: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        citations = []
        for evidence_id in evidence_refs:
            item = evidence[evidence_id]
            payload_bytes = json.dumps(
                item.get("payload", {}), sort_keys=True, separators=(",", ":"), default=str
            ).encode("utf-8")
            citations.append(
                {
                    "type": "citation",
                    "citationId": "citation:qg:" + hashlib.sha256(evidence_id.encode()).hexdigest()[:16],
                    "sourceId": item["toolName"],
                    "sourceRevision": item["toolVersion"],
                    "title": item["summary"][:500],
                    "contentSha256": hashlib.sha256(payload_bytes).hexdigest(),
                    "evidenceRefs": [evidence_id],
                }
            )
        candidate = deepcopy(dict(request))
        digest_input = json.dumps(
            {
                "requestRecordId": request["recordId"],
                "text": response_text,
                "actions": actions,
                "confidence": confidence,
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        candidate.update(
            {
                "recordId": "vf-task-v1-" + hashlib.sha256(digest_input).hexdigest()[:24],
                "recordKind": "inference-response",
                "output": {
                    "assistantText": {
                        "type": "assistant-text",
                        "text": response_text,
                        "evidenceRefs": evidence_refs,
                    },
                    "structuredActions": list(actions),
                    "citations": citations,
                },
            }
        )
        try:
            return validate_task_record(candidate)
        except TaskContractError as error:
            raise GenerationQualityError(
                "QG_ACTION_SCHEMA_INVALID",
                "The gated response failed the typed task contract.",
            ) from error

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.casefold())).strip()

    @classmethod
    def _support_tokens(cls, text: str) -> set[str]:
        return {
            token
            for token in cls._normalize(text).split()
            if len(token) >= 4 and token not in _SUPPORT_STOP_WORDS
        }

    def _validate_repetition(self, text: str) -> None:
        tokens = self._normalize(text).split()
        if len(tokens) < 12:
            return
        grams = Counter(tuple(tokens[index : index + 4]) for index in range(len(tokens) - 3))
        if grams and max(grams.values()) > 2:
            self._reject("QG_REPETITION", "The neural candidate repeats a phrase excessively.")

    def _validate_completion(self, text: str) -> None:
        if text.count("```") % 2 or text[-1] not in ".!?;:)}]`":
            self._reject("QG_TRUNCATED", "The neural candidate appears incomplete.")

    @staticmethod
    def _is_warning(text: str) -> bool:
        lowered = text.casefold()
        return any(marker in lowered for marker in ("do not", "never", "unsafe", "danger", "warning"))

    @staticmethod
    def _reject(code: str, message: str) -> None:
        raise GenerationQualityError(code, message)


def fallback_metadata(
    model_health: Mapping[str, Any], *, context_compiler_ready: bool = False
) -> dict[str, Any]:
    """Describe the honest deterministic path without claiming a neural attempt."""

    if model_health.get("ready") is True:
        if context_compiler_ready:
            reason = "NEURAL_GENERATION_NOT_ENABLED"
            gate_code = "QG_NOT_RUN_GENERATION_DISABLED"
        else:
            reason = "NEURAL_CONTEXT_COMPILER_NOT_READY"
            gate_code = "QG_NOT_RUN_CONTEXT_UNAVAILABLE"
    else:
        reason = str(model_health.get("code") or "NEURAL_MODEL_UNAVAILABLE")
        gate_code = "QG_NOT_RUN_MODEL_UNAVAILABLE"
    return {
        "mode": "deterministic-fallback",
        "fallbackUsed": True,
        "fallbackReasonCode": reason,
        "fallbackSource": "voltforge-deterministic-tools",
        "neuralAttempted": False,
        "neuralArtifactId": model_health.get("artifactId"),
        "qualityGate": {
            "policyId": QUALITY_GATE_POLICY_ID,
            "status": "not-run",
            "code": gate_code,
            "rawOutputStored": False,
        },
    }


def resolve_generation(
    *,
    candidate_text: str | None,
    request_record: Mapping[str, Any],
    model_health: Mapping[str, Any],
    deterministic_factory: Callable[[], Any],
    gate: GenerationQualityGate | None = None,
    metrics: GenerationQualityMetrics | None = None,
    context_compiler_ready: bool = False,
) -> GenerationResolution:
    """Resolve a candidate without ever exposing unvalidated neural text.

    A missing/unavailable candidate and every rejected candidate invoke the
    deterministic factory.  Only an accepted decision can carry neural text or
    a neural-built typed response record across this boundary.
    """

    observed_metrics = metrics or generation_quality_metrics
    if model_health.get("ready") is not True or candidate_text is None:
        metadata = fallback_metadata(
            model_health, context_compiler_ready=context_compiler_ready
        )
        observed_metrics.record_fallback(
            str(metadata["fallbackReasonCode"]), neural_attempted=False
        )
        return GenerationResolution(
            metadata=metadata,
            deterministic_value=deterministic_factory(),
        )

    quality_gate = gate or GenerationQualityGate(metrics=observed_metrics)
    decision = quality_gate.evaluate(candidate_text, request_record)
    if quality_gate.metrics is not observed_metrics:
        observed_metrics.record_gate(decision)
    if not decision.accepted:
        observed_metrics.record_fallback(decision.code, neural_attempted=False)
        return GenerationResolution(
            metadata={
                "mode": "deterministic-fallback",
                "fallbackUsed": True,
                "fallbackReasonCode": decision.code,
                "fallbackSource": "voltforge-deterministic-tools",
                "neuralAttempted": True,
                "neuralArtifactId": model_health.get("artifactId"),
                "qualityGate": decision.public_metadata(),
            },
            deterministic_value=deterministic_factory(),
        )

    return GenerationResolution(
        metadata={
            "mode": "neural-quality-gated",
            "fallbackUsed": False,
            "fallbackReasonCode": None,
            "fallbackSource": None,
            "neuralAttempted": True,
            "neuralArtifactId": model_health.get("artifactId"),
            "qualityGate": decision.public_metadata(),
        },
        response_text=decision.response_text,
        response_record=decision.response_record,
    )
