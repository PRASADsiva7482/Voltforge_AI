"""Exact evidence catalog, conflict comparison, and visible uncertainty gate."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any, Iterable, Mapping, Sequence

from grounding.schema import (
    EvidenceConflict,
    GroundedClaim,
    GroundingCitation,
    GroundingContractError,
    GroundingReport,
    GroundingUncertainty,
    load_policy,
    sha256_json,
)


_DATASHEET_RATING = re.compile(
    r"\b(?:absolute\s+maximum|datasheet|operating|rated|rating|supply|logic|input|output|forward|maximum|minimum|typical)?\s*"
    r"(?:voltage|current|frequency|temperature|power|resistance|capacitance|clock)?[^\n]{0,48}"
    r"(?:\d+(?:\.\d+)?\s*(?:mv|v|ma|a|ohm|ω|kohm|kω|mhz|khz|hz|°c|c|w|mw|uf|µf|nf|pf))\b",
    re.IGNORECASE,
)
_PIN_CAPABILITY = re.compile(
    r"\b(?:pin|gpio|d\d+|a\d+)\b[^\n]{0,64}\b(?:supports?|capable|available|provides?|allows?|is|isn't|does\s+not)\b[^\n]{0,48}"
    r"\b(?:pwm|adc|analog|digital|interrupt|i2c|spi|uart|tx|rx|sda|scl|mosi|miso|sck|5v|3\.3v)\b|"
    r"\b(?:pwm|adc|analog|interrupt|i2c|spi|uart)\b[^\n]{0,48}\b(?:pin|gpio|d\d+|a\d+)\b",
    re.IGNORECASE,
)
_LIBRARY_API = re.compile(
    r"(?:#include\s*[<\"][^>\"]+[>\"]|\b(?:library|sdk|api|function|method|class|firmware\s+api)\b[^\n]{0,80}"
    r"(?:\w+\s*\([^\n)]*\)|version\s+\d+(?:\.\d+)+))",
    re.IGNORECASE,
)
_CURRENT_WEB = re.compile(
    r"\b(?:currently|latest|newest|recent|today|as\s+of|released?|available\s+now)\b|"
    r"\bcurrent\s+(?:version|release|status|availability|price|documentation|support)\b",
    re.IGNORECASE,
)
_UNCERTAINTY = re.compile(
    r"\b(?:unknown|uncertain|cannot\s+verify|can't\s+verify|insufficient\s+evidence|not\s+verified|evidence\s+conflicts?)\b",
    re.IGNORECASE,
)
_CITATION_MARKER = re.compile(r"\[(citation:[a-z0-9._:-]{3,150})\]", re.IGNORECASE)
_TOKEN = re.compile(r"[a-z][a-z0-9_+.-]{2,}|0x[0-9a-f]+|\d+(?:\.\d+)?(?:mv|v|ma|a|mhz|khz|hz|ohm|kohm|w|mw|uf|nf|pf)?", re.IGNORECASE)
_CRITICAL = re.compile(
    r"0x[0-9a-f]+|\b(?:[da]\d+|gpio\d+|\d+(?:\.\d+)?\s*(?:mv|v|ma|a|mhz|khz|hz|ohm|kohm|w|mw|uf|µf|nf|pf))\b|\b[a-z_]\w*\s*\(",
    re.IGNORECASE,
)
_STOP = {
    "about", "available", "because", "circuit", "component", "current", "datasheet",
    "from", "into", "library", "maximum", "minimum", "must", "rating", "should",
    "supports", "that", "this", "typical", "voltage", "with",
}


@dataclass(frozen=True)
class _Assertion:
    subject: str
    property: str
    value: str
    claim_type: str


@dataclass
class _EvidenceEntry:
    citation: GroundingCitation
    text: str
    assertions: list[_Assertion] = field(default_factory=list)


@dataclass(frozen=True)
class GroundingResult:
    text: str
    report: GroundingReport
    citations: tuple[GroundingCitation, ...]


class NeuralGroundingError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _clean(value: Any, maximum: int) -> str:
    return " ".join(str(value or "").split())[:maximum].strip()


def _tokens(value: str) -> set[str]:
    return {
        token.casefold().replace(" ", "")
        for token in _TOKEN.findall(value)
        if token.casefold() not in _STOP
    }


def _critical(value: str) -> set[str]:
    return {
        re.sub(r"\s+", "", token.casefold()).rstrip("(")
        for token in _CRITICAL.findall(value)
    }


def _subject(value: str) -> str:
    tokens = [
        token
        for token in sorted(_tokens(value))
        if token not in {"evidence", "internet", "local", "retrieval", "source"}
    ]
    return " ".join(tokens[:8]) or "unspecified-subject"


def classify_claim(text: str) -> tuple[str, ...]:
    kinds = []
    if _DATASHEET_RATING.search(text):
        kinds.append("datasheet-rating")
    if _PIN_CAPABILITY.search(text):
        kinds.append("pin-capability")
    if _LIBRARY_API.search(text):
        kinds.append("library-api")
    if _CURRENT_WEB.search(text):
        kinds.append("current-web")
    return tuple(kinds)


def _evidence_kind(item: Mapping[str, Any]) -> str:
    tool = str(item.get("toolName") or "")
    policy = str((item.get("payload") or {}).get("policyId") or "")
    if tool == "tool:curated-local-retrieval" or policy == "vfai022-curated-local-retrieval-v1":
        return "local"
    if tool == "tool:secure-internet-evidence" or policy == "vfai023-secure-internet-evidence-v1":
        return "internet"
    if item.get("authority") == "deterministic":
        return "deterministic"
    return "project"


def _authority(kind: str) -> str:
    return "deterministic" if kind == "deterministic" else "retrieved" if kind in {"local", "internet"} else "project-state"


def _source_map(source_citations: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {
        str(item.get("citationId")): item
        for item in source_citations
        if isinstance(item, Mapping) and item.get("citationId")
    }


def _fact_assertion(result: Mapping[str, Any], title: str) -> list[_Assertion]:
    raw = result.get("fact")
    if not isinstance(raw, str):
        return []
    try:
        fact = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(fact, Mapping) or not fact.get("property"):
        return []
    property_name = str(fact["property"])
    value = json.dumps(fact.get("value"), sort_keys=True, separators=(",", ":"), default=str)
    lowered = property_name.casefold()
    claim_type = "pin-capability" if "pin" in lowered or "capab" in lowered else "library-api" if "api" in lowered or "function" in lowered else "datasheet-rating"
    return [_Assertion(_subject(title), property_name, value, claim_type)]


def _text_assertions(title: str, text: str) -> list[_Assertion]:
    assertions: list[_Assertion] = []
    subject = _subject(title)
    rating = re.search(
        r"\b(operating\s+voltage|logic\s+voltage|maximum\s+current|clock\s+frequency)\s*(?:is|:|=)\s*([0-9.]+\s*(?:mv|v|ma|a|mhz|khz|hz))",
        text,
        re.IGNORECASE,
    )
    if rating:
        assertions.append(
            _Assertion(subject, re.sub(r"\s+", "-", rating.group(1).casefold()), re.sub(r"\s+", "", rating.group(2).casefold()), "datasheet-rating")
        )
    capability = re.search(
        r"\b(pin\s+[a-z0-9]+|gpio\s*\d+)\s+(supports|does\s+not\s+support|is\s+not\s+capable\s+of|is\s+capable\s+of)\s+(pwm|adc|analog|interrupt|i2c|spi|uart)",
        text,
        re.IGNORECASE,
    )
    if capability:
        positive = "not" not in capability.group(2).casefold()
        assertions.append(
            _Assertion(subject, f"{capability.group(1).casefold()}:{capability.group(3).casefold()}", str(positive).lower(), "pin-capability")
        )
    return assertions


def _entry_from_result(
    result: Mapping[str, Any],
    *,
    kind: str,
    evidence_id: str,
    payload_sha: str,
    tool_version: str,
    sources: Mapping[str, Mapping[str, Any]],
) -> _EvidenceEntry | None:
    citation_id = str(result.get("citationId") or "")
    if not citation_id:
        return None
    source = sources.get(citation_id, {})
    title = _clean(result.get("title") or result.get("subject") or source.get("title") or citation_id, 500)
    snippet = _clean(result.get("snippet") or result.get("fact") or source.get("snippet") or title, 800)
    content_hash = str(result.get("contentSha256") or source.get("contentSha256") or sha256_json(result))
    source_id = _clean(result.get("sourceId") or source.get("sourceId") or f"grounding-{kind}-source", 160)
    source_id = re.sub(r"[^a-z0-9._:-]", "-", source_id.casefold()).strip("-")[:160]
    if len(source_id) < 3:
        source_id = f"grounding-{kind}-source"
    citation = GroundingCitation(
        citationId=citation_id,
        evidenceKind=kind,
        authority=_authority(kind),
        sourceId=source_id,
        sourceRevision=_clean(result.get("sourceRevision") or source.get("sourceRevision") or tool_version, 160),
        title=title,
        locator=source.get("locator") or source.get("url") or result.get("sourceUrl"),
        url=source.get("url") or result.get("sourceUrl"),
        snippet=snippet,
        contentSha256=content_hash,
        modelPayloadSha256=payload_sha,
        evidenceRefs=[evidence_id],
        supportStatus="reference-only",
        sourceTimestamp=result.get("retrievedAt") or source.get("retrievedAt"),
        untrustedContent=kind == "internet",
    )
    text = f"{title} {snippet} {json.dumps(result, sort_keys=True, default=str)}"
    return _EvidenceEntry(citation, text, [*_fact_assertion(result, title), *_text_assertions(title, snippet)])


def build_evidence_catalog(
    task_record: Mapping[str, Any],
    source_citations: Sequence[Mapping[str, Any]] = (),
) -> list[_EvidenceEntry]:
    policy = load_policy()
    maximum = int(policy["limits"]["maximumCitations"])
    sources = _source_map(source_citations)
    entries: list[_EvidenceEntry] = []
    project = task_record["input"]["projectContext"]
    project_payload = project.get("payload") or {}
    project_sha = sha256_json(project)
    project_id = re.sub(r"[^a-z0-9._:-]", "-", str(project.get("projectId") or "project-state").casefold()).strip("-")
    if len(project_id) < 3:
        project_id = "project-state"
    entries.append(
        _EvidenceEntry(
            GroundingCitation(
                citationId=f"citation:project:{project_sha[:24]}",
                evidenceKind="project",
                authority="project-state",
                sourceId=project_id[:160],
                sourceRevision=str(project["sourceProjectRevision"]),
                title="Bounded project state seen by the model",
                snippet=_clean(f"Board: {project.get('boardType') or 'unspecified'}", 800),
                contentSha256=project_sha,
                modelPayloadSha256=project_sha,
                evidenceRefs=[],
                supportStatus="reference-only",
            ),
            json.dumps(project_payload, sort_keys=True, default=str),
        )
    )
    for item in task_record["input"]["toolEvidence"]:
        payload = item.get("payload") or {}
        payload_sha = sha256_json(payload)
        kind = _evidence_kind(item)
        results = payload.get("results") if isinstance(payload, Mapping) else None
        if kind in {"local", "internet"} and isinstance(results, list):
            for result in results:
                if not isinstance(result, Mapping):
                    continue
                entry = _entry_from_result(
                    result,
                    kind=kind,
                    evidence_id=str(item["evidenceId"]),
                    payload_sha=payload_sha,
                    tool_version=str(item["toolVersion"]),
                    sources=sources,
                )
                if entry is not None:
                    entries.append(entry)
            continue
        citation_id = f"citation:{kind}:{hashlib.sha256(str(item['evidenceId']).encode()).hexdigest()[:24]}"
        title = _clean(item.get("summary") or item.get("toolName"), 500)
        citation = GroundingCitation(
            citationId=citation_id,
            evidenceKind=kind,
            authority=_authority(kind),
            sourceId=str(item["toolName"]),
            sourceRevision=str(item["toolVersion"]),
            title=title,
            snippet=title,
            contentSha256=payload_sha,
            modelPayloadSha256=payload_sha,
            evidenceRefs=[str(item["evidenceId"])],
            supportStatus="reference-only",
        )
        text = f"{title} {json.dumps(payload, sort_keys=True, default=str)}"
        entries.append(_EvidenceEntry(citation, text, _text_assertions(title, text)))
    unique: dict[str, _EvidenceEntry] = {}
    for entry in entries:
        unique.setdefault(entry.citation.citationId, entry)
    return list(unique.values())[:maximum]


def public_citation_catalog(
    task_record: Mapping[str, Any],
    source_citations: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    return [item.citation.model_dump(mode="json") for item in build_evidence_catalog(task_record, source_citations)]


def _conflicts(entries: Sequence[_EvidenceEntry]) -> list[EvidenceConflict]:
    policy = load_policy()
    groups: dict[tuple[str, str, str], list[tuple[_EvidenceEntry, _Assertion]]] = {}
    for entry in entries:
        for assertion in entry.assertions:
            groups.setdefault((assertion.claim_type, assertion.subject, assertion.property.casefold()), []).append((entry, assertion))
    conflicts = []
    for (claim_type, subject, property_name), rows in sorted(groups.items()):
        values = sorted({row.value for _, row in rows})[:8]
        citation_ids = sorted({entry.citation.citationId for entry, _ in rows})[:8]
        if len(values) < 2 or len(citation_ids) < 2:
            continue
        identity = {"claimType": claim_type, "subject": subject, "property": property_name, "values": values, "citations": citation_ids}
        conflicts.append(
            EvidenceConflict(
                conflictId=f"conflict:grounding:{sha256_json(identity)[:24]}",
                claimType=claim_type,
                subject=subject,
                property=property_name,
                valueHashes=[hashlib.sha256(value.encode()).hexdigest() for value in values],
                citationIds=citation_ids,
                evidenceRefs=sorted({reference for entry, _ in rows for reference in entry.citation.evidenceRefs})[:16],
            )
        )
    return conflicts[: int(policy["conflictPolicy"]["maximumConflicts"])]


def _supports(claim: str, kind: str, entry: _EvidenceEntry) -> bool:
    allowed = set(load_policy()["claimTypes"][kind]["allowedEvidenceKinds"])
    if entry.citation.evidenceKind not in allowed:
        return False
    claim_tokens = _tokens(claim)
    evidence_tokens = _tokens(entry.text)
    critical = _critical(claim)
    if critical and not critical.issubset(_critical(entry.text)):
        return False
    overlap = claim_tokens.intersection(evidence_tokens)
    minimum = 1 if len(claim_tokens) <= 2 else 2
    return len(overlap) >= minimum


def _matching_conflict(claim: str, kinds: Sequence[str], conflicts: Sequence[EvidenceConflict]) -> EvidenceConflict | None:
    claim_tokens = _tokens(claim)
    for conflict in conflicts:
        if conflict.claimType not in kinds:
            continue
        if claim_tokens.intersection(_tokens(conflict.subject)):
            return conflict
    return None


def _uncertainty_line(kind: str, conflict: bool = False) -> str:
    labels = {
        "datasheet-rating": "datasheet rating",
        "pin-capability": "pin capability",
        "library-api": "library API claim",
        "current-web": "current web claim",
    }
    label = labels[kind]
    if conflict:
        return f"Available evidence conflicts about the requested {label}; verify the exact variant and cited source before use."
    return f"I cannot verify the requested {label} from the bounded evidence available."


def enforce_response_grounding(
    response_text: str,
    task_record: Mapping[str, Any],
    source_citations: Sequence[Mapping[str, Any]] = (),
) -> GroundingResult:
    policy = load_policy()
    entries = build_evidence_catalog(task_record, source_citations)
    by_id = {item.citation.citationId: item for item in entries}
    conflicts = _conflicts(entries)
    claims: list[GroundedClaim] = []
    output_lines: list[str] = []
    used_citations: set[str] = set()
    missing: set[str] = set()
    maximum_claims = int(policy["limits"]["maximumClaims"])
    raw_lines = response_text.splitlines()
    classified_line_count = sum(
        bool(classify_claim(_CITATION_MARKER.sub("", line))) for line in raw_lines
    )
    ordinary_claim_limit = maximum_claims - 1 if classified_line_count > maximum_claims else maximum_claims
    overflow_recorded = False
    for line in raw_lines:
        explicit_ids = [match.group(1) for match in _CITATION_MARKER.finditer(line)]
        clean_line = _CITATION_MARKER.sub("", line).rstrip()
        kinds = classify_claim(clean_line)
        if not kinds:
            output_lines.append(clean_line)
            continue
        if len(claims) >= ordinary_claim_limit:
            kind = kinds[0]
            missing.add(policy["claimTypes"][kind]["missingReasonCode"])
            if not overflow_recorded:
                claims.append(
                    GroundedClaim(
                        claimId=f"claim:grounding:{sha256_json({'overflow': classified_line_count})[:24]}",
                        claimTypes=list(kinds),
                        supportStatus="unsupported",
                        reasonCode="CLAIM_LIMIT_EXCEEDED",
                        visibleTextTransformed=True,
                    )
                )
                overflow_recorded = True
            output_lines.append(_uncertainty_line(kind))
            continue
        claim_id = f"claim:grounding:{sha256_json({'index': len(claims), 'line': clean_line})[:24]}"
        if _UNCERTAINTY.search(clean_line):
            for kind in kinds:
                missing.add(policy["claimTypes"][kind]["missingReasonCode"])
            claims.append(
                GroundedClaim(
                    claimId=claim_id,
                    claimTypes=list(kinds),
                    supportStatus="unsupported",
                    reasonCode="VISIBLE_UNCERTAINTY_PRESENT",
                )
            )
            output_lines.append(clean_line)
            continue
        if any(citation_id not in by_id for citation_id in explicit_ids):
            explicit_ids = []
        supporting = []
        candidates = [by_id[item] for item in explicit_ids] if explicit_ids else entries
        for entry in candidates:
            if all(_supports(clean_line, kind, entry) for kind in kinds):
                supporting.append(entry)
        supporting.sort(
            key=lambda item: (
                policy["evidencePrecedence"].index(item.citation.evidenceKind),
                item.citation.citationId,
            )
        )
        conflict = _matching_conflict(clean_line, kinds, conflicts)
        if conflict is not None:
            kind = str(conflict.claimType)
            replacement = _uncertainty_line(kind, conflict=True)
            claims.append(
                GroundedClaim(
                    claimId=claim_id,
                    claimTypes=list(kinds),
                    supportStatus="conflicted",
                    reasonCode="EVIDENCE_CONFLICT",
                    citationIds=conflict.citationIds[:2],
                    evidenceRefs=conflict.evidenceRefs[:8],
                    visibleTextTransformed=True,
                )
            )
            used_citations.update(conflict.citationIds[:2])
            output_lines.append(replacement)
            continue
        if not supporting:
            kind = kinds[0]
            missing.add(policy["claimTypes"][kind]["missingReasonCode"])
            claims.append(
                GroundedClaim(
                    claimId=claim_id,
                    claimTypes=list(kinds),
                    supportStatus="unsupported",
                    reasonCode=policy["claimTypes"][kind]["missingReasonCode"],
                    visibleTextTransformed=True,
                )
            )
            output_lines.append(_uncertainty_line(kind))
            continue
        selected = supporting[: int(policy["limits"]["maximumCitationsPerClaim"])]
        citation_ids = [item.citation.citationId for item in selected]
        evidence_refs = sorted({reference for item in selected for reference in item.citation.evidenceRefs})
        claims.append(
            GroundedClaim(
                claimId=claim_id,
                claimTypes=list(kinds),
                supportStatus="supported",
                reasonCode="CLAIM_EVIDENCE_BOUND",
                citationIds=citation_ids,
                evidenceRefs=evidence_refs,
            )
        )
        used_citations.update(citation_ids)
        markers = " ".join(f"[{citation_id}]" for citation_id in citation_ids if citation_id not in explicit_ids)
        output_lines.append(f"{clean_line} {markers}".rstrip())
    supported = sum(item.supportStatus == "supported" for item in claims)
    unsupported = sum(item.supportStatus == "unsupported" for item in claims)
    conflicted = sum(item.supportStatus == "conflicted" for item in claims)
    status = "conflicted" if conflicted else "uncertain" if unsupported else "grounded" if claims else "no-high-risk-claims"
    maximum_confidence = (
        float(policy["conflictPolicy"]["maximumConfidence"])
        if conflicted
        else float(policy["uncertaintyPolicy"]["maximumConfidence"])
        if unsupported
        else min((float(policy["claimTypes"][kind]["maximumConfidence"]) for item in claims for kind in item.claimTypes), default=0.8)
    )
    selected_citations = []
    claim_map = {citation_id: [item.claimId for item in claims if citation_id in item.citationIds] for citation_id in used_citations}
    conflict_ids = {citation_id for conflict in conflicts for citation_id in conflict.citationIds}
    for citation_id in sorted(used_citations):
        entry = by_id[citation_id]
        selected_citations.append(
            entry.citation.model_copy(
                update={
                    "claimIds": claim_map.get(citation_id, []),
                    "supportStatus": "conflicted" if citation_id in conflict_ids else "supported",
                }
            )
        )
    selected_by_id = {item.citationId: item for item in selected_citations}
    public_citations = [
        selected_by_id.get(entry.citation.citationId, entry.citation)
        for entry in entries
    ]
    used_refs = sorted({reference for item in selected_citations for reference in item.evidenceRefs})
    identity = {
        "claims": [item.model_dump(mode="json") for item in claims],
        "conflicts": [item.model_dump(mode="json") for item in conflicts],
        "citations": [item.citationId for item in public_citations],
        "responseSha256": hashlib.sha256("\n".join(output_lines).encode()).hexdigest(),
    }
    uncertainty = GroundingUncertainty(
        level="high" if conflicted else "medium" if unsupported else "none",
        reasonCode="EVIDENCE_CONFLICT" if conflicted else "EVIDENCE_REQUIRED" if unsupported else "NO_GROUNDING_UNCERTAINTY",
        missingEvidence=sorted(missing)[: int(policy["uncertaintyPolicy"]["maximumMissingEvidenceItems"])],
    )
    report = GroundingReport(
        policySha256=policy["policySha256"],
        reportId=f"vf-grounding-report-v1-{sha256_json(identity)[:24]}",
        status=status,
        claims=claims,
        conflicts=conflicts,
        claimCount=len(claims),
        supportedClaimCount=supported,
        unsupportedClaimCount=unsupported,
        conflictedClaimCount=conflicted,
        citationCount=len(public_citations),
        usedEvidenceRefs=used_refs,
        uncertainty=uncertainty,
        maximumConfidence=maximum_confidence,
    )
    return GroundingResult("\n".join(output_lines), report, tuple(public_citations))


def validate_neural_claim(
    text: str,
    evidence_refs: Sequence[str],
    evidence: Mapping[str, Mapping[str, Any]],
) -> None:
    kinds = classify_claim(text)
    if not kinds:
        return
    entries = []
    for reference in evidence_refs:
        item = evidence.get(reference)
        if item is None:
            continue
        kind = _evidence_kind(item)
        payload = item.get("payload") or {}
        payload_sha = sha256_json(payload)
        citation = GroundingCitation(
            citationId=f"citation:neural:{hashlib.sha256(reference.encode()).hexdigest()[:24]}",
            evidenceKind=kind,
            authority=_authority(kind),
            sourceId=str(item.get("toolName") or "tool:unknown"),
            sourceRevision=str(item.get("toolVersion") or "1.0.0"),
            title=_clean(item.get("summary") or item.get("toolName"), 500),
            snippet=_clean(json.dumps(payload, sort_keys=True, default=str), 800),
            contentSha256=payload_sha,
            modelPayloadSha256=payload_sha,
            evidenceRefs=[reference],
            supportStatus="reference-only",
            untrustedContent=kind == "internet",
        )
        entry_text = f"{item.get('summary', '')} {json.dumps(payload, sort_keys=True, default=str)}"
        entries.append(_EvidenceEntry(citation, entry_text, _text_assertions(citation.title, entry_text)))
    codes = {
        "datasheet-rating": "QG_DATASHEET_EVIDENCE_REQUIRED",
        "pin-capability": "QG_PIN_CAPABILITY_EVIDENCE_REQUIRED",
        "library-api": "QG_LIBRARY_API_EVIDENCE_REQUIRED",
        "current-web": "QG_CURRENT_WEB_EVIDENCE_REQUIRED",
    }
    for kind in kinds:
        if not any(_supports(text, kind, entry) for entry in entries):
            raise NeuralGroundingError(codes[kind], "A high-risk neural claim lacks exact eligible evidence.")
    if _matching_conflict(text, kinds, _conflicts(entries)) is not None:
        raise NeuralGroundingError("QG_EVIDENCE_CONFLICT", "Neural output cannot resolve conflicting evidence as fact.")
