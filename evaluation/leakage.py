"""Held-out fixture freezing and training/retrieval leakage prevention."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Iterator, Sequence


AI_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ROOT = Path(__file__).resolve().parent
FIXTURE_PATH = EVALUATION_ROOT / "fixtures" / "v1" / "suite.jsonl"
METRICS_PATH = EVALUATION_ROOT / "metrics.v1.json"
CASE_SCHEMA_PATH = EVALUATION_ROOT / "evaluation-case.schema.json"
MANIFEST_PATH = EVALUATION_ROOT / "frozen-manifest.v1.json"
DEFAULT_SEMANTIC_THRESHOLD = 0.86
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", re.IGNORECASE)
SPACE_PATTERN = re.compile(r"\s+")


class FrozenSuiteError(RuntimeError):
    """The held-out suite or its freeze manifest is invalid."""


@dataclass(frozen=True)
class ProtectedSegment:
    case_id: str
    normalized_hash: str
    semantic_fingerprint: str
    shingles: frozenset[str]
    token_count: int


@dataclass(frozen=True)
class LeakageMatch:
    case_id: str
    match_type: str
    similarity: float
    protected_hash: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "caseId": self.case_id,
            "matchType": self.match_type,
            "similarity": round(self.similarity, 6),
            "protectedHash": self.protected_hash,
        }


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    content = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def normalized_text(value: str) -> str:
    tokens = TOKEN_PATTERN.findall(value.casefold())
    return SPACE_PATTERN.sub(" ", " ".join(tokens)).strip()


def text_hash(value: str) -> str:
    return sha256_bytes(normalized_text(value).encode("utf-8"))


def semantic_shingles(value: str, width: int = 3) -> frozenset[str]:
    tokens = TOKEN_PATTERN.findall(value.casefold())
    if len(tokens) < width:
        return frozenset(tokens)
    return frozenset(" ".join(tokens[index : index + width]) for index in range(len(tokens) - width + 1))


def semantic_fingerprint(value: str) -> str:
    shingles = sorted(semantic_shingles(value))
    return sha256_bytes("\n".join(shingles).encode("utf-8"))


def semantic_similarity(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left.intersection(right)) / len(left.union(right))


def string_leaves(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from string_leaves(child)
    elif isinstance(value, list):
        for child in value:
            yield from string_leaves(child)


def load_cases(path: Path = FIXTURE_PATH) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise FrozenSuiteError(f"Could not read evaluation fixtures: {path}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as error:
            raise FrozenSuiteError(f"Invalid fixture JSON at line {line_number}") from error
        if not isinstance(case, dict):
            raise FrozenSuiteError(f"Fixture line {line_number} must be an object")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise FrozenSuiteError(f"Fixture line {line_number} has no case ID")
        if case_id in seen:
            raise FrozenSuiteError(f"Duplicate evaluation case ID: {case_id}")
        seen.add(case_id)
        cases.append(case)
    if not cases:
        raise FrozenSuiteError("The evaluation suite contains no cases")
    return cases


def protected_texts(case: dict[str, Any]) -> list[str]:
    values = [*string_leaves(case.get("input", {})), *string_leaves(case.get("assertions", []))]
    protected: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = normalized_text(value)
        token_count = len(TOKEN_PATTERN.findall(normalized))
        if len(normalized) < 24 or token_count < 4 or normalized in seen:
            continue
        seen.add(normalized)
        protected.append(value)
    return protected


def protected_segments(cases: Sequence[dict[str, Any]]) -> list[ProtectedSegment]:
    segments: list[ProtectedSegment] = []
    for case in cases:
        for value in protected_texts(case):
            shingles = semantic_shingles(value)
            segments.append(
                ProtectedSegment(
                    case_id=str(case["id"]),
                    normalized_hash=text_hash(value),
                    semantic_fingerprint=semantic_fingerprint(value),
                    shingles=shingles,
                    token_count=len(TOKEN_PATTERN.findall(normalized_text(value))),
                )
            )
    return segments


def build_manifest_data(
    fixture_path: Path = FIXTURE_PATH,
    metrics_path: Path = METRICS_PATH,
) -> dict[str, Any]:
    cases = load_cases(fixture_path)
    case_entries = []
    segment_entries = []
    for case in cases:
        record_hash = sha256_bytes(canonical_json(case).encode("utf-8"))
        case_entries.append({"id": case["id"], "recordSha256": record_hash})
        for segment in protected_segments([case]):
            segment_entries.append(
                {
                    "caseId": segment.case_id,
                    "normalizedSha256": segment.normalized_hash,
                    "semanticFingerprint": segment.semantic_fingerprint,
                    "tokenCount": segment.token_count,
                }
            )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    suite_material = canonical_json(
        {
            "cases": case_entries,
            "metricsSha256": sha256_file(metrics_path),
            "caseSchemaSha256": sha256_file(CASE_SCHEMA_PATH),
            "suiteVersion": metrics.get("suiteVersion"),
        }
    )
    return {
        "schemaVersion": 1,
        "suiteId": "vfai-release-suite-v1",
        "suiteVersion": "1.0.0",
        "freezeStatus": "locked",
        "fixturePath": str(fixture_path.relative_to(AI_ROOT)).replace("\\", "/"),
        "fixtureFileSha256": sha256_file(fixture_path),
        "metricsPath": str(metrics_path.relative_to(AI_ROOT)).replace("\\", "/"),
        "metricsFileSha256": sha256_file(metrics_path),
        "caseSchemaPath": str(CASE_SCHEMA_PATH.relative_to(AI_ROOT)).replace("\\", "/"),
        "caseSchemaFileSha256": sha256_file(CASE_SCHEMA_PATH),
        "suiteSha256": sha256_bytes(suite_material.encode("utf-8")),
        "caseCount": len(cases),
        "metricCount": len(metrics.get("metrics", [])),
        "protectedSegmentCount": len(segment_entries),
        "cases": case_entries,
        "protectedSegments": segment_entries,
        "leakagePolicy": {
            "excludedFrom": ["training", "retrieval"],
            "exactMethod": "SHA-256 of normalized token text",
            "semanticMethod": "Jaccard similarity over normalized token trigrams",
            "semanticThreshold": DEFAULT_SEMANTIC_THRESHOLD,
            "minimumProtectedTokens": 4,
            "gate": "evaluation.leakage.exclude_held_out_records",
        },
    }


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FrozenSuiteError(f"Could not read frozen evaluation manifest: {path}") from error
    if not isinstance(manifest, dict) or manifest.get("freezeStatus") != "locked":
        raise FrozenSuiteError("Evaluation manifest is not locked")
    return manifest


def verify_frozen_suite(manifest_path: Path = MANIFEST_PATH) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    expected = build_manifest_data(FIXTURE_PATH, METRICS_PATH)
    immutable_fields = (
        "suiteId",
        "suiteVersion",
        "fixturePath",
        "fixtureFileSha256",
        "metricsPath",
        "metricsFileSha256",
        "caseSchemaPath",
        "caseSchemaFileSha256",
        "suiteSha256",
        "caseCount",
        "metricCount",
        "protectedSegmentCount",
        "cases",
        "protectedSegments",
        "leakagePolicy",
    )
    mismatches = [field for field in immutable_fields if manifest.get(field) != expected.get(field)]
    if mismatches:
        raise FrozenSuiteError(
            "Frozen evaluation suite does not match its manifest: " + ", ".join(mismatches)
        )
    return manifest


class HeldOutRegistry:
    def __init__(self, cases: Sequence[dict[str, Any]], semantic_threshold: float):
        self.segments = protected_segments(cases)
        self.semantic_threshold = semantic_threshold
        self.exact_hashes: dict[str, ProtectedSegment] = {
            segment.normalized_hash: segment for segment in self.segments
        }

    @classmethod
    def from_frozen_suite(cls) -> "HeldOutRegistry":
        manifest = verify_frozen_suite()
        threshold = float(manifest["leakagePolicy"]["semanticThreshold"])
        return cls(load_cases(), threshold)

    def match_text(self, value: str) -> LeakageMatch | None:
        normalized = normalized_text(value)
        if len(normalized) < 24:
            return None
        exact = self.exact_hashes.get(text_hash(value))
        if exact is not None:
            return LeakageMatch(exact.case_id, "exact", 1.0, exact.normalized_hash)
        candidate_shingles = semantic_shingles(value)
        candidate_tokens = len(TOKEN_PATTERN.findall(normalized))
        if candidate_tokens < 4:
            return None
        best: tuple[float, ProtectedSegment] | None = None
        for segment in self.segments:
            length_ratio = min(candidate_tokens, segment.token_count) / max(candidate_tokens, segment.token_count)
            if length_ratio < 0.5:
                continue
            similarity = semantic_similarity(candidate_shingles, segment.shingles)
            if similarity >= self.semantic_threshold and (best is None or similarity > best[0]):
                best = (similarity, segment)
        if best is None:
            return None
        return LeakageMatch(best[1].case_id, "semantic", best[0], best[1].normalized_hash)

    def match_record(self, record: Any) -> LeakageMatch | None:
        leaves = [value for value in string_leaves(record) if len(normalized_text(value)) >= 24]
        combined = " ".join(leaves)
        for value in [*leaves, combined]:
            match = self.match_text(value)
            if match is not None:
                return match
        return None


@lru_cache(maxsize=1)
def get_held_out_registry() -> HeldOutRegistry:
    return HeldOutRegistry.from_frozen_suite()


def exclude_held_out_records(
    records: Iterable[dict[str, Any]],
    registry: HeldOutRegistry | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    active_registry = registry or get_held_out_registry()
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        match = active_registry.match_record(record)
        if match is None:
            accepted.append(record)
        else:
            rejected.append({"recordIndex": index, **match.as_dict()})
    return accepted, rejected


def default_corpus_paths() -> list[Path]:
    artifacts = AI_ROOT / "model" / "artifacts"
    return [AI_ROOT / "dataset.txt", *sorted(artifacts.glob("*.jsonl"))]


def _iter_corpus_records(path: Path) -> Iterator[tuple[int, Any]]:
    if path.name == "dataset.txt":
        content = path.read_text(encoding="utf-8")
        for index, chunk in enumerate(content.split("[Q]")):
            if "[A]" in chunk:
                question, answer = chunk.split("[A]", 1)
                yield index, {"question": question.strip(), "answer": answer.strip()}
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError:
                yield line_number, {"raw": line.strip()}


def scan_corpora(
    corpus_paths: Sequence[Path] | None = None,
    registry: HeldOutRegistry | None = None,
) -> dict[str, Any]:
    active_registry = registry or get_held_out_registry()
    paths = list(corpus_paths or default_corpus_paths())
    collisions: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    total_records = 0
    for path in paths:
        if not path.is_file():
            files.append({"path": str(path.relative_to(AI_ROOT)).replace("\\", "/"), "state": "missing", "records": 0})
            continue
        record_count = 0
        for record_number, record in _iter_corpus_records(path):
            record_count += 1
            total_records += 1
            match = active_registry.match_record(record)
            if match is not None:
                collisions.append(
                    {
                        "path": str(path.relative_to(AI_ROOT)).replace("\\", "/"),
                        "record": record_number,
                        **match.as_dict(),
                    }
                )
        files.append(
            {
                "path": str(path.relative_to(AI_ROOT)).replace("\\", "/"),
                "state": "scanned",
                "records": record_count,
                "sha256": sha256_file(path),
            }
        )
    return {
        "status": "pass" if not collisions else "fail",
        "exactAndSemanticCollisions": len(collisions),
        "recordsScanned": total_records,
        "filesScanned": sum(1 for item in files if item["state"] == "scanned"),
        "semanticThreshold": active_registry.semantic_threshold,
        "files": files,
        "collisions": collisions,
    }
