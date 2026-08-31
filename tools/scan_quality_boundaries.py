"""Fail-closed static scan for the VoltForge-only quality boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable


AI_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = AI_ROOT.parent
PIPELINE_MANIFEST = AI_ROOT / "quality_pipeline.v1.json"

PROVIDER_MARKERS = (
    "api.openai.com",
    "openai",
    "anthropic",
    "cohere",
    "huggingface_hub",
    "transformers",
    "langchain",
    "google.generativeai",
    "generativelanguage.googleapis.com",
    "vertexai",
    "bedrock-runtime",
)
SECRET_PATTERNS = (
    ("provider-secret-prefix", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    (
        "private-key-material",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    ),
    (
        "hardcoded-credential-assignment",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret[_-]?key|access[_-]?token)\s*[:=]\s*['\"][^'\"]{16,}['\"]"
        ),
    ),
)
CODE_SUFFIXES = {".py", ".java", ".ts", ".tsx", ".js", ".jsx", ".json", ".xml", ".yml", ".yaml", ".txt"}
PRODUCTION_SPECS = (
    (AI_ROOT, ("api", "model", "engine", "context_compiler", "feedback_governance", "grounding", "internet_retrieval", "local_retrieval", "memory_store")),
    (AI_ROOT, ("config.py", "main.py", "web_search_engine.py", "requirements.txt", "requirements-gen1.txt", "requirements-dev.txt")),
    (WORKSPACE_ROOT / "Voltforge_BL", ("src/main/java/in/voltforge/api/ai", "src/main/java/in/voltforge/api/config/VoltforgeAiConfig.java", "pom.xml")),
    (WORKSPACE_ROOT / "Voltforge_UI", ("src/features/ai", "src/api/services.ts", "package.json")),
)
JSON_CONTRACT_PATHS = (
    "evaluation/metrics.v1.json",
    "evaluation/evaluation-report.schema.json",
    "api_contract/policy.v1.json",
    "api_contract/chat-request.schema.json",
    "api_contract/chat-response.schema.json",
    "api_contract/sse-event.schema.json",
    "context_compiler/policy.v1.json",
    "feedback_governance/policy.v1.json",
    "grounding/policy.v1.json",
    "internet_retrieval/policy.v1.json",
    "local_retrieval/policy.v1.json",
    "memory_store/policy.v1.json",
    "model/generation-quality-policy.v1.json",
    "model/gen1/optimization-policy.v1.json",
    "model/release-policy.v1.json",
    "model/registry/active_model.json",
    "model/registry/trust/trusted-keys.json",
    "quality_pipeline.v1.json",
)


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _iter_spec_files(root: Path, specs: Iterable[str]) -> tuple[list[Path], list[dict[str, str]]]:
    files: list[Path] = []
    violations: list[dict[str, str]] = []
    for raw_spec in specs:
        path = root / raw_spec
        if not path.exists():
            violations.append({"code": "missing-production-boundary", "path": raw_spec, "marker": "required-path"})
            continue
        if path.is_file():
            files.append(path)
            continue
        for candidate in sorted(path.rglob("*")):
            if candidate.is_file() and candidate.suffix.lower() in CODE_SUFFIXES:
                files.append(candidate)
    return files, violations


def _scan_text_files() -> list[dict[str, str]]:
    violations: list[dict[str, str]] = []
    seen: set[Path] = set()
    for root, specs in PRODUCTION_SPECS:
        files, missing = _iter_spec_files(root, specs)
        violations.extend(
            {**item, "path": f"{root.name}/{item['path']}"} for item in missing
        )
        for path in files:
            if path in seen:
                continue
            seen.add(path)
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                violations.append({"code": "unreadable-production-file", "path": str(path), "marker": "utf8"})
                continue
            relative = str(path.relative_to(WORKSPACE_ROOT)).replace("\\", "/")
            lower = text.lower()
            for marker in PROVIDER_MARKERS:
                if re.search(r"(?<![a-z0-9_])" + re.escape(marker) + r"(?![a-z0-9_])", lower):
                    violations.append({"code": "forbidden-generation-provider", "path": relative, "marker": marker})
            for marker, pattern in SECRET_PATTERNS:
                if pattern.search(text):
                    violations.append({"code": "credential-material", "path": relative, "marker": marker})
    return violations


def _load_json(path: Path, violations: list[dict[str, str]]) -> Any | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        violations.append({"code": "malformed-json", "path": str(path.relative_to(AI_ROOT)).replace("\\", "/"), "marker": "json"})
        return None


def _validate_contract_json() -> list[dict[str, str]]:
    violations: list[dict[str, str]] = []
    documents: dict[str, Any] = {}
    for relative in JSON_CONTRACT_PATHS:
        path = AI_ROOT / relative
        if not path.is_file():
            violations.append({"code": "missing-contract-json", "path": relative, "marker": "required-json"})
            continue
        value = _load_json(path, violations)
        if value is not None:
            documents[relative] = value

    retrieval = documents.get("internet_retrieval/policy.v1.json")
    if isinstance(retrieval, dict):
        providers = retrieval.get("providers")
        network = retrieval.get("network")
        if not isinstance(providers, list) or not providers:
            violations.append({"code": "unsafe-web-policy", "path": "internet_retrieval/policy.v1.json", "marker": "providers"})
        elif any(provider.get("resultUrlsAreCitationsOnly") is not True or provider.get("resultUrlFetchingAllowed") is not False for provider in providers if isinstance(provider, dict)):
            violations.append({"code": "unsafe-web-policy", "path": "internet_retrieval/policy.v1.json", "marker": "citation-only-results"})
        if not isinstance(network, dict) or any(
            network.get(key) != expected
            for key, expected in (
                ("httpsOnly", True),
                ("allowedPort", 443),
                ("redirectPolicy", "reject-all"),
                ("maximumRedirects", 0),
                ("environmentProxyUseAllowed", False),
                ("privateOrReservedAddressUseAllowed", False),
                ("validateEveryResolvedAddress", True),
                ("retryCount", 0),
            )
        ):
            violations.append({"code": "unsafe-web-policy", "path": "internet_retrieval/policy.v1.json", "marker": "network-boundary"})

    registry = documents.get("model/registry/active_model.json")
    if isinstance(registry, dict):
        artifacts = registry.get("artifacts")
        active_id = registry.get("activeArtifactId")
        if not isinstance(artifacts, list) or not isinstance(registry.get("signature"), dict):
            violations.append({"code": "malformed-artifact-registry", "path": "model/registry/active_model.json", "marker": "registry-shape"})
        else:
            artifact_ids: set[str] = set()
            for artifact in artifacts:
                if not isinstance(artifact, dict):
                    violations.append({"code": "malformed-artifact-registry", "path": "model/registry/active_model.json", "marker": "artifact-entry"})
                    continue
                artifact_id = artifact.get("artifactId")
                if not isinstance(artifact_id, str) or artifact_id in artifact_ids:
                    violations.append({"code": "malformed-artifact-registry", "path": "model/registry/active_model.json", "marker": "unique-artifact-id"})
                if isinstance(artifact_id, str):
                    artifact_ids.add(artifact_id)
                if artifact.get("releaseStatus") not in {"experimental", "candidate", "stable", "rejected"}:
                    violations.append({"code": "malformed-artifact-registry", "path": "model/registry/active_model.json", "marker": "release-status"})
                if not isinstance(artifact.get("activationEligible"), bool):
                    violations.append({"code": "malformed-artifact-registry", "path": "model/registry/active_model.json", "marker": "activation-eligible"})
                root = artifact.get("root")
                if not isinstance(root, str) or Path(root).is_absolute() or ".." in Path(root).parts:
                    violations.append({"code": "unsafe-artifact-path", "path": "model/registry/active_model.json", "marker": "relative-root"})
            if active_id is not None and active_id not in artifact_ids:
                violations.append({"code": "malformed-artifact-registry", "path": "model/registry/active_model.json", "marker": "active-artifact-reference"})
            if active_id is not None:
                active = next((item for item in artifacts if isinstance(item, dict) and item.get("artifactId") == active_id), None)
                if not isinstance(active, dict) or active.get("releaseStatus") != "stable" or active.get("activationEligible") is not True:
                    violations.append({"code": "unsafe-active-artifact", "path": "model/registry/active_model.json", "marker": "stable-signed-artifact"})
    return violations


def scan_quality_boundaries() -> dict[str, Any]:
    violations = _scan_text_files()
    violations.extend(_validate_contract_json())
    unique = sorted({(item["code"], item["path"], item["marker"]): item for item in violations}.values(), key=lambda item: (item["code"], item["path"], item["marker"]))
    scanned_paths: set[Path] = set()
    for root, specs in PRODUCTION_SPECS:
        files, _ = _iter_spec_files(root, specs)
        scanned_paths.update(files)
    scanned_paths.update(AI_ROOT / relative for relative in JSON_CONTRACT_PATHS if (AI_ROOT / relative).is_file())
    return {
        "schemaVersion": 1,
        "scanId": "vfai032-quality-boundaries-v1",
        "passed": not unique,
        "filesScanned": len(scanned_paths),
        "violations": unique,
        "networkAccessed": False,
        "rawContentStored": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Retained for explicit no-write CI invocation.")
    args = parser.parse_args(argv)
    del args
    result = scan_quality_boundaries()
    result["scanSha256"] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
