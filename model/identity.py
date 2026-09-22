"""Canonical identity contract for VoltForge-owned language model artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import re


PRODUCT_NAME = "VoltForge AI"
FAMILY_NAME = "VoltForge Domain Language Model"
FAMILY_SLUG = "vfdlm"
DEPLOYMENT_PROFILES = frozenset({"edge", "core", "server"})
ARTIFACT_ID_PATTERN_TEXT = (
    r"^vfdlm-g(?P<generation>[1-9][0-9]*)-"
    r"(?P<profile>edge|core|server)-"
    r"v(?P<version>(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*))"
    r"(?:-(?P<prerelease>[0-9A-Za-z][0-9A-Za-z.-]*))?$"
)
ARTIFACT_ID_PATTERN = re.compile(ARTIFACT_ID_PATTERN_TEXT)


@dataclass(frozen=True)
class ModelIdentity:
    artifact_id: str
    generation: int
    deployment_profile: str
    semantic_version: str
    prerelease: str | None = None

    @property
    def display_version(self) -> str:
        if self.prerelease:
            return f"{self.semantic_version}-{self.prerelease}"
        return self.semantic_version

    def health_fields(self) -> dict[str, object]:
        return {
            "productName": PRODUCT_NAME,
            "familyName": FAMILY_NAME,
            "familySlug": FAMILY_SLUG,
            "artifactId": self.artifact_id,
            "generation": self.generation,
            "deploymentProfile": self.deployment_profile,
            "semanticVersion": self.display_version,
        }


def parse_artifact_id(artifact_id: str) -> ModelIdentity:
    if not isinstance(artifact_id, str):
        raise ValueError("Model artifact ID must be a string.")
    match = ARTIFACT_ID_PATTERN.fullmatch(artifact_id)
    if match is None:
        raise ValueError(
            "Model artifact ID must match "
            "vfdlm-g{generation}-{edge|core|server}-v{semanticVersion}."
        )
    return ModelIdentity(
        artifact_id=artifact_id,
        generation=int(match.group("generation")),
        deployment_profile=match.group("profile"),
        semantic_version=match.group("version"),
        prerelease=match.group("prerelease"),
    )


def empty_identity_health() -> dict[str, object]:
    return {
        "productName": PRODUCT_NAME,
        "familyName": FAMILY_NAME,
        "familySlug": FAMILY_SLUG,
        "artifactId": None,
        "generation": None,
        "deploymentProfile": None,
        "semanticVersion": None,
    }
