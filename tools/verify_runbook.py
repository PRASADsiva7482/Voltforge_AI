"""Verify the content-free VFAI-035 operations runbook contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


AI_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = AI_ROOT / "docs" / "runbook-contract.v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON contract must be an object: {path.name}")
    return value


def verify() -> dict[str, Any]:
    contract = _read_json(CONTRACT_PATH)
    if contract.get("schemaVersion") != 1 or contract.get("contractId") != "vfai035-runbook-v1":
        raise ValueError("VFAI-035 runbook contract identity is invalid")
    runbook_path = AI_ROOT / str(contract.get("runbookPath", ""))
    if not runbook_path.is_file():
        raise ValueError("VFAI-035 runbook is missing")
    runbook = runbook_path.read_text(encoding="utf-8")

    sections = contract.get("requiredSections")
    commands = contract.get("requiredCommands")
    claims = contract.get("requiredClaims")
    sources = contract.get("requiredSourceFiles")
    forbidden = contract.get("forbiddenClaims")
    if not all(isinstance(values, list) for values in (sections, commands, claims, sources, forbidden)):
        raise ValueError("VFAI-035 runbook contract lists are invalid")

    missing_sections = [value for value in sections if not isinstance(value, str) or value not in runbook]
    missing_commands = [value for value in commands if not isinstance(value, str) or value not in runbook]
    missing_claims = [value for value in claims if not isinstance(value, str) or value not in runbook]
    missing_sources = [
        value
        for value in sources
        if not isinstance(value, str) or not (AI_ROOT / value).is_file()
    ]
    forbidden_claims = [value for value in forbidden if isinstance(value, str) and value in runbook]
    if missing_sections or missing_commands or missing_claims or missing_sources or forbidden_claims:
        raise ValueError(
            "VFAI-035 runbook contract failed: "
            f"missingSections={missing_sections}, missingCommands={missing_commands}, "
            f"missingClaims={missing_claims}, missingSources={missing_sources}, "
            f"forbiddenClaims={forbidden_claims}"
        )

    return {
        "ok": True,
        "contractId": contract["contractId"],
        "runbookSha256": _sha256(runbook_path),
        "contractSha256": _sha256(CONTRACT_PATH),
        "requiredSectionCount": len(sections),
        "requiredCommandCount": len(commands),
        "requiredSourceFileCount": len(sources),
        "forbiddenClaimCount": 0,
        "networkAccessed": False,
        "rawContentStored": False,
    }


def main() -> int:
    print(json.dumps(verify(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
