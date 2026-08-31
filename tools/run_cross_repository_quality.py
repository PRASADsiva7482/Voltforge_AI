"""Run the immutable cross-repository VoltForge quality manifest."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable


AI_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = AI_ROOT.parent
MANIFEST_PATH = AI_ROOT / "quality_pipeline.v1.json"
DEFAULT_REPORT_PATH = AI_ROOT / "evaluation" / "reports" / "cross-repository-quality-v1.json"
LANES = {"fast", "model", "all"}
PLACEHOLDERS = {"__AI_PYTHON__", "__MAVEN__", "__NPM__", "__GIT__"}


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    lane: str
    category: str
    status: str
    exit_code: int | None
    duration_ms: int
    command: tuple[str, ...]


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_manifest() -> dict[str, Any]:
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("QUALITY_MANIFEST_INVALID") from error
    if not isinstance(manifest, dict) or manifest.get("schemaVersion") != 1 or not isinstance(manifest.get("checks"), list):
        raise RuntimeError("QUALITY_MANIFEST_INVALID")
    seen: set[str] = set()
    for check in manifest["checks"]:
        if not isinstance(check, dict) or not isinstance(check.get("id"), str) or check["id"] in seen:
            raise RuntimeError("QUALITY_MANIFEST_INVALID")
        seen.add(check["id"])
        if check.get("lane") not in {"fast", "model"} or not isinstance(check.get("command"), list) or not check["command"]:
            raise RuntimeError("QUALITY_MANIFEST_INVALID")
        if any(not isinstance(token, str) or (token.startswith("__") and token not in PLACEHOLDERS) for token in check["command"]):
            raise RuntimeError("QUALITY_MANIFEST_INVALID")
        if not isinstance(check.get("cwd"), str) or not isinstance(check.get("category"), str):
            raise RuntimeError("QUALITY_MANIFEST_INVALID")
    if set(manifest.get("toolPlaceholders", [])) != PLACEHOLDERS:
        raise RuntimeError("QUALITY_MANIFEST_INVALID")
    return manifest


def _resolve_command_tools() -> dict[str, str]:
    python_path = AI_ROOT / ".toolchains" / "gen1" / "Scripts" / "python.exe"
    if not python_path.is_file():
        raise RuntimeError("PINNED_AI_PYTHON_MISSING")

    def find(*names: str, candidates: Iterable[Path] = ()) -> str:
        for name in names:
            found = shutil.which(name)
            if found:
                return found
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        raise RuntimeError(f"REQUIRED_TOOL_MISSING:{names[0]}")

    maven_candidates: list[Path] = []
    for variable in ("MAVEN_HOME", "M2_HOME"):
        value = os.environ.get(variable)
        if value:
            maven_candidates.append(Path(value) / "bin" / "mvn.cmd")
            maven_candidates.append(Path(value) / "bin" / "mvn")
    for repository_root in (WORKSPACE_ROOT / "Voltforge_BL", WORKSPACE_ROOT / "Voltforge_UI"):
        maven_candidates.extend((repository_root / name for name in ("mvnw.cmd", "mvnw")))
    jetbrains_root = Path("C:/Program Files/JetBrains")
    if jetbrains_root.is_dir():
        maven_candidates.extend(jetbrains_root.glob("*/plugins/maven/lib/maven3/bin/mvn.cmd"))

    return {
        "__AI_PYTHON__": str(python_path),
        "__MAVEN__": find("mvn.cmd", "mvn", candidates=maven_candidates),
        "__NPM__": find("npm.cmd", "npm"),
        "__GIT__": find("git.exe", "git"),
    }


def _version(executable: str, args: tuple[str, ...]) -> str:
    try:
        completed = subprocess.run([executable, *args], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    output = (completed.stdout or completed.stderr or "").splitlines()
    return output[0].strip()[:160] if output else "unavailable"


def _is_excluded(relative: str, exclusions: Iterable[str]) -> bool:
    normalized = relative.replace("\\", "/").strip("/")
    for raw_prefix in exclusions:
        prefix = str(raw_prefix).replace("\\", "/").strip("/")
        if normalized == prefix or normalized.startswith(prefix + "/"):
            return True
    return False


def _tree_fingerprint(root: Path, exclusions: Iterable[str]) -> dict[str, Any]:
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if _is_excluded(relative, exclusions):
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(data)).encode("ascii"))
        digest.update(b"\0")
        digest.update(data)
        file_count += 1
        total_bytes += len(data)
    return {"sha256": digest.hexdigest(), "fileCount": file_count, "bytes": total_bytes}


def _git_state(executable: str, root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run([executable, *args], cwd=root, capture_output=True, text=True, timeout=30, check=False)
        return (completed.stdout or "").strip()

    return {"head": run("rev-parse", "HEAD") or "unavailable", "clean": not bool(run("status", "--porcelain"))}


def _expand_command(command: list[str], tools: dict[str, str]) -> tuple[str, ...]:
    return tuple(tools.get(token, token) for token in command)


def _selected_checks(manifest: dict[str, Any], lane: str) -> list[dict[str, Any]]:
    lanes = {"fast"} if lane == "fast" else {"model"} if lane == "model" else {"fast", "model"}
    return [check for check in manifest["checks"] if check["lane"] in lanes]


def _run_check(check: dict[str, Any], tools: dict[str, str], verbose: bool) -> CheckResult:
    command = _expand_command(check["command"], tools)
    cwd = WORKSPACE_ROOT / check["cwd"]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            capture_output=not verbose,
            text=True,
            timeout=int(check.get("timeoutSeconds", 600)),
            check=False,
            env=os.environ.copy(),
        )
        status = "pass" if completed.returncode == 0 else "fail"
        exit_code: int | None = completed.returncode
    except subprocess.TimeoutExpired:
        status = "timeout"
        exit_code = None
    except (OSError, subprocess.SubprocessError):
        status = "error"
        exit_code = None
    duration_ms = round((time.perf_counter() - started) * 1000)
    return CheckResult(check["id"], check["lane"], check["category"], status, exit_code, duration_ms, command)


def _clean_worktrees(manifest: dict[str, Any], git: str) -> list[dict[str, Any]]:
    results = []
    for relative in manifest["releaseRequirements"]["cleanWorktrees"]:
        root = WORKSPACE_ROOT / relative
        state = _git_state(git, root)
        results.append({"repository": relative, "clean": state["clean"], "head": state["head"]})
    return results


def run_pipeline(*, lane: str, release: bool = False, report_path: Path = DEFAULT_REPORT_PATH, verbose: bool = False) -> tuple[int, dict[str, Any]]:
    manifest = _load_manifest()
    if lane not in LANES:
        raise ValueError(f"Unsupported lane: {lane}")
    tools = _resolve_command_tools()
    if release and lane not in set(manifest["releaseRequirements"]["allowedLanes"]):
        raise RuntimeError("RELEASE_REQUIRES_MODEL_LANE")

    exclusions = manifest["sourceFingerprintExclusions"]
    inputs = {
        "manifestSha256": _sha256(MANIFEST_PATH.read_bytes()),
        "repositories": {
            name: _tree_fingerprint(WORKSPACE_ROOT / relative, exclusions)
            for name, relative in manifest["sourceRoots"].items()
        },
    }
    git_states = {name: _git_state(tools["__GIT__"], WORKSPACE_ROOT / relative) for name, relative in manifest["sourceRoots"].items()}
    clean_worktrees = _clean_worktrees(manifest, tools["__GIT__"])
    selected = _selected_checks(manifest, lane)
    results: list[CheckResult] = []
    if release and any(not item["clean"] for item in clean_worktrees):
        results.append(CheckResult("release-clean-worktrees", lane, "release-preflight", "fail", 1, 0, tuple()))
    else:
        for check in selected:
            result = _run_check(check, tools, verbose)
            results.append(result)
            print(f"[{result.status.upper()}] {result.check_id} ({result.duration_ms} ms)")

    deterministic_checks = [
        {"id": result.check_id, "lane": result.lane, "category": result.category, "status": result.status, "exitCode": result.exit_code}
        for result in results
    ]
    tool_versions = {
        "python": _version(tools["__AI_PYTHON__"], ("--version",)),
        "maven": _version(tools["__MAVEN__"], ("--version",)),
        "npm": _version(tools["__NPM__"], ("--version",)),
        "git": _version(tools["__GIT__"], ("--version",)),
    }
    scorecard = {
        "schemaVersion": 1,
        "pipelineId": manifest["pipelineId"],
        "lane": lane,
        "release": release,
        "manifestSha256": inputs["manifestSha256"],
        "inputs": inputs["repositories"],
        "gitHeads": {name: state["head"] for name, state in git_states.items()},
        "toolVersions": tool_versions,
        "checks": deterministic_checks,
        "summary": {
            "passed": sum(result.status == "pass" for result in results),
            "failed": sum(result.status in {"fail", "timeout", "error"} for result in results),
            "skipped": 0,
        },
    }
    scorecard_sha = _sha256(_canonical(scorecard))
    report = {
        "schemaVersion": 1,
        "reportId": "vfai032-cross-repository-quality-v1",
        "generatedAtUtc": _utc_now(),
        "lane": lane,
        "release": release,
        "manifestSha256": inputs["manifestSha256"],
        "inputs": inputs,
        "git": git_states,
        "releaseWorktrees": clean_worktrees,
        "toolVersions": tool_versions,
        "checks": [
            {
                "id": result.check_id,
                "lane": result.lane,
                "category": result.category,
                "status": result.status,
                "exitCode": result.exit_code,
                "durationMs": result.duration_ms,
            }
            for result in results
        ],
        "scorecard": {**scorecard, "scorecardSha256": scorecard_sha},
        "privacy": {
            "rawStdoutStored": False,
            "rawStderrStored": False,
            "rawPromptStored": False,
            "rawProjectContentStored": False,
        },
    }
    report["reportSha256"] = _sha256(_canonical(report))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    failed = report["scorecard"]["summary"]["failed"]
    print(f"Quality lane {lane}: {report['scorecard']['summary']['passed']} passed, {failed} failed")
    print(f"Scorecard SHA-256: {scorecard_sha}")
    print(f"Report: {report_path}")
    return (1 if failed else 0), report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=sorted(LANES), default="fast")
    parser.add_argument("--release", action="store_true", help="Require model/all lane and clean repository worktrees.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--verbose", action="store_true", help="Forward child output; disabled by default to avoid retaining project content.")
    args = parser.parse_args(argv)
    try:
        status, _ = run_pipeline(lane=args.lane, release=args.release, report_path=args.report, verbose=args.verbose)
    except (RuntimeError, ValueError) as error:
        print(json.dumps({"ok": False, "code": str(error)}))
        return 2
    return status


if __name__ == "__main__":
    raise SystemExit(main())
