"""Create/verify a recoverable source snapshot for LLM-TASK-016.

Copies only the listed source/docs and Python tests; no .env or signing keys.
Protected model/data bytes are inventoried for no-change verification.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


AI = Path(__file__).resolve().parents[1]
ROOT = AI.parent
SOURCES = [
    "api/chat.py", "model/runtime_service.py", "model/local_engine.py",
    "model/registry_manager.py", "model/gen1/model.py", "model/generation_quality.py",
    "model/reasoning_llm.py", "model/infer.py", "engine/reasoning.py",
    "tools/activate_domain_artifact.py", "tools/audit_llm_foundation.py",
    "tools/verify_llm_backlogs.py", "tools/snapshot_llm_cleanup.py",
    "docs/VOLTForge_AI_RUNBOOK.md", "docs/runbook-contract.v1.json",
    "../VOLTFORGE_LOCAL_LLM_BUILD_BACKLOG.json", "../VOLTFORGE_LOCAL_LLM_BUILD_BACKLOG.md",
    "../AI_CHAT_ROOT_CAUSE_BACKLOG.json", "../AI_CHAT_ROOT_CAUSE_ANALYSIS.md",
]


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def descriptor(path):
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
            "sha256": sha(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        directory = args.verify.resolve()
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        for item in manifest["source_files"]:
            saved = directory / "files" / item["path"]
            if not saved.is_file() or sha(saved) != item["sha256"]:
                raise RuntimeError("Snapshot file mismatch: " + item["path"])
        for item in manifest["protected_files"]:
            if sha(ROOT / item["path"]) != item["sha256"]:
                raise RuntimeError("Protected artifact changed: " + item["path"])
        print(json.dumps({"snapshot": str(directory), "verified_sources": len(manifest["source_files"]),
                          "unchanged_protected_files": len(manifest["protected_files"])}))
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = ROOT / "backlog_history" / ("llm-task-016-before-" + stamp)
    directory.mkdir(exist_ok=False)
    sources = {(AI / path).resolve() for path in SOURCES}
    sources.update((AI / "tests").glob("*.py"))
    protected = []
    for relative in ("model/registry", "model/server/artifacts", "model/tokenizers", "synthetic_data/releases"):
        folder = AI / relative
        if folder.exists():
            protected.extend(path for path in folder.rglob("*") if path.is_file())
    manifest = {"task_id": "LLM-TASK-016", "created_at_utc": stamp,
                "source_files": [], "protected_files": [descriptor(path) for path in sorted(protected)],
                "restore": "Copy individual files from files/<workspace-relative-path> after reviewing later changes. No automatic reset or deletion.",
                "excluded": [".env files", "private signing keys", "raw user conversations"]}
    for source in sorted(sources):
        relative = source.relative_to(ROOT)
        destination = directory / "files" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        item = descriptor(source)
        if sha(destination) != item["sha256"]:
            raise RuntimeError("Snapshot verification failed")
        manifest["source_files"].append(item)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"snapshot": str(directory), "source_files": len(sources), "protected_files": len(protected)}))


if __name__ == "__main__":
    main()
