"""Validate foundation backlog IDs, dependencies, counts, evidence and Markdown.

Run from any directory. This command is read-only and uses only the standard library.
It validates planning consistency, not neural implementation or release acceptance.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=unique_object)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    backlog = read_json(ROOT / "VOLTFORGE_LOCAL_LLM_BUILD_BACKLOG.json")
    causes = read_json(ROOT / "AI_CHAT_ROOT_CAUSE_BACKLOG.json")
    markdown = (ROOT / "VOLTFORGE_LOCAL_LLM_BUILD_BACKLOG.md").read_text(encoding="utf-8")
    tasks = {task["id"]: task for task in backlog["tasks"]}
    legacy = {task["id"]: task for task in backlog["legacy_tasks"]}
    require(len(tasks) == len(backlog["tasks"]) == backlog["total_tasks"], "Task IDs/count mismatch")
    require(len(legacy) == len(backlog["legacy_tasks"]) == backlog["total_legacy_tasks"], "Legacy IDs/count mismatch")
    require(not set(tasks).intersection(legacy), "Active and legacy IDs overlap")
    require(all(re.fullmatch(r"LLM-TASK-\d{3}", item) for item in (*tasks, *legacy)), "Invalid task ID")
    counts = Counter(task["status"] for task in tasks.values())
    for status in set(counts) | set(backlog["status_summary"]):
        require(counts[status] == backlog["status_summary"].get(status, 0), f"Status count mismatch: {status}")
    require(dict(Counter(task["status"] for task in legacy.values())) == backlog["legacy_status_summary"], "Legacy status mismatch")

    seen, visiting = set(), set()

    def visit(task_id):
        require(task_id in tasks, f"Missing dependency {task_id}")
        require(task_id not in visiting, f"Dependency cycle at {task_id}")
        if task_id in seen:
            return
        visiting.add(task_id)
        for dependency in tasks[task_id]["depends_on"]:
            visit(dependency)
        visiting.remove(task_id)
        seen.add(task_id)

    for task_id, task in tasks.items():
        visit(task_id)
        require(task["work"] and task["definition_of_done"] and task["required_evidence"], f"Incomplete task {task_id}")
        heading = f"#### {task_id} — {task['title']}"
        require(markdown.count(heading) == 1, f"Missing/duplicate Markdown task {task_id}")
        section = markdown.split(heading, 1)[1].split("\n#### ", 1)[0].split("\n### ", 1)[0]
        require(f"**Status:** {task['status']}." in section, f"Markdown status mismatch {task_id}")
        require(f"**Depends on:** {', '.join(task['depends_on']) or 'none'}." in section, f"Markdown dependencies mismatch {task_id}")
        for text in task["work"] + task["definition_of_done"] + task["required_evidence"]:
            require(text in section, f"Markdown detail drift {task_id}")
        if task["status"] == "COMPLETED":
            require(all(tasks[dep]["status"] == "COMPLETED" for dep in task["depends_on"]), f"Incomplete dependency for {task_id}")
            require(bool(task.get("evidence_records")), f"No evidence for completed task {task_id}")
            for path in task["evidence_records"]:
                require((ROOT / path).is_file(), f"Missing completed-task evidence {path}")

    order = backlog["execution"]["recommended_order"]
    require(len(order) == len(set(order)) and set(order) == set(tasks), "Execution order does not cover each task once")
    positions = {task_id: index for index, task_id in enumerate(order)}
    for task in tasks.values():
        require(all(positions[dep] < positions[task["id"]] for dep in task["depends_on"]), f"Execution order violates dependency for {task['id']}")
    next_id = backlog["execution"]["next_task_id"]
    if next_id is None:
        require(all(task["status"] == "COMPLETED" for task in tasks.values()), "Missing next task with incomplete work")
    else:
        require(next_id in tasks and tasks[next_id]["status"] != "COMPLETED", "Invalid next task")
        require(all(tasks[dep]["status"] == "COMPLETED" for dep in tasks[next_id]["depends_on"]), "Next task is not ready")

    milestone_ids = {item["id"] for item in backlog["milestones"]}
    require(len(milestone_ids) == len(backlog["milestones"]) == backlog["total_milestones"], "Milestone count mismatch")
    assigned = []
    for milestone in backlog["milestones"]:
        assigned.extend(milestone["task_ids"])
        require(all(tasks[task_id]["milestone"] == milestone["id"] for task_id in milestone["task_ids"]), "Milestone assignment mismatch")
    require(len(assigned) == len(tasks) and set(assigned) == set(tasks), "Missing/duplicate milestone task")
    for item in legacy.values():
        require(item["status"] == "SUPERSEDED", "Legacy completion claim restored")
        require(all(ref in tasks for ref in item["replacement_tasks"]), "Missing legacy replacement")

    records = causes["root_causes"]
    require(len({item["id"] for item in records}) == len(records), "Duplicate root cause ID")
    require(causes["summary"]["total_root_causes_identified"] == len(records), "Root cause count mismatch")
    for status, field in (("OPEN", "open"), ("PARTIAL", "partial"), ("RESOLVED", "resolved")):
        require(sum(item["status"] == status for item in records) == causes["summary"][f"total_root_causes_{field}"], "Root cause status mismatch")
    for item in records:
        require(item["evidence"] and item["remediation_task_ids"], "Root cause missing evidence/remediation")
        require(all(ref in tasks for ref in item["remediation_task_ids"]), "Missing remediation task")
        if item["status"] == "RESOLVED":
            require(all(tasks[ref]["status"] == "COMPLETED" for ref in item["remediation_task_ids"]), "Root cause resolved before remediation completion")
    for item in causes["legacy_remediation_actions"]:
        require(item["status"] == "SUPERSEDED", "Legacy remediation completion claim restored")
        require(all(ref in tasks for ref in item["replacement_task_ids"]), "Missing legacy action replacement")
    for path in causes["evidence_files"]:
        require((ROOT / path).is_file(), f"Missing root cause evidence {path}")

    for name in ("VOLTFORGE_LOCAL_LLM_BUILD_BACKLOG.md", "AI_CHAT_ROOT_CAUSE_ANALYSIS.md"):
        content = (ROOT / name).read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)]+)\)", content):
            if target.startswith(("https://", "http://", "#")):
                continue
            require((ROOT / target.split("#", 1)[0]).exists(), f"Broken local link in {name}: {target}")
    print(json.dumps({"status": "passed", "active_tasks": len(tasks), "legacy_tasks": len(legacy),
                      "milestones": len(milestone_ids), "root_causes": len(records),
                      "status_counts": counts, "next_task": next_id,
                      "scope": "planning consistency only; no model release credit"}))


if __name__ == "__main__":
    main()
