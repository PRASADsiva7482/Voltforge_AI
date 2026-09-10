"""Verify the owned Gen2 design contract; never grant model release approval."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

AI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI))
from model.identity import parse_artifact_id


FILES = ("foundation/contract.v1.json", "foundation/release-policy.v1.json", "foundation/compatibility-map.v1.json")
MODULES = {"data", "tokenizer", "model", "trainer", "inference", "context", "tools", "application", "evaluation-release"}
KINDS = ("base-model", "instruction-model", "integrated-assistant")
PARENTS = ("random-initialization", "accepted-base-model", "accepted-instruction-model")
PROTECTED = {
    "gen1_decision/policy.v1.json", "gen1_training/context-revision-policy.v1.json",
    "gen1_training/tokenizer-v1.1-revision-policy.v1.json", "gen1_sweep/joint-tokenizer-model-policy.v1.json",
    "gen1_optimization/accelerator-policy.v1.json", "corpus_expansion/policy.v1.json",
    "model/release-policy.v1.json", "model/registry/active_model.json", "model/identity.py",
    "api_contract/policy.v1.json", "AI_FOLLOWUP_BACKLOG.json", "synthetic_data/retention-policy.v1.json",
}
RETAINED = {"No imported pretrained weights or vocabulary", "No hosted generation or private-data export",
            "Source licenses and permitted uses remain separately governed", "No automatic live-chat training",
            "Exact compiler/hardware and project-revision evidence", "Checkpoint safety and exact runtime compatibility",
            "Immutable signed artifact/data/tokenizer history", "Finite compute/storage budget before allocation",
            "Independent backup/restore and release review"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate contract key: " + key)
        result[key] = value
    return result


def read(path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def python_routes(path):
    result = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if (isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute)
                    and isinstance(decorator.func.value, ast.Name) and decorator.func.value.id == "router"
                    and decorator.args and isinstance(decorator.args[0], ast.Constant)):
                result.add((decorator.func.attr.upper(), decorator.args[0].value))
    return result


def verify(contract=None, policy=None, mapping=None, *, preserve_baseline=False):
    c, p, m = [read(AI / path) if value is None else value for path, value in zip(FILES, (contract, policy, mapping))]
    require(c["schemaVersion"] == p["schemaVersion"] == m["schemaVersion"] == 1, "Schema version drift")
    require(c["version"] == p["version"] == m["version"] == "1.0.0", "Contract version drift")
    require(c["contractId"] == "voltforge-owned-foundation-v1" and c["status"] == "frozen-design-not-runtime", "Design identity drift")
    d = c["decisions"]
    require(d["weightOrigin"] == "project-owned-random-initialization", "Random initialization required")
    for field in ("importedPretrainedWeightsAllowed", "thirdPartyTokenizerAllowed", "hostedGenerationAllowed", "workstationGpuIsArchitectureLimit", "liveChatTrainingAllowed"):
        require(d[field] is False, "Forbidden foundation decision: " + field)
    require(d["sourceRightsSeparateFromWeightOwnership"] is True and d["servingBoundary"] == "private-owner-controlled-server", "Data/serving boundary drift")
    identity = c["identity"]
    require(identity["familyNamespace"] == p["familyNamespace"] == "vfdlm-g2" and identity["generation"] == 2, "Gen2 family required")
    require(identity["architectureId"] == "vfdlm-gen2-decoder-v1" and identity["tokenizerNamespace"] == "vfdlm-g2-byte-bpe", "Separate architecture/tokenizer identity required")
    require(identity["relabelGen1Allowed"] is False and identity["newLoaderImplemented"] is False and identity["examplesAreReservedNotExistingArtifacts"] is True, "Design cannot relabel history or claim a runtime")
    require(identity["baseExample"] != identity["instructionExample"], "Stage identities must differ")
    for field in ("baseExample", "instructionExample"):
        parsed = parse_artifact_id(identity[field])
        require(parsed.generation == 2 and parsed.deployment_profile == "server", "Canonical Gen2 identity mismatch")
    require(identity["assistantBundleExample"].startswith("vf-assistant-g2-v"), "Assistant bundle is not a model ID")
    scope = c["scope"]
    require(scope["primaryLanguage"] == "en" and scope["otherLanguagesRequireDataAndEvaluation"] is True, "Language scope drift")
    require({"greeting", "clarification", "unsupported-topic-handling", "embedded-c-cpp-firmware", "electronics-explanation"} <= set(scope["supportedIntents"]), "Incomplete assistant scope")
    a = c["architecture"]
    require((a["kind"], a["normalization"], a["positions"], a["feedForward"], a["attention"], a["embeddings"]) == ("causal-decoder-only-transformer", "pre-norm-rmsnorm", "rotary", "swiglu", "grouped-query-causal", "tied-input-output"), "Frozen decoder choice drift")
    require(a["sequenceBudget"] == {"initialTotalTokens": 4096, "extensionTotalTokens": 8192, "inputPlusOutputMustFit": True, "extensionRequiresTrainingAndEvaluation": True}, "Context budget/training drift")
    require(a["allocationRequiresFiniteRunBudget"] is True and a["defaultGen1AllocationCeilingInherited"] is False and a["scalesAreHypotheses"] is True, "Allocation/scale boundary drift")
    lineage = c["lineage"]
    require(lineage["instructionParent"] == "accepted-owned-g2-base-checkpoint" and lineage["assistantParent"] == "accepted-owned-g2-instruction-checkpoint", "Owned stage parent required")
    require(lineage["gen1OrExternalWeightParentAllowed"] is False and lineage["silentTokenizerOrContextChangeAllowed"] is False, "Lineage bypass forbidden")
    require({"contract-sha256", "initialization-receipt-sha256", "checkpoint-sha256", "immutable-data-release-sha256", "split-manifest-sha256", "tokenizer-manifest-sha256", "evaluation-policy-sha256", "evaluation-suite-sha256", "evaluation-report-sha256"} <= set(lineage["commonBindings"]), "Missing lineage binding")
    require({"method=random", "imported-checkpoint=null", "initial-tensor-digest"} <= set(lineage["initializationReceipt"]), "Initialization receipt incomplete")
    require({"optimizer-and-scheduler-state", "rng-per-rank-and-dataloader-cursor", "precision-scaler-and-world-size"} <= set(lineage["resumeReceipt"]), "Resume receipt incomplete")
    backlog = read(AI.parent / "VOLTFORGE_LOCAL_LLM_BUILD_BACKLOG.json")
    tasks = {row["id"] for row in backlog["tasks"]}
    require(len(c["modules"]) == len(MODULES) and {row["id"] for row in c["modules"]} == MODULES, "Incomplete/duplicate module boundary")
    for row in c["modules"]:
        require(all(row.get(key) for key in ("input", "output", "target", "forbidden", "ownerTasks")), "Incomplete module interface")
        require(set(row["ownerTasks"]) <= tasks, "Unknown implementation task")
        require(all((AI / path).exists() for path in row["reuse"]), "Missing reuse source")
    api = c["api"]
    current_api = read(AI / "api_contract/policy.v1.json")
    require(api["pythonPrefix"] == current_api["apiBasePath"] == "/voltForge-ai/api/v1/model" and api["springPrefix"] == "/api/v1/ai", "API route prefix drift")
    require(api["schemaVersion"] == 1 and api["contractVersion"] == current_api["contractVersion"] == "1.0.0", "API version drift")
    require(api["legacyModelRoutingLabel"] == "voltforge-local-engine-v1" and api["identityField"] == "artifact.artifactId", "Public model identity envelope drift")
    require(set(api["revisionFields"]) == {"projectRevision", "sourceProjectRevision"}, "Revision binding required")
    require(api["terminalEvents"] == current_api["events"]["terminalEvents"] and api["generationModes"] == ["neural-quality-gated", "deterministic-fallback", "unavailable"], "SSE semantics drift")
    required_routes = {tuple(row) for row in api["pythonRoutes"]}
    require({("POST", "/chat"), ("POST", "/chat/stream"), ("DELETE", "/chat/requests/{request_id}"), ("GET", "/memory")} <= required_routes, "Required compatibility route omitted")
    require(required_routes <= python_routes(AI / "api/routes.py"), "Python route declaration drift")
    spring = (AI.parent / "Voltforge_BL/src/main/java/in/voltforge/api/ai/controller/AiController.java").read_text(encoding="utf-8")
    require(f'@RequestMapping("{api["springPrefix"]}")' in spring, "Spring prefix drift")
    for method, route in api["springRoutes"]:
        require(re.search(r"@" + method.title() + r'Mapping\(\s*(?:value\s*=\s*)?"' + re.escape(route) + '"', spring) is not None, "Spring route drift")
    acceptance = c["acceptance"]
    require(acceptance["policyId"] == p["policyId"] == "voltforge-owned-release-policy-v1", "Release policy drift")
    require(all(acceptance[key] is False for key in ("trainingAllowedByThisContract", "activationAllowedByThisContract", "designValidationIsReleaseEvidence")), "Design cannot approve training or activation")
    require(acceptance["thresholdFreezeTask"] == p["thresholdContract"]["ownerTask"] == "LLM-TASK-018" and acceptance["thresholdState"] == "pending-independent-suite", "Threshold ownership/state drift")
    require(p["thresholdContract"]["currentlyAvailable"] is False and p["thresholdContract"]["missingOrUnfrozenDecision"] == acceptance["missingEvidenceDecision"] == "blocked", "Missing acceptance evidence must block release")
    require(len(p["stages"]) == 3, "Three separate release stages required")
    for stage, kind, parent in zip(p["stages"], KINDS, PARENTS):
        require(stage["kind"] == kind and stage["parentKind"] == parent, "Invalid release parent/stage")
        require(stage["chatEligible"] is (kind == "integrated-assistant"), "Only integrated assistant can be chat eligible")
        require(len(stage["requiredEvidence"]) >= 6 and set(stage["ownerTasks"]) <= tasks, "Incomplete stage evidence/tasks")
    require("actual-weights-logit-and-token-invocation" in p["stages"][0]["requiredEvidence"] and "raw-neural-no-rule-substitution" in p["stages"][1]["requiredEvidence"], "Neural invocation proof required")
    require({"browser-review-apply-undo", "backup-restore-and-rollback", "independent-final-release-review"} <= set(p["stages"][2]["requiredEvidence"]), "Product acceptance incomplete")
    life = p["lifecycle"]
    require(life["activationRequiresAcceptedAssistant"] is True and life["expectedRegistryRevisionRequired"] is True and life["automaticActivationAllowed"] is False and life["historyRewriteAllowed"] is False, "Activation/history bypass forbidden")
    require(life["transitions"] == {"experimental": ["candidate", "rejected"], "candidate": ["accepted", "rejected"], "accepted": [], "rejected": []}, "Release transition drift")
    old = {row["id"]: row for row in read(AI / m["followupBacklogPath"])["items"]}
    require(len(m["followups"]) == len(old) and {row["id"] for row in m["followups"]} == set(old), "Incomplete/duplicate follow-up mapping")
    for row in m["followups"]:
        require(row["observedStatus"] in {"done", "promoted", "accepted_for_later"} and row["statusChangedByThisMapping"] is False, "Invalid historical follow-up status")
        if preserve_baseline:
            require(row["observedStatus"] == old[row["id"]]["status"], "Historical follow-up status drift")
        require(row["tasks"] and set(row["tasks"]) <= tasks and row["decision"], "Invalid follow-up task mapping")
    require(RETAINED <= set(m["retainedBoundaries"]), "Retained governance boundary missing")
    require(len(m["protectedFiles"]) == len(PROTECTED) and {row["path"] for row in m["protectedFiles"]} == PROTECTED, "Incomplete historical preservation inventory")
    for row in m["protectedFiles"]:
        require(re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is not None, "Invalid historical digest")
        if preserve_baseline:
            require(digest(AI / row["path"]) == row["sha256"], "Historical file changed: " + row["path"])
    require(len(m["policyOverrides"]) == 7 and len({row["path"] for row in m["policyOverrides"]}) == 7, "Incomplete/duplicate policy mapping")
    for row in m["policyOverrides"]:
        require(row["path"] in PROTECTED and (AI / row["replacementPath"]).is_file() and set(row["tasks"]) <= tasks, "Invalid policy replacement")
        require(row["appliesTo"] == "vfdlm-g2-only" and row["changesHistoricalPolicy"] is False and row["implementationRequired"] is True, "Policy replacement cannot mutate Gen1 or claim implementation")
    design = (AI / c["designPath"]).read_text(encoding="utf-8")
    for number in range(1, 11):
        require(f"## {number}. " in design, "Missing design section")
    for row in m["followups"]:
        require(f'| {row["id"]} | {row["observedStatus"]} | {", ".join(row["tasks"])} | {row["decision"]} |' in design, "Readable follow-up mapping drift")
    for row in m["policyOverrides"]:
        require(row["path"] in design and row["newRule"] in design, "Readable policy mapping drift")
    if preserve_baseline:
        require(read(AI / "model/registry/active_model.json")["activeArtifactId"] is None, "Task 017 must not activate a model")
    return {"status": "passed", "contractId": c["contractId"], "familyNamespace": identity["familyNamespace"],
            "releaseStages": list(KINDS), "moduleCount": len(MODULES), "mappedFollowups": len(old),
            "historicalBindings": len(PROTECTED), "preservationChecked": preserve_baseline,
            "preservedHistoricalFiles": len(PROTECTED) if preserve_baseline else 0,
            "compatiblePythonRoutes": len(required_routes),
            "scope": "design consistency only; no decoder, training, runtime or release acceptance",
            "modelReleaseApproved": False, "networkAccessed": False, "nativeModelImported": "torch" in sys.modules,
            "inputs": [{"path": path, "sha256": digest(AI / path)} for path in (*FILES, c["designPath"], "tools/verify_llm_foundation.py")],
            "thresholds": "pending LLM-TASK-018", "nextTask": "LLM-TASK-018"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--check-preserved-history", action="store_true", help="Also verify task-017 baseline bytes/statuses remained unchanged; not a permanent ban on later governed releases.")
    args = parser.parse_args()
    report = verify(preserve_baseline=args.check_preserved_history)
    if args.report:
        report["generatedAtUtc"] = datetime.now(timezone.utc).isoformat()
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
