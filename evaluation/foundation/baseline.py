"""Current rule baseline and actual in-process defect probes; no model activation."""
from __future__ import annotations

from datetime import datetime, timezone
import re

from .harness import evaluate
from .observer import InvocationLedger
from .suite import AI, ROOT, load_split, read, sha, verify_suite


def run_baseline():
    from engine.deterministic_tools import DeterministicElectronicsTools
    from model.runtime_service import ModelRuntimeService
    from tools.audit_llm_foundation import synthetic_probes
    policy = read(ROOT / "acceptance-policy.v1.json")
    summary = verify_suite()
    cases = load_split("acceptance") + load_split("adversarial") + load_split("compile")
    runtime = ModelRuntimeService()
    health = runtime.start()
    runtime.stop()
    if health.get("ready"):
        raise ValueError("This no-model baseline cannot be reused after activation; select a candidate evaluation run")
    tools = DeterministicElectronicsTools()
    def rule_adapter(request):
        result = tools.reason_and_solve(request["message"])
        return {"text": result.get("answer", ""), "generationSource": "deterministic-tools"}
    def unavailable_model(request):
        return {"text": "", "unavailable": health["code"]}
    def unavailable_product(request):
        return {"text": "", "unavailable": "NO_DEPLOYED_GEN2_PRODUCT_CANDIDATE"}
    scorecards = []
    for lane in policy["lanes"]:
        for seed in policy["seeds"]["generation"]:
            adapter = rule_adapter if lane == "rule_only" else unavailable_product if lane == "product" else unavailable_model
            scorecards.append(evaluate(cases, adapter, lane=lane, seed=seed, ledger=InvocationLedger()))
    probes = synthetic_probes()
    led = next(row for row in probes["chat_results"] if row["prompt"] == "What is an LED?")
    repeated = []
    for row in probes["chat_results"]:
        lines = [re.sub(r"\s+", " ", line.strip()) for line in row["reply"].splitlines() if "cannot verify" in line.casefold()]
        if len(lines) > len(set(lines)):
            repeated.append({"prompt": row["prompt"], "uncertaintyLines": len(lines), "uniqueUncertaintyLines": len(set(lines))})
    first = tools.reason_and_solve("Calculate capacitive reactance for 100nF at 1kHz")
    second = tools.reason_and_solve("Calculate capacitive reactance for 100nF at 2kHz")
    a, b = first["calculation"]["reactanceOhm"], second["calculation"]["reactanceOhm"]
    return {"schemaVersion": 1, "reportId": "llm-task-018-independent-baseline-v1", "status": "baseline-recorded-no-model-accepted",
            "generatedAtUtc": datetime.now(timezone.utc).isoformat(), "suite": summary,
            "runtime": {key: health.get(key) for key in ("ready", "code", "activeArtifactId", "runtimeOperational")},
            "scorecards": scorecards,
            "observedNeuralForwardCalls": sum(row["observedForwardCalls"] for row in scorecards),
            "observedModelTokens": sum(row["observedGeneratedTokens"] for row in scorecards),
            "regressions": {"ledOledConfusion": {"detected": "oled" in led["reply"].casefold() or "ssd1306" in led["reply"].casefold(), "replySha256": led["reply_sha256"]},
                            "repeatedUncertainty": repeated,
                            "changedValues": {"oneKHzOhm": a, "twoKHzOhm": b, "inverseFrequencyRelationshipPassed": abs(a - 2 * b) < 1e-9, "source": "deterministic-tools"},
                            "noModel": {"ready": health["ready"], "code": health["code"], "neuralCredit": False}},
            "inProcessSyntheticProbes": probes,
            "sourceFingerprints": [{"path": path, "sha256": sha((AI / path).read_bytes())} for path in
                                   ("api/chat.py", "engine/reasoning.py", "engine/deterministic_tools.py", "model/runtime_service.py", "tools/audit_llm_foundation.py")],
            "limits": ["No model quality measurement: all neural/product lanes are unavailable and stay in the denominator.",
                       "Rule-only output receives no neural credit; absent signed expert reviews are pending, never passes.",
                       "Compilation cases are designed against exact tuples; no generated candidate source exists to compile in this baseline.",
                       "Acceptance is checksum locked, not secret/encrypted; separate custody and independent audits remain release requirements.",
                       "Application probes are in-process only; no deployed HTTP gateway, browser or performance acceptance."]}
