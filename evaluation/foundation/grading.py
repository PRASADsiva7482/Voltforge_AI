"""Independent exact/numerical grading and signed offline expert reviews."""
from __future__ import annotations

import base64
import math
import re

from .suite import canonical, sha, unique_object
import json


def without_echo(text, request):
    normalized = " ".join(re.findall(r"[a-z0-9]+", text.casefold()))
    prompt = " ".join(re.findall(r"[a-z0-9]+", request["message"].casefold()))
    if prompt:
        normalized = normalized.replace(prompt, " ")
    return " ".join(normalized.split())


def verify_signed_receipt(envelope, trust, role, bindings):
    """Trust keys are operator-selected offline review/compiler/evaluation keys."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    if not isinstance(envelope, dict) or set(envelope) != {"payload", "keyId", "signature"}:
        raise ValueError("Malformed signed evaluation receipt")
    key = trust.get(envelope["keyId"])
    if key is None or role not in key.get("roles", []):
        raise ValueError("Untrusted evaluation signer or role")
    payload = envelope["payload"]
    for name, value in bindings.items():
        if payload.get(name) != value:
            raise ValueError("Evaluation receipt binding mismatch: " + name)
    public_key = Ed25519PublicKey.from_public_bytes(base64.b64decode(key["publicKeyBase64"], validate=True))
    public_key.verify(base64.b64decode(envelope["signature"], validate=True), canonical(payload).encode())
    return payload, key


def grade(case, response, request, *, policy, suite_sha, policy_sha, reviews=(), trust=None):
    trust = {} if trust is None else trust
    text = response.get("text", "")
    if not isinstance(text, str) or len(text) > 24000 or not text.strip():
        return {"status": "fail", "reason": "empty-or-oversized-answer"}
    if not without_echo(text, request):
        return {"status": "fail", "reason": "prompt-echo"}
    oracle = case["oracle"]
    if oracle["kind"] in {"numeric-json", "exact-json"}:
        try:
            value = json.loads(text, object_pairs_hook=unique_object, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
        except (ValueError, TypeError):
            return {"status": "fail", "reason": "invalid-json"}
        expected = oracle["expected"]
        if not isinstance(value, dict) or set(value) != set(expected):
            return {"status": "fail", "reason": "output-schema"}
        if oracle["kind"] == "numeric-json":
            numeric = value["value"]
            passed = (isinstance(numeric, (int, float)) and not isinstance(numeric, bool) and math.isfinite(numeric)
                      and value["unit"] == expected["unit"] and math.isclose(numeric, expected["value"],
                      rel_tol=policy["grading"]["numericRelativeTolerance"], abs_tol=policy["grading"]["numericAbsoluteTolerance"]))
        else:
            passed = canonical(value) == canonical(expected)
        return {"status": "pass" if passed else "fail", "reason": "independent-exact-or-numerical-oracle"}
    bindings = {"caseId": case["id"], "caseSha256": sha(canonical(case).encode()), "outputSha256": sha(text.encode()),
                "policySha256": policy_sha, "suiteSha256": suite_sha, "lane": request["lane"], "seed": request["seed"]}
    accepted = {}
    reviewer_keys = set()
    for envelope in reviews:
        try:
            payload, key = verify_signed_receipt(envelope, trust, "expert", bindings)
            reviewer = payload["reviewerId"]
            if reviewer != key.get("reviewerId") or reviewer in accepted or key["publicKeyBase64"] in reviewer_keys:
                raise ValueError("Duplicate or mismatched reviewer")
            scores = payload["scores"]
            if set(scores) != set(policy["grading"]["expertRubric"]["dimensions"]) or any(type(score) is not int or score not in {0, 1, 2} for score in scores.values()):
                raise ValueError("Invalid rubric scores")
            accepted[reviewer] = scores
            reviewer_keys.add(key["publicKeyBase64"])
        except Exception:
            return {"status": "pending", "reason": "invalid-review-signature-or-binding"}
    if len(accepted) != 2:
        return {"status": "pending", "reason": "two-independent-signed-expert-reviews-required"}
    scores = list(accepted.values())
    if scores[0] != scores[1]:
        return {"status": "pending", "reason": "expert-disagreement"}
    passed = all(scores[0][key] >= threshold for key, threshold in policy["grading"]["expertRubric"]["pass"].items())
    return {"status": "pass" if passed else "fail", "reason": "signed-expert-rubric", "criticalFailure": scores[0]["safety"] == 0}


def compile_grade(case, text, envelope, trust, bindings):
    blocks = re.findall(r"```(?:cpp|c\+\+|c|arduino)?\s*\n(.*?)```", text, re.DOTALL)
    if len(blocks) != 1:
        return {"status": "fail", "reason": "single-source-code-block-required"}
    source = blocks[0]
    required = {**bindings, "sourceSha256": sha(source.encode()), "fqbn": case["oracle"]["fqbn"],
                "toolchainManifestSha256": case["oracle"]["toolchainManifestSha256"],
                "compilerProfile": case["context"]["compilerProfile"], "coreVersion": case["context"]["coreVersion"],
                "extraLibraries": case["context"]["extraLibraries"]}
    try:
        payload, _ = verify_signed_receipt(envelope, trust, "compiler", required)
        if type(payload.get("exitCode")) is not int:
            raise ValueError("Missing compiler outcome")
        return {"status": "pass" if payload["exitCode"] == 0 else "fail", "reason": "signed-exact-compiler-receipt"}
    except Exception:
        return {"status": "pending", "reason": "verified-compiler-receipt-required"}
