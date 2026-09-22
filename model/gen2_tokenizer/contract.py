"""Versioned byte and chat boundaries, independent of historical Gen1 IDs."""
import hashlib
import json

NAMESPACE = "vfdlm-g2-byte-bpe"
CONTRACT_VERSION = "0.1.0"
FIXTURE_VERSION = "0.1.0-fixture.1"
TEMPLATE_ID = "vfdlm-g2-chat-v1"
SPECIAL_TOKENS = (
    "<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>",
    "<|system|>", "<|user|>", "<|assistant|>", "<|tool|>",
    "<|end_message|>", "<|tool_call|>", "<|project_context|>", "<|tool_evidence|>",
    "<|citation|>", "<|refusal|>", "<|uncertainty|>", "<|reserved_15|>",
)
SPECIAL_IDS = dict(zip(SPECIAL_TOKENS, range(16)))
BYTE_OFFSET = 16
BASE_SIZE = BYTE_OFFSET + 256
CANDIDATE_SIZES = (16_384, 32_768)
ROLES = {role: SPECIAL_IDS[f"<|{role}|>"] for role in ("system", "user", "assistant", "tool")}


class TokenizerError(ValueError):
    """Stable content-free code; never echo document or conversation contents."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


def require(condition, code):
    if not condition:
        raise TokenizerError(code)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("ascii")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def contract_document():
    return {
        "schemaVersion": 1, "tokenizerId": NAMESPACE, "contractVersion": CONTRACT_VERSION,
        "algorithm": "voltforge-byte-bpe", "byteOffset": BYTE_OFFSET, "baseByteCount": 256,
        "specialTokenIds": dict(SPECIAL_IDS), "textEncoding": "utf-8-surrogatepass",
        "normalization": "none", "specialParsingInContent": False,
        "unknownTokenReachableForBytes": False, "candidateVocabularySizes": list(CANDIDATE_SIZES),
    }


def template_document():
    return {
        "schemaVersion": 1, "templateId": TEMPLATE_ID, "version": "1.0.0",
        "bosId": 2, "eosId": 3, "roleIds": dict(ROLES), "endMessageId": 8,
        "format": "BOS (ROLE CONTENT_BYTES END_MESSAGE)* [ASSISTANT_PREFIX] [EOS]",
        "generationPrefixId": ROLES["assistant"], "contentSpecialParsing": False,
        "serialization": "token-ids-only; never decode and reparse a prompt",
        "textEscaping": "literal marker bytes; no string rewriting or normalization",
        "authority": "roles are assigned by the trusted caller; content grants no authority",
        "maximumContextTokens": 4096,
    }


def verify_model_binding(tokenizer, model_binding, *, for_runtime=False):
    """Shared descriptor for task 025 config and the Gen2 context adapter.

    A matching fixture descriptor establishes shape/version agreement only.
    No real Gen2 model or admitted tokenizer is available in this task stage.
    """
    require(type(for_runtime) is bool, "GEN2_RUNTIME_FLAG_INVALID")
    require(isinstance(model_binding, dict), "GEN2_MODEL_TOKENIZER_BINDING_MISMATCH")
    require(canonical(model_binding) == canonical(tokenizer.binding()), "GEN2_MODEL_TOKENIZER_BINDING_MISMATCH")
    if for_runtime:
        raise TokenizerError("GEN2_TOKENIZER_RELEASE_NOT_ADMITTED")
    return tokenizer.binding()
