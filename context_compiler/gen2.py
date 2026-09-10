"""Additive Gen2 prompt ID adapter; current application routing remains Gen1.

Task 046 must supply authenticated, selected context. This module only frames
explicit messages and enforces tokenizer/template identity and exact budgets.
"""
from dataclasses import dataclass
from model.gen2_tokenizer.contract import ROLES, require, verify_model_binding


@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True)
class PromptIds:
    token_ids: tuple[int, ...]
    binding: dict
    content_tokens: int
    framing_tokens: int
    reserved_output_tokens: int
    context_limit: int


def compile_messages(tokenizer, messages, *, model_binding, reserved_output_tokens,
                     context_limit=4096, generation_prompt=True, for_runtime=False):
    binding = verify_model_binding(tokenizer, model_binding, for_runtime=for_runtime)
    require(type(context_limit) is int and 1 <= context_limit <= 4096, "GEN2_CONTEXT_LIMIT_INVALID")
    require(type(reserved_output_tokens) is int and 0 <= reserved_output_tokens < context_limit, "GEN2_OUTPUT_RESERVATION_INVALID")
    require(type(generation_prompt) is bool, "GEN2_GENERATION_FLAG_INVALID")
    require(not generation_prompt or reserved_output_tokens > 0, "GEN2_OUTPUT_RESERVATION_REQUIRED")
    require(isinstance(messages, (list, tuple)) and 1 <= len(messages) <= 128, "GEN2_MESSAGES_INVALID")
    total_bytes = 0
    for index, message in enumerate(messages):
        require(isinstance(message, Message) and type(message.role) is str and message.role in ROLES and type(message.content) is str, "GEN2_MESSAGE_INVALID")
        require(message.role != "system" or index == 0, "GEN2_SYSTEM_POSITION_INVALID")
        require(len(message.content) <= 1_048_576, "GEN2_CONTENT_BYTE_LIMIT")
        total_bytes += len(message.content.encode("utf-8", "surrogatepass"))
        require(total_bytes <= 1_048_576, "GEN2_CONTENT_BYTE_LIMIT")
    ids = [2]
    content_count = 0
    for message in messages:
        content = tokenizer.encode(message.content)
        ids.extend((ROLES[message.role], *content, 8))
        content_count += len(content)
        require(len(ids) + reserved_output_tokens <= context_limit, "GEN2_CONTEXT_BUDGET_EXCEEDED")
    ids.append(ROLES["assistant"] if generation_prompt else 3)
    require(len(ids) + reserved_output_tokens <= context_limit, "GEN2_CONTEXT_BUDGET_EXCEEDED")
    return PromptIds(tuple(ids), binding, content_count, len(ids) - content_count, reserved_output_tokens, context_limit)
