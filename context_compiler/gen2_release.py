"""Bind the released tokenizer, shared chat template and real decoder inputs."""
from model.gen2_tokenizer.contract import canonical, require
from context_compiler.gen2 import Message, PromptIds, compile_messages
from model.gen2_tokenizer_training.release import ReleasedTokenizer


def compile_for_model(tokenizer, model, messages, *, reserved_output_tokens, context_limit=4096, generation_prompt=True):
    require(type(tokenizer) is ReleasedTokenizer, "GEN2_CONTEXT_REQUIRES_RELEASED_TOKENIZER")
    binding = model.assert_binding(tokenizer.binding())
    # The frozen compiler already implements this exact shared contract. Its
    # default path accepts matching descriptors; the historical runtime gate
    # remains closed. This adapter does not activate application serving.
    prompt = compile_messages(tokenizer, messages, model_binding=binding, reserved_output_tokens=reserved_output_tokens, context_limit=context_limit, generation_prompt=generation_prompt, for_runtime=False)
    require(canonical(prompt.binding) == canonical(model.tokenizer_binding), "GEN2_CONTEXT_MODEL_VERSION_MISMATCH")
    return prompt
