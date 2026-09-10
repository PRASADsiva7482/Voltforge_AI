"""Owned BPE mathematics reused through composition; no Gen1 release is edited."""
import codecs
from collections.abc import Iterable

from model.tokenizer import VoltForgeTokenizer
from .contract import (
    BASE_SIZE, BYTE_OFFSET, FIXTURE_VERSION, NAMESPACE, SPECIAL_TOKENS,
    canonical, contract_document, require, sha, template_document,
)


class ByteBPE:
    """Strict reversible codec with a separately versioned Gen2 contract.

    Merges are owned byte-pair rules, not imported tokenizer vocabulary.
    Construction never fits text and never establishes release admission.
    """

    def __init__(self, merges=(), *, target_vocab_size=BASE_SIZE):
        require(type(target_vocab_size) is int and BASE_SIZE <= target_vocab_size <= 65536, "GEN2_VOCAB_TARGET_INVALID")
        engine = VoltForgeTokenizer(vocab_size=target_vocab_size)
        values = [None] * BYTE_OFFSET + [bytes([value]) for value in range(256)]
        seen = set(values[BYTE_OFFSET:])
        pairs = set()
        rules = []
        vocabulary_bytes = 256
        # The length cap also protects callers loading untrusted artifacts.
        for rule in merges:
            require(len(values) < target_vocab_size, "GEN2_MERGE_COUNT_INVALID")
            require(isinstance(rule, (list, tuple)) and len(rule) == 3, "GEN2_MERGE_INVALID")
            left, right, token_id = rule
            require(all(type(item) is int for item in rule), "GEN2_MERGE_INVALID")
            require(token_id == len(values) and BYTE_OFFSET <= left < token_id and BYTE_OFFSET <= right < token_id, "GEN2_MERGE_INVALID")
            require((left, right) not in pairs, "GEN2_MERGE_DUPLICATE")
            value = values[left] + values[right]
            require(len(value) <= 1_048_576 and value not in seen, "GEN2_MERGE_BYTES_INVALID")
            vocabulary_bytes += len(value)
            require(vocabulary_bytes <= 33_554_432, "GEN2_VOCAB_BYTE_LIMIT")
            values.append(value)
            seen.add(value)
            pairs.add((left, right))
            rules.append((left, right, token_id))
        engine.token_bytes = values
        engine.merges = rules
        engine._refresh_indexes()
        self._engine = engine
        self.target_vocab_size = target_vocab_size
        self._values = tuple(values)
        self.merges = tuple(rules)

    @property
    def vocab_size(self):
        return len(self._values)

    def encode_bytes(self, value: bytes) -> list[int]:
        require(type(value) is bytes, "GEN2_BYTES_REQUIRED")
        return self._engine.encode_bytes(value)

    def encode(self, text: str, *, add_bos=False, add_eos=False) -> list[int]:
        require(type(text) is str, "GEN2_TEXT_REQUIRED")
        require(type(add_bos) is bool and type(add_eos) is bool, "GEN2_BOUNDARY_FLAG_INVALID")
        # Never recognize special-token-looking text, including authored fixtures.
        return ([2] if add_bos else []) + self.encode_bytes(text.encode("utf-8", "surrogatepass")) + ([3] if add_eos else [])

    def decode_bytes(self, token_ids: Iterable[int], *, skip_special=True) -> bytes:
        require(type(skip_special) is bool, "GEN2_BOUNDARY_FLAG_INVALID")
        chunks = []
        for token_id in token_ids:
            require(type(token_id) is int and 0 <= token_id < self.vocab_size, "GEN2_TOKEN_ID_INVALID")
            if token_id < BYTE_OFFSET:
                if not skip_special:
                    chunks.append(SPECIAL_TOKENS[token_id].encode("ascii"))
            else:
                chunks.append(self._values[token_id])
        return b"".join(chunks)

    def decode(self, token_ids: Iterable[int], *, skip_special=True) -> str:
        # Arbitrary bytes use decode_bytes. Invalid UTF-8 must not silently become U+FFFD.
        return self.decode_bytes(token_ids, skip_special=skip_special).decode("utf-8", "surrogatepass")

    def stream_decoder(self):
        return IncrementalDecoder(self)

    def vocab_document(self):
        return list(SPECIAL_TOKENS) + ["hex:" + value.hex() for value in self._values[BYTE_OFFSET:]]

    def binding(self):
        return {
            "tokenizerId": NAMESPACE, "version": FIXTURE_VERSION,
            "releaseKind": "implementation-fixture-only", "vocabSize": self.vocab_size,
            "padId": 0, "unkId": 1, "bosId": 2, "eosId": 3,
            "contractSha256": sha(canonical(contract_document())),
            "templateSha256": sha(canonical(template_document())),
            "mergesSha256": sha(canonical(self.merges)),
            "vocabSha256": sha(canonical(self.vocab_document())),
        }


class IncrementalDecoder:
    """Buffer split UTF-8 sequences across generated tokens, including surrogates."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._decoder = codecs.getincrementaldecoder("utf-8")("surrogatepass")
        self._closed = False

    def push(self, token_ids=(), *, final=False):
        require(not self._closed and type(final) is bool, "GEN2_DECODER_CLOSED_OR_FLAG_INVALID")
        raw = self._tokenizer.decode_bytes(token_ids)
        try:
            result = self._decoder.decode(raw, final=final)
        except UnicodeDecodeError:
            self._closed = True
            raise
        self._closed = final
        return result
