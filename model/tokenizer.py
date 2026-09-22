"""VoltForge-owned deterministic byte-level BPE tokenizer.

The implementation trains from raw UTF-8 bytes and project-approved text. It
does not import a tokenizer library, pretrained vocabulary, pretrained merges,
or external model asset.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import heapq
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


TOKENIZER_CONTRACT_VERSION = "1.0.0"
TOKENIZER_ALGORITHM = "voltforge-byte-bpe"
TOKENIZER_ARTIFACT_FILENAMES = {
    "vocab": "vocab.json",
    "merges": "merges.json",
    "config": "tokenizer_config.json",
    "manifest": "tokenizer_manifest.json",
}
DEFAULT_TOKENIZER_RELEASE_PATH = (
    Path(__file__).resolve().parent / "tokenizers" / "vfdlm-byte-bpe-v1.0.0"
)
SPECIAL_TOKENS = (
    "[PAD]",
    "[UNK]",
    "[BOS]",
    "[EOS]",
    "[SYS]",
    "[USER]",
    "[ASSISTANT]",
    "[CODE]",
    "[PROJECT_CONTEXT]",
    "[TOOL_EVIDENCE]",
    "[STRUCTURED_ACTION]",
    "[CITATION]",
    "[REFUSAL]",
    "[UNCERTAINTY]",
    "[SEP]",
    "[MASK]",
)
SPECIAL_TOKEN_TO_ID = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
BYTE_TOKEN_OFFSET = len(SPECIAL_TOKENS)
BASE_VOCAB_SIZE = BYTE_TOKEN_OFFSET + 256
SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")


class TokenizerContractError(ValueError):
    """A tokenizer artifact or operation violated the immutable v1 contract."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_bytes(data)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _encode_text_bytes(text: str) -> bytes:
    return text.encode("utf-8", errors="surrogatepass")


class _PairTrainer:
    """Incremental linked-list BPE trainer with deterministic pair tie breaks."""

    def __init__(self, documents: Sequence[bytes]):
        self.tokens: list[int] = []
        self.previous: list[int] = []
        self.next: list[int] = []
        self.active: list[bool] = []
        for document in documents:
            start = len(self.tokens)
            for offset, byte in enumerate(document):
                index = len(self.tokens)
                self.tokens.append(BYTE_TOKEN_OFFSET + byte)
                self.previous.append(index - 1 if offset else -1)
                self.next.append(index + 1 if offset + 1 < len(document) else -1)
                self.active.append(True)
            if document:
                self.previous[start] = -1

        self.occurrences: dict[tuple[int, int], set[int]] = defaultdict(set)
        for left, right in enumerate(self.next):
            if right != -1:
                self.occurrences[(self.tokens[left], self.tokens[right])].add(left)
        self.versions: dict[tuple[int, int], int] = defaultdict(int)
        self.heap: list[tuple[int, int, int, int]] = []
        for pair, positions in self.occurrences.items():
            heapq.heappush(self.heap, (-len(positions), pair[0], pair[1], 0))

    def _pair_at(self, left: int) -> tuple[int, int] | None:
        if left < 0 or left >= len(self.tokens) or not self.active[left]:
            return None
        right = self.next[left]
        if right == -1 or not self.active[right]:
            return None
        return self.tokens[left], self.tokens[right]

    def _remove(self, pair: tuple[int, int] | None, left: int, dirty: set[tuple[int, int]]) -> None:
        if pair is None:
            return
        positions = self.occurrences.get(pair)
        if positions is not None and left in positions:
            positions.remove(left)
            dirty.add(pair)

    def _add(self, pair: tuple[int, int] | None, left: int, dirty: set[tuple[int, int]]) -> None:
        if pair is None:
            return
        positions = self.occurrences[pair]
        if left not in positions:
            positions.add(left)
            dirty.add(pair)

    def _best_pair(self, minimum_frequency: int) -> tuple[int, int] | None:
        while self.heap:
            negative_count, left_token, right_token, version = heapq.heappop(self.heap)
            pair = (left_token, right_token)
            positions = self.occurrences.get(pair, set())
            if version != self.versions[pair] or -negative_count != len(positions):
                continue
            if len(positions) < minimum_frequency:
                return None
            return pair
        return None

    def train(
        self,
        *,
        target_vocab_size: int,
        minimum_frequency: int,
        token_bytes: list[bytes | None],
    ) -> list[tuple[int, int, int]]:
        merges: list[tuple[int, int, int]] = []
        byte_values = {item for item in token_bytes if item is not None}
        while len(token_bytes) < target_vocab_size:
            pair = self._best_pair(minimum_frequency)
            if pair is None:
                break
            new_token_id = len(token_bytes)
            left_bytes = token_bytes[pair[0]]
            right_bytes = token_bytes[pair[1]]
            if left_bytes is None or right_bytes is None:
                raise TokenizerContractError(
                    "TOKENIZER_TRAINING_STATE_INVALID", "BPE attempted to merge a special token."
                )
            merged_bytes = left_bytes + right_bytes
            if merged_bytes in byte_values:
                raise TokenizerContractError(
                    "TOKENIZER_DUPLICATE_TOKEN", "BPE produced duplicate byte content."
                )
            byte_values.add(merged_bytes)
            token_bytes.append(merged_bytes)
            merges.append((pair[0], pair[1], new_token_id))

            dirty: set[tuple[int, int]] = set()
            for left in sorted(tuple(self.occurrences[pair])):
                right = self.next[left] if 0 <= left < len(self.next) else -1
                if (
                    right == -1
                    or not self.active[left]
                    or not self.active[right]
                    or self.tokens[left] != pair[0]
                    or self.tokens[right] != pair[1]
                ):
                    self._remove(pair, left, dirty)
                    continue
                before = self.previous[left]
                after = self.next[right]
                if before != -1:
                    self._remove(self._pair_at(before), before, dirty)
                self._remove(pair, left, dirty)
                if after != -1:
                    self._remove(self._pair_at(right), right, dirty)

                self.tokens[left] = new_token_id
                self.active[right] = False
                self.next[left] = after
                if after != -1:
                    self.previous[after] = left

                if before != -1:
                    self._add(self._pair_at(before), before, dirty)
                self._add(self._pair_at(left), left, dirty)

            for changed_pair in dirty:
                self.versions[changed_pair] += 1
                count = len(self.occurrences.get(changed_pair, ()))
                if count:
                    heapq.heappush(
                        self.heap,
                        (-count, changed_pair[0], changed_pair[1], self.versions[changed_pair]),
                    )
        return merges


class VoltForgeTokenizer:
    """Lossless byte BPE with fixed special IDs and checksum-bound persistence."""

    def __init__(self, vocab_size: int = 2048):
        if vocab_size < BASE_VOCAB_SIZE:
            raise TokenizerContractError(
                "TOKENIZER_VOCAB_TOO_SMALL",
                f"Byte-level vocabulary requires at least {BASE_VOCAB_SIZE} entries.",
            )
        self.target_vocab_size = int(vocab_size)
        self.vocab_size = BASE_VOCAB_SIZE
        self.special_tokens = list(SPECIAL_TOKENS)
        self.special_token_to_id = dict(SPECIAL_TOKEN_TO_ID)
        self.id_to_special_token = {value: key for key, value in SPECIAL_TOKEN_TO_ID.items()}
        self.token_bytes: list[bytes | None] = [None] * BYTE_TOKEN_OFFSET + [bytes([value]) for value in range(256)]
        self.merges: list[tuple[int, int, int]] = []
        self.merge_ranks: dict[tuple[int, int], tuple[int, int]] = {}
        self.vocab: dict[str, int] = {}
        self.id_to_vocab: dict[int, str] = {}
        self.tokenizer_id = "vfdlm-byte-bpe"
        self.version = TOKENIZER_CONTRACT_VERSION
        self.training_metadata: dict[str, Any] = {}
        self._refresh_indexes()

    def _refresh_indexes(self) -> None:
        vocab: dict[str, int] = dict(self.special_token_to_id)
        for token_id in range(BYTE_TOKEN_OFFSET, len(self.token_bytes)):
            value = self.token_bytes[token_id]
            if value is None:
                raise TokenizerContractError(
                    "TOKENIZER_VOCAB_INVALID", f"Byte token {token_id} has no byte value."
                )
            key = "hex:" + value.hex()
            if key in vocab:
                raise TokenizerContractError(
                    "TOKENIZER_DUPLICATE_TOKEN", f"Duplicate byte token content at ID {token_id}."
                )
            vocab[key] = token_id
        self.vocab = vocab
        self.id_to_vocab = {token_id: token for token, token_id in vocab.items()}
        self.vocab_size = len(self.token_bytes)
        self.merge_ranks = {
            (left, right): (rank, token_id)
            for rank, (left, right, token_id) in enumerate(self.merges)
        }

    def train(
        self,
        texts: Iterable[str | bytes],
        target_vocab_size: int | None = None,
        *,
        minimum_frequency: int = 2,
        training_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        target = int(target_vocab_size or self.target_vocab_size)
        if target < BASE_VOCAB_SIZE or target > 65_536:
            raise TokenizerContractError(
                "TOKENIZER_VOCAB_SIZE_INVALID",
                f"Target vocabulary must be between {BASE_VOCAB_SIZE} and 65536.",
            )
        if minimum_frequency < 2:
            raise TokenizerContractError(
                "TOKENIZER_FREQUENCY_INVALID", "Minimum BPE pair frequency must be at least two."
            )
        documents = [value if isinstance(value, bytes) else _encode_text_bytes(value) for value in texts]
        if not documents or not any(documents):
            raise TokenizerContractError(
                "TOKENIZER_CORPUS_EMPTY", "Tokenizer training requires non-empty project text."
            )
        self.target_vocab_size = target
        self.token_bytes = [None] * BYTE_TOKEN_OFFSET + [bytes([value]) for value in range(256)]
        trainer = _PairTrainer(documents)
        self.merges = trainer.train(
            target_vocab_size=target,
            minimum_frequency=minimum_frequency,
            token_bytes=self.token_bytes,
        )
        self.training_metadata = dict(training_metadata or {})
        self._refresh_indexes()

    def encode_bytes(self, value: bytes) -> list[int]:
        if not value:
            return []
        tokens = [BYTE_TOKEN_OFFSET + byte for byte in value]
        previous = [index - 1 if index else -1 for index in range(len(tokens))]
        following = [index + 1 if index + 1 < len(tokens) else -1 for index in range(len(tokens))]
        active = [True] * len(tokens)
        heap: list[tuple[int, int, int, int, int]] = []

        def push(left: int) -> None:
            if left == -1 or not active[left]:
                return
            right = following[left]
            if right == -1 or not active[right]:
                return
            merge = self.merge_ranks.get((tokens[left], tokens[right]))
            if merge is not None:
                rank, new_token = merge
                heapq.heappush(heap, (rank, left, tokens[left], tokens[right], new_token))

        for index in range(len(tokens) - 1):
            push(index)
        while heap:
            rank, left, expected_left, expected_right, new_token = heapq.heappop(heap)
            if not active[left] or tokens[left] != expected_left:
                continue
            right = following[left]
            if right == -1 or not active[right] or tokens[right] != expected_right:
                continue
            current = self.merge_ranks.get((tokens[left], tokens[right]))
            if current != (rank, new_token):
                continue
            before = previous[left]
            after = following[right]
            tokens[left] = new_token
            active[right] = False
            following[left] = after
            if after != -1:
                previous[after] = left
            push(before)
            push(left)

        encoded: list[int] = []
        index = 0
        while index != -1:
            if active[index]:
                encoded.append(tokens[index])
            index = following[index]
        return encoded

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        *,
        allowed_special: bool = False,
    ) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.special_token_to_id["[BOS]"])
        if allowed_special:
            pattern = "(" + "|".join(re.escape(token) for token in sorted(SPECIAL_TOKENS, key=len, reverse=True)) + ")"
            for segment in re.split(pattern, text):
                if not segment:
                    continue
                special_id = self.special_token_to_id.get(segment)
                ids.extend([special_id] if special_id is not None else self.encode_bytes(_encode_text_bytes(segment)))
        else:
            ids.extend(self.encode_bytes(_encode_text_bytes(text)))
        if add_eos:
            ids.append(self.special_token_to_id["[EOS]"])
        return ids

    def decode_bytes(self, token_ids: Iterable[int], *, skip_special: bool = True) -> bytes:
        chunks: list[bytes] = []
        for token_id in token_ids:
            if not isinstance(token_id, int):
                raise TokenizerContractError(
                    "TOKENIZER_ID_INVALID", "Token IDs must be integers."
                )
            if token_id in self.id_to_special_token:
                if not skip_special:
                    chunks.append(self.id_to_special_token[token_id].encode("ascii"))
                continue
            if token_id < 0 or token_id >= len(self.token_bytes):
                raise TokenizerContractError(
                    "TOKENIZER_ID_OUT_OF_RANGE", f"Token ID {token_id} is outside the vocabulary."
                )
            token = self.token_bytes[token_id]
            if token is None:
                raise TokenizerContractError(
                    "TOKENIZER_VOCAB_INVALID", f"Token ID {token_id} has no byte representation."
                )
            chunks.append(token)
        return b"".join(chunks)

    def decode(self, token_ids: Iterable[int], skip_special: bool = True) -> str:
        value = self.decode_bytes(token_ids, skip_special=skip_special)
        try:
            return value.decode("utf-8", errors="surrogatepass")
        except UnicodeDecodeError:
            return value.decode("utf-8", errors="replace")

    def artifact_documents(self) -> tuple[dict[str, int], list[dict[str, int]], dict[str, Any]]:
        vocab = {token: token_id for token, token_id in sorted(self.vocab.items(), key=lambda item: item[1])}
        merges = [
            {"rank": rank, "leftId": left, "rightId": right, "tokenId": token_id}
            for rank, (left, right, token_id) in enumerate(self.merges)
        ]
        config = {
            "schemaVersion": 1,
            "contractVersion": TOKENIZER_CONTRACT_VERSION,
            "tokenizerId": self.tokenizer_id,
            "version": self.version,
            "algorithm": TOKENIZER_ALGORITHM,
            "encoding": "UTF-8 with surrogatepass for lossless malformed-Unicode handling",
            "vocab_size": self.vocab_size,
            "baseByteTokenCount": 256,
            "mergeCount": len(self.merges),
            "special_tokens": self.special_tokens,
            "specialTokenIds": self.special_token_to_id,
            "byteTokenOffset": BYTE_TOKEN_OFFSET,
            "unknownTokenReachableForBytes": False,
            "defaultSpecialParsing": False,
            "training": self.training_metadata,
        }
        return vocab, merges, config

    def save(
        self,
        directory: str | Path,
        *,
        release_status: str = "development",
        lineage: Mapping[str, Any] | None = None,
    ) -> Path:
        destination = Path(directory)
        vocab, merges, config = self.artifact_documents()
        paths = {
            key: destination / filename for key, filename in TOKENIZER_ARTIFACT_FILENAMES.items()
        }
        _write_atomic(paths["vocab"], _json_bytes(vocab))
        _write_atomic(paths["merges"], _json_bytes(merges))
        config["vocabSha256"] = _sha256_file(paths["vocab"])
        config["mergesSha256"] = _sha256_file(paths["merges"])
        config["specialTokenContractSha256"] = _sha256_bytes(
            _canonical_json(self.special_token_to_id).encode("utf-8")
        )
        _write_atomic(paths["config"], _json_bytes(config))
        manifest = {
            "schemaVersion": 1,
            "artifactKind": "voltforge-tokenizer",
            "tokenizerId": self.tokenizer_id,
            "version": self.version,
            "contractVersion": TOKENIZER_CONTRACT_VERSION,
            "releaseStatus": release_status,
            "algorithm": TOKENIZER_ALGORITHM,
            "vocabSize": self.vocab_size,
            "mergeCount": len(self.merges),
            "specialTokenContractSha256": config["specialTokenContractSha256"],
            "files": {
                key: {"path": paths[key].name, "sha256": _sha256_file(paths[key])}
                for key in ("vocab", "merges", "config")
            },
            "lineage": dict(lineage or {}),
        }
        manifest["artifactSha256"] = _sha256_bytes(_canonical_json(manifest).encode("utf-8"))
        _write_atomic(paths["manifest"], _json_bytes(manifest))
        return paths["manifest"]

    def load(self, directory: str | Path) -> None:
        source = Path(directory)
        paths = {key: source / filename for key, filename in TOKENIZER_ARTIFACT_FILENAMES.items()}
        try:
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            config = json.loads(paths["config"].read_text(encoding="utf-8"))
            vocab = json.loads(paths["vocab"].read_text(encoding="utf-8"))
            merges = json.loads(paths["merges"].read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise TokenizerContractError(
                "TOKENIZER_ARTIFACT_MISSING", f"Tokenizer artifact is missing: {error.filename}"
            ) from error
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise TokenizerContractError(
                "TOKENIZER_ARTIFACT_INVALID", "Tokenizer artifact JSON is unreadable."
            ) from error
        if not isinstance(manifest, dict) or not isinstance(config, dict):
            raise TokenizerContractError("TOKENIZER_ARTIFACT_INVALID", "Tokenizer metadata must be objects.")
        declared_artifact_hash = manifest.get("artifactSha256")
        actual_artifact_hash = _sha256_bytes(
            _canonical_json({key: value for key, value in manifest.items() if key != "artifactSha256"}).encode("utf-8")
        )
        if declared_artifact_hash != actual_artifact_hash:
            raise TokenizerContractError("TOKENIZER_MANIFEST_CHECKSUM_MISMATCH", "Tokenizer manifest was modified.")
        if manifest.get("algorithm") != TOKENIZER_ALGORITHM or config.get("algorithm") != TOKENIZER_ALGORITHM:
            raise TokenizerContractError("TOKENIZER_ALGORITHM_UNSUPPORTED", "Tokenizer algorithm is unsupported.")
        for key in ("vocab", "merges", "config"):
            descriptor = (manifest.get("files") or {}).get(key)
            if not isinstance(descriptor, dict) or descriptor.get("path") != paths[key].name:
                raise TokenizerContractError("TOKENIZER_MANIFEST_INVALID", f"Tokenizer {key} descriptor is invalid.")
            if descriptor.get("sha256") != _sha256_file(paths[key]):
                raise TokenizerContractError("TOKENIZER_FILE_CHECKSUM_MISMATCH", f"Tokenizer {key} checksum changed.")
        if config.get("specialTokenIds") != self.special_token_to_id or config.get("special_tokens") != self.special_tokens:
            raise TokenizerContractError("TOKENIZER_SPECIAL_CONTRACT_MISMATCH", "Special-token IDs changed.")
        if not isinstance(vocab, dict) or not isinstance(merges, list):
            raise TokenizerContractError("TOKENIZER_VOCAB_INVALID", "Tokenizer vocabulary or merges are invalid.")
        ids = list(vocab.values())
        if ids != list(range(len(ids))) or int(config.get("vocab_size", -1)) != len(ids):
            raise TokenizerContractError("TOKENIZER_VOCAB_INVALID", "Tokenizer IDs must be contiguous and sized exactly.")

        token_bytes: list[bytes | None] = [None] * len(ids)
        for token, token_id in vocab.items():
            if token_id < BYTE_TOKEN_OFFSET:
                if SPECIAL_TOKENS[token_id] != token:
                    raise TokenizerContractError("TOKENIZER_SPECIAL_CONTRACT_MISMATCH", "Special token ordering changed.")
            else:
                if not isinstance(token, str) or not token.startswith("hex:"):
                    raise TokenizerContractError("TOKENIZER_VOCAB_INVALID", "Byte token encoding is invalid.")
                try:
                    token_bytes[token_id] = bytes.fromhex(token[4:])
                except ValueError as error:
                    raise TokenizerContractError("TOKENIZER_VOCAB_INVALID", "Byte token hex is invalid.") from error
        loaded_merges: list[tuple[int, int, int]] = []
        for rank, item in enumerate(merges):
            if not isinstance(item, dict) or item.get("rank") != rank:
                raise TokenizerContractError("TOKENIZER_MERGES_INVALID", "Merge ranks must be contiguous.")
            left, right, token_id = item.get("leftId"), item.get("rightId"), item.get("tokenId")
            if not all(isinstance(value, int) for value in (left, right, token_id)):
                raise TokenizerContractError("TOKENIZER_MERGES_INVALID", "Merge IDs must be integers.")
            if token_id != BASE_VOCAB_SIZE + rank or left >= token_id or right >= token_id:
                raise TokenizerContractError("TOKENIZER_MERGES_INVALID", "Merge dependency ordering is invalid.")
            left_bytes, right_bytes = token_bytes[left], token_bytes[right]
            if left_bytes is None or right_bytes is None or token_bytes[token_id] != left_bytes + right_bytes:
                raise TokenizerContractError("TOKENIZER_MERGES_INVALID", "Merge bytes do not match vocabulary.")
            loaded_merges.append((left, right, token_id))
        if len(loaded_merges) != int(config.get("mergeCount", -1)):
            raise TokenizerContractError("TOKENIZER_MERGES_INVALID", "Merge count changed.")
        self.tokenizer_id = str(config["tokenizerId"])
        self.version = str(config["version"])
        self.token_bytes = token_bytes
        self.merges = loaded_merges
        self.training_metadata = dict(config.get("training") or {})
        self.target_vocab_size = len(token_bytes)
        self._refresh_indexes()

    def dependency_descriptor(self, directory: str | Path) -> dict[str, Any]:
        manifest_path = Path(directory) / TOKENIZER_ARTIFACT_FILENAMES["manifest"]
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise TokenizerContractError("TOKENIZER_MANIFEST_INVALID", "Tokenizer manifest is unavailable.") from error
        return {
            "tokenizerId": manifest.get("tokenizerId"),
            "version": manifest.get("version"),
            "vocabSize": manifest.get("vocabSize"),
            "artifactSha256": manifest.get("artifactSha256"),
            "manifestSha256": _sha256_file(manifest_path),
        }
