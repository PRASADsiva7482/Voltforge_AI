"""Approved-corpus loading, lossless packing, and resumable batch cursors."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor

from data_governance.governance import (
    default_manifest_path,
    require_approved_shard,
    sha256_file,
)
from evaluation.leakage import exclude_held_out_records
from model.gen1.config import Gen1Config
from model.tokenizer import DEFAULT_TOKENIZER_RELEASE_PATH, VoltForgeTokenizer
from task_schema.compiler import compile_task_record
from task_schema.io import read_task_shard


AI_ROOT = Path(__file__).resolve().parents[1]
APPROVED_CORPUS_ID = "vfai009-approved-packed-v1"


class TrainingDataContractError(RuntimeError):
    """Raised when training data, lineage, split, or packed tensors are invalid."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _corpus_hash(texts: Iterable[str]) -> str:
    return _sha256_bytes(b"".join(text.encode("utf-8") + b"\0" for text in texts))


def _token_hash(values: Iterable[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(int(value).to_bytes(4, "little", signed=False))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PackedSplit:
    input_ids: Tensor
    labels: Tensor
    attention_mask: Tensor
    token_count: int
    predicted_token_count: int
    stream_sha256: str

    def __post_init__(self) -> None:
        if self.input_ids.dtype != torch.long or self.labels.dtype != torch.long:
            raise TrainingDataContractError("packed token and label tensors must be torch.long")
        if self.attention_mask.dtype != torch.bool:
            raise TrainingDataContractError("packed attention masks must be torch.bool")
        if self.input_ids.ndim != 2 or self.input_ids.shape != self.labels.shape:
            raise TrainingDataContractError("packed inputs and labels must have matching rank-2 shapes")
        if self.attention_mask.shape != self.input_ids.shape:
            raise TrainingDataContractError("packed attention mask shape must match input_ids")
        actual_predictions = int((self.labels[:, 1:] != -100).sum().item())
        if actual_predictions != self.predicted_token_count:
            raise TrainingDataContractError("packed predicted-token accounting mismatch")
        self.validate_integrity()

    def validate_integrity(self) -> None:
        stream: list[int] = []
        previous_last: int | None = None
        for row_index in range(self.input_ids.shape[0]):
            mask = self.attention_mask[row_index]
            valid_length = int(mask.sum().item())
            if valid_length < 2 or not torch.all(mask[:valid_length]) or torch.any(mask[valid_length:]):
                raise TrainingDataContractError("packed attention masks must be contiguous prefixes")
            inputs = self.input_ids[row_index, :valid_length]
            labels = self.labels[row_index]
            if not torch.equal(inputs, labels[:valid_length]) or torch.any(
                labels[valid_length:] != -100
            ):
                raise TrainingDataContractError("packed labels do not match valid input tokens")
            values = inputs.tolist()
            if previous_last is not None and values[0] != previous_last:
                raise TrainingDataContractError("adjacent packed blocks lost their overlap token")
            stream.extend(values if not stream else values[1:])
            previous_last = values[-1]
        if (
            len(stream) != self.token_count
            or len(stream) - 1 != self.predicted_token_count
            or _token_hash(stream) != self.stream_sha256
        ):
            raise TrainingDataContractError("packed token stream checksum/accounting mismatch")

    @property
    def block_count(self) -> int:
        return self.input_ids.shape[0]

    @property
    def block_size(self) -> int:
        return self.input_ids.shape[1]


@dataclass(frozen=True, slots=True)
class PackedCorpus:
    train: PackedSplit
    validation: PackedSplit
    manifest: Mapping[str, Any]
    fingerprint: str
    corpus_kind: str

    @classmethod
    def for_testing(
        cls,
        *,
        training_sequences: Sequence[Sequence[int]],
        validation_sequences: Sequence[Sequence[int]],
        block_size: int,
        pad_token_id: int = 0,
        dataset_id: str = "unit-test-corpus",
    ) -> "PackedCorpus":
        train = pack_token_sequences(training_sequences, block_size, pad_token_id)
        validation = pack_token_sequences(validation_sequences, block_size, pad_token_id)
        manifest = {
            "schemaVersion": 1,
            "datasetId": dataset_id,
            "approvalStatus": "unit-test-only",
            "blockSize": block_size,
            "training": _split_descriptor(train),
            "validation": _split_descriptor(validation),
        }
        fingerprint = _sha256_bytes(_canonical_json(manifest).encode("utf-8"))
        return cls(train, validation, manifest, fingerprint, "unit-test")


def pack_token_sequences(
    sequences: Sequence[Sequence[int]], block_size: int, pad_token_id: int
) -> PackedSplit:
    """Pack a token stream with one-token overlap so no transition is dropped."""

    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 2:
        raise TrainingDataContractError("block_size must be an integer >= 2")
    if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int) or pad_token_id < 0:
        raise TrainingDataContractError("pad_token_id must be a non-negative integer")
    stream: list[int] = []
    for sequence in sequences:
        if len(sequence) < 2:
            raise TrainingDataContractError("every packed document must contain at least two tokens")
        for token_id in sequence:
            if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
                raise TrainingDataContractError("packed token IDs must be non-negative integers")
            stream.append(token_id)
    if len(stream) < 2:
        raise TrainingDataContractError("packed corpus must contain at least two tokens")

    inputs: list[list[int]] = []
    labels: list[list[int]] = []
    masks: list[list[bool]] = []
    start = 0
    while start < len(stream) - 1:
        chunk = stream[start : start + block_size]
        if len(chunk) < 2:
            break
        valid_length = len(chunk)
        padding = block_size - valid_length
        inputs.append(chunk + [pad_token_id] * padding)
        labels.append(chunk + [-100] * padding)
        masks.append([True] * valid_length + [False] * padding)
        start += block_size - 1

    packed = PackedSplit(
        input_ids=torch.tensor(inputs, dtype=torch.long),
        labels=torch.tensor(labels, dtype=torch.long),
        attention_mask=torch.tensor(masks, dtype=torch.bool),
        token_count=len(stream),
        predicted_token_count=len(stream) - 1,
        stream_sha256=_token_hash(stream),
    )
    if packed.predicted_token_count != len(stream) - 1:
        raise TrainingDataContractError("packing dropped or duplicated next-token targets")
    return packed


def _split_descriptor(split: PackedSplit) -> dict[str, Any]:
    return {
        "blockCount": split.block_count,
        "blockSize": split.block_size,
        "tokenCount": split.token_count,
        "predictedTokenCount": split.predicted_token_count,
        "tokenStreamSha256": split.stream_sha256,
    }


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrainingDataContractError(f"unable to read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise TrainingDataContractError(f"{label} must be a JSON object")
    return value


def _resolve_approved_path(relative_path: Any) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise TrainingDataContractError("approved shard path is missing")
    candidate = (AI_ROOT / relative_path).resolve()
    try:
        candidate.relative_to(AI_ROOT.resolve())
    except ValueError as exc:
        raise TrainingDataContractError("approved shard path escapes the AI workspace") from exc
    return candidate


def load_approved_corpus(
    model_config: Gen1Config,
    *,
    tokenizer_directory: str | Path = DEFAULT_TOKENIZER_RELEASE_PATH,
    packing_block_size: int | None = None,
) -> PackedCorpus:
    """Load only checksum-approved VFAI-009 records using the frozen VFAI-010 split."""

    block_size = packing_block_size or model_config.max_sequence_length
    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size < 2
        or block_size > model_config.max_sequence_length
    ):
        raise TrainingDataContractError(
            "packing_block_size must be between 2 and the model max_sequence_length"
        )
    tokenizer_root = Path(tokenizer_directory)
    tokenizer = VoltForgeTokenizer(model_config.vocab_size)
    tokenizer.load(tokenizer_root)
    tokenizer_manifest_path = tokenizer_root / "tokenizer_manifest.json"
    tokenizer_manifest = _read_json_object(tokenizer_manifest_path, "tokenizer manifest")
    if tokenizer_manifest.get("releaseStatus") != "approved":
        raise TrainingDataContractError("tokenizer release is not approved")
    if tokenizer_manifest.get("vocabSize") != model_config.vocab_size:
        raise TrainingDataContractError("model vocabulary does not match the approved tokenizer")
    lineage = tokenizer_manifest.get("lineage")
    split = lineage.get("split") if isinstance(lineage, Mapping) else None
    shards = lineage.get("shards") if isinstance(lineage, Mapping) else None
    if not isinstance(split, Mapping) or not isinstance(shards, list) or not shards:
        raise TrainingDataContractError("tokenizer manifest lacks immutable data lineage/split")

    records: list[dict[str, Any]] = []
    shard_lineage: list[dict[str, Any]] = []
    for descriptor in shards:
        if not isinstance(descriptor, Mapping):
            raise TrainingDataContractError("tokenizer shard descriptor is invalid")
        shard_path = _resolve_approved_path(descriptor.get("path"))
        manifest_path = _resolve_approved_path(descriptor.get("manifestPath"))
        governed = require_approved_shard(
            shard_path, "training", manifest_path=manifest_path
        )
        expected = {
            "shardId": descriptor.get("shardId"),
            "sha256": descriptor.get("sha256"),
            "recordCount": descriptor.get("recordCount"),
        }
        actual = {
            "shardId": governed.get("shardId"),
            "sha256": governed.get("sha256"),
            "recordCount": governed.get("recordCount"),
        }
        if actual != expected or sha256_file(manifest_path) != descriptor.get("manifestSha256"):
            raise TrainingDataContractError("approved shard differs from tokenizer lineage")
        shard_records = read_task_shard(shard_path)
        records.extend(shard_records)
        shard_lineage.append(
            {
                "shardId": governed["shardId"],
                "path": descriptor["path"],
                "sha256": governed["sha256"],
                "recordCount": governed["recordCount"],
                "manifestPath": descriptor["manifestPath"],
                "manifestSha256": descriptor["manifestSha256"],
                "sourceIds": descriptor["sourceIds"],
            }
        )

    if len({record.get("recordId") for record in records}) != len(records):
        raise TrainingDataContractError("approved training record IDs are not unique")
    accepted, held_out = exclude_held_out_records(records)
    if held_out or len(accepted) != len(records):
        raise TrainingDataContractError("approved model corpus collides with frozen release evaluation")
    evaluation_ids = split.get("evaluationRecordIds")
    if not isinstance(evaluation_ids, list) or len(set(evaluation_ids)) != len(evaluation_ids):
        raise TrainingDataContractError("frozen validation record IDs are invalid")
    evaluation_id_set = set(evaluation_ids)
    if not evaluation_id_set.issubset({record["recordId"] for record in records}):
        raise TrainingDataContractError("frozen validation split references an unknown record")
    training_records = sorted(
        (record for record in records if record["recordId"] not in evaluation_id_set),
        key=lambda record: record["recordId"],
    )
    validation_records = sorted(
        (record for record in records if record["recordId"] in evaluation_id_set),
        key=lambda record: record["recordId"],
    )
    if len(training_records) != split.get("trainingRecordCount") or len(
        validation_records
    ) != split.get("evaluationRecordCount"):
        raise TrainingDataContractError("frozen split record counts changed")

    training_texts = [compile_task_record(record) for record in training_records]
    validation_texts = [compile_task_record(record) for record in validation_records]
    training_ids_hash = _sha256_bytes(
        "\n".join(record["recordId"] for record in training_records).encode("utf-8")
    )
    if training_ids_hash != split.get("trainingRecordIdsSha256"):
        raise TrainingDataContractError("frozen training record IDs changed")
    if _corpus_hash(training_texts) != split.get("trainingCorpusSha256"):
        raise TrainingDataContractError("frozen training corpus content changed")
    if _corpus_hash(validation_texts) != split.get("evaluationCorpusSha256"):
        raise TrainingDataContractError("frozen validation corpus content changed")

    bos = tokenizer.special_token_to_id["[BOS]"]
    eos = tokenizer.special_token_to_id["[EOS]"]
    pad = tokenizer.special_token_to_id["[PAD]"]
    training_sequences = [[bos, *tokenizer.encode(text), eos] for text in training_texts]
    validation_sequences = [[bos, *tokenizer.encode(text), eos] for text in validation_texts]
    training_packed = pack_token_sequences(
        training_sequences, block_size, pad
    )
    validation_packed = pack_token_sequences(
        validation_sequences, block_size, pad
    )
    manifest = {
        "schemaVersion": 1,
        "datasetId": APPROVED_CORPUS_ID,
        "approvalStatus": "approved",
        "splitId": split.get("id"),
        "splitSeed": split.get("seed"),
        "blockSize": block_size,
        "sourceIds": sorted(
            {source_id for shard in shard_lineage for source_id in shard["sourceIds"]}
        ),
        "shards": shard_lineage,
        "trainingRecordCount": len(training_records),
        "validationRecordCount": len(validation_records),
        "trainingRecordIdsSha256": training_ids_hash,
        "validationRecordIds": evaluation_ids,
        "trainingCorpusSha256": split["trainingCorpusSha256"],
        "validationCorpusSha256": split["evaluationCorpusSha256"],
        "training": _split_descriptor(training_packed),
        "validation": _split_descriptor(validation_packed),
        "tokenizer": tokenizer.dependency_descriptor(tokenizer_root),
    }
    fingerprint = _sha256_bytes(_canonical_json(manifest).encode("utf-8"))
    return PackedCorpus(
        training_packed,
        validation_packed,
        manifest,
        fingerprint,
        "approved-vfai009",
    )


class PackedBatchStream:
    """Deterministically shuffled, fixed-size batches with an exact resume cursor."""

    def __init__(self, split: PackedSplit, batch_size: int, seed: int) -> None:
        if split.block_count < 1:
            raise TrainingDataContractError("training split has no packed blocks")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise TrainingDataContractError("batch_size must be a positive integer")
        self.split = split
        self.batch_size = batch_size
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)
        self.epoch = 0
        self.position = 0
        self.batches_emitted = 0
        self.examples_emitted = 0
        self.permutation = self._next_permutation()

    def _next_permutation(self) -> list[int]:
        return torch.randperm(self.split.block_count, generator=self.generator).tolist()

    def next_batch(self, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
        indexes: list[int] = []
        while len(indexes) < self.batch_size:
            remaining = len(self.permutation) - self.position
            take = min(self.batch_size - len(indexes), remaining)
            indexes.extend(self.permutation[self.position : self.position + take])
            self.position += take
            if self.position == len(self.permutation):
                self.epoch += 1
                self.position = 0
                self.permutation = self._next_permutation()
        self.batches_emitted += 1
        self.examples_emitted += len(indexes)
        index = torch.tensor(indexes, dtype=torch.long)
        return (
            self.split.input_ids.index_select(0, index).to(device),
            self.split.labels.index_select(0, index).to(device),
            self.split.attention_mask.index_select(0, index).to(device),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "blockCount": self.split.block_count,
            "batchSize": self.batch_size,
            "epoch": self.epoch,
            "position": self.position,
            "permutation": list(self.permutation),
            "generatorState": self.generator.get_state(),
            "batchesEmitted": self.batches_emitted,
            "examplesEmitted": self.examples_emitted,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "schemaVersion",
            "blockCount",
            "batchSize",
            "epoch",
            "position",
            "permutation",
            "generatorState",
            "batchesEmitted",
            "examplesEmitted",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise TrainingDataContractError("packed batch cursor keys are invalid")
        permutation = state["permutation"]
        if (
            state["schemaVersion"] != 1
            or state["blockCount"] != self.split.block_count
            or state["batchSize"] != self.batch_size
            or not isinstance(permutation, list)
            or sorted(permutation) != list(range(self.split.block_count))
            or not 0 <= state["position"] < self.split.block_count
        ):
            raise TrainingDataContractError("packed batch cursor is incompatible")
        generator_state = state["generatorState"]
        if not isinstance(generator_state, Tensor) or generator_state.dtype != torch.uint8:
            raise TrainingDataContractError("packed batch RNG state is invalid")
        self.epoch = int(state["epoch"])
        self.position = int(state["position"])
        self.permutation = list(permutation)
        self.batches_emitted = int(state["batchesEmitted"])
        self.examples_emitted = int(state["examplesEmitted"])
        self.generator.set_state(generator_state.cpu())
