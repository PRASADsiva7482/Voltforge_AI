"""Read-only evaluator for the quarantined pre-VFAI tokenizer.

This module is never imported by tokenizer training or runtime. It exists only
to produce the VFAI-010 comparison report after the project-owned tokenizer has
already been trained.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any


class LegacyTokenizerBaseline:
    def __init__(self, artifact_root: Path):
        self.vocab = json.loads((artifact_root / "vocab.json").read_text(encoding="utf-8"))
        self.merges = [tuple(item) for item in json.loads((artifact_root / "merges.json").read_text(encoding="utf-8"))]
        config = json.loads((artifact_root / "tokenizer_config.json").read_text(encoding="utf-8"))
        self.special_tokens = list(config["special_tokens"])
        self.unknown_id = int(self.vocab["[UNK]"])
        self.ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self._cache: dict[str, list[str]] = {}

    def _tokenize_piece(self, piece: str) -> list[str]:
        cached = self._cache.get(piece)
        if cached is not None:
            return cached
        tokens = list(piece)
        while len(tokens) >= 2:
            best_rank: int | None = None
            best_index = -1
            for index in range(len(tokens) - 1):
                rank = self.ranks.get((tokens[index], tokens[index + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_index = index
            if best_index == -1:
                break
            tokens[best_index : best_index + 2] = [tokens[best_index] + tokens[best_index + 1]]
        self._cache[piece] = tokens
        return tokens

    def encode(self, text: str) -> tuple[list[int], int]:
        ids: list[int] = []
        unknowns = 0
        special_pattern = "(" + "|".join(re.escape(item) for item in self.special_tokens) + ")"
        for segment in re.split(special_pattern, text):
            if not segment:
                continue
            if segment in self.vocab and segment in self.special_tokens:
                ids.append(int(self.vocab[segment]))
                continue
            for piece in re.findall(r"\w+|[^\w\s]|\s+", segment, re.UNICODE):
                for token in self._tokenize_piece(piece):
                    token_id = int(self.vocab.get(token, self.unknown_id))
                    ids.append(token_id)
                    unknowns += token_id == self.unknown_id
        return ids, unknowns

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


def legacy_artifact_descriptors(artifact_root: Path) -> dict[str, Any]:
    import hashlib

    descriptors = {}
    for name in ("vocab.json", "merges.json", "tokenizer_config.json"):
        path = artifact_root / name
        descriptors[name] = {
            "path": path.relative_to(artifact_root.parents[1]).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return descriptors
