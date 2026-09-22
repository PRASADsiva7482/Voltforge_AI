# VoltForge tokenizer releases

`vfdlm-byte-bpe-v1.0.0` is the first approved tokenizer for the VoltForge Domain Language Model. It remains immutable because inactive Gen1 artifacts bind its exact files.

`vfdlm-byte-bpe-v1.1.0` is the current approved tokenizer release, trained from scratch on the expanded 11-board, compiler-verified VFAI-009 corpus. It is not the runtime default until a separately governed Gen1 retrain and release evaluation accepts a model artifact bound to its exact manifest.

The fixed contract preserves `[PAD]=0`, `[UNK]=1`, `[BOS]=2`, and `[EOS]=3`; reserves typed task-role tokens; and maps every byte to a base token. Ordinary text never interprets literal special-token spellings unless the caller explicitly enables special parsing. UTF-8 with `surrogatepass` provides exact round trips for valid Unicode and lone surrogate code points, while byte APIs preserve arbitrary binary sequences.

The release manifest binds the vocabulary, ordered merges, tokenizer configuration, evaluation report, trainer implementation, source IDs, approved shard IDs, and deterministic train/evaluation split. The model artifact registry requires this manifest as `tokenizerManifest`, verifies its checksum and internal file checksums, and rejects model/tokenizer lineage mismatches.

Build after deliberate review:

```powershell
python tools/build_tokenizer.py --write
```

Retrain and compare every expected byte without writing:

```powershell
python tools/build_tokenizer.py --check
```

The quarantined legacy tokenizer is loaded only by the comparison evaluator after new-tokenizer training. It is never a vocabulary seed, merge seed, runtime fallback, or training input.
