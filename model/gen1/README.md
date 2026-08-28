# VoltForge Domain Language Model — Gen1 architecture

This package is the production architecture contract for the project-owned
VoltForge model. It does not download or load pretrained weights, pretrained
tokenizers, hosted inference clients, or third-party model APIs. PyTorch is used
only as the tensor/autograd framework.

The decoder uses pre-normalized RMSNorm blocks, rotary positional embeddings,
causal grouped-query self-attention, SwiGLU feed-forward layers, and tied token
embedding/output weights. KV caches retain only unexpanded key/value heads.

`Gen1Config()` is deliberately a 289,088-parameter executable contract model
using the released 3,072-token vocabulary. It is not the Gen1 release profile;
VFAI-013 must select that profile from measured quality and runtime sweeps.
Construction performs exact analytical parameter accounting before allocating.
The default 75,000,000-parameter allocation ceiling prevents old 1B-labelled
experiments or unreviewed server profiles from being instantiated accidentally.

Training and inference have separate fail-safe entry points:

```python
from model.gen1 import Gen1Config, VoltForgeGen1

model = VoltForgeGen1.for_training(Gen1Config())
model.save_checkpoint("path/to/new-empty-checkpoint-directory")
inference_model = VoltForgeGen1.from_checkpoint(
    "path/to/new-empty-checkpoint-directory"
)
```

`for_training()` materializes deterministic random weights from the config seed.
`from_checkpoint()` validates the manifest, file checksums, architecture,
configuration fingerprint, parameter formula, tensor names, shapes, dtypes, and
tied weights before materializing an inference model. Missing weights are never
initialized.

## Reproducible Windows CPU test environment

Newer PyTorch Windows wheels currently reproduce upstream `c10.dll` WinError
1114 on the benchmark machine. The verified VFAI-011 environment uses isolated
CPython 3.12 and the CPU-only PyTorch 2.8.0 wheel:

```powershell
uv venv .toolchains/gen1 --python 3.12
uv pip install --python .toolchains/gen1/Scripts/python.exe `
  --index-url https://download.pytorch.org/whl/cpu `
  -r requirements-gen1.txt
uv pip install --python .toolchains/gen1/Scripts/python.exe `
  -r requirements.txt -r requirements-dev.txt
.toolchains/gen1/Scripts/python.exe -m pytest tests/test_gen1_model.py -q
```

`.toolchains/` is ignored and must never be committed as a model artifact.
